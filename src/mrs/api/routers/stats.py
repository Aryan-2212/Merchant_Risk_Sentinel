"""Aggregate read-only endpoints for the dashboard's Command Center (Dev Plan §21
View 1; Phase 8 Step 7 redesign).

Every value here is a COUNT/GROUP BY/SUM over already-persisted rows, or a bounded
read of real relationships already implied by the transactions table -- this module
computes no risk, behavioral, or policy logic itself; it only summarizes what
mrs.risk/mrs.behavioral/mrs.policy already decided and mrs.db.populate/
mrs.policy.engine already persisted. Added because the alternative -- paginating
through hundreds of thousands of rows client-side to derive these -- would violate
Dev Plan Sec 31 (no expensive client-side aggregation, no loading the full dataset
into the browser).
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from mrs.api import schemas
from mrs.api.deps import get_db
from mrs.api.lookups import entity_deviation_rates
from mrs.db.models import Alert, Customer, RiskScore, Terminal, Transaction
from mrs.risk.aggregate import CRITICAL, HIGH, HIGH_RISK, RISK_RISING

router = APIRouter(prefix="/stats", tags=["stats"])

_AT_RISK_STATES = (RISK_RISING, HIGH_RISK)


def _counts(db: Session, column) -> dict[str, int]:
    rows = db.execute(select(column, func.count()).group_by(column)).all()
    return {str(key): count for key, count in rows}


def _latest_customer_states(db: Session) -> list[tuple[int, str | None, int | None]]:
    """One row per customer: its MOST RECENT (by tx_datetime) customer_risk_state --
    a temporal snapshot, never a permanent label (Dev Plan Sec 8)."""
    rows = db.execute(
        text(
            """
            SELECT DISTINCT ON (rs.customer_id)
                rs.customer_id, rs.customer_risk_state, rs.customer_risk_severity
            FROM risk_scores rs
            JOIN transactions t ON t.transaction_id = rs.transaction_id
            ORDER BY rs.customer_id, t.tx_datetime DESC
            """
        )
    ).all()
    return [(r[0], r[1], r[2]) for r in rows]


def _latest_terminal_states(db: Session) -> list[tuple[int, str | None, int | None]]:
    rows = db.execute(
        text(
            """
            SELECT DISTINCT ON (rs.terminal_id)
                rs.terminal_id, rs.terminal_risk_state, rs.terminal_risk_severity
            FROM risk_scores rs
            JOIN transactions t ON t.transaction_id = rs.transaction_id
            ORDER BY rs.terminal_id, t.tx_datetime DESC
            """
        )
    ).all()
    return [(r[0], r[1], r[2]) for r in rows]


@router.get("/overview", response_model=schemas.OverviewStats)
def get_overview_stats(db: Session = Depends(get_db)) -> schemas.OverviewStats:
    total_transactions = db.execute(select(func.count()).select_from(Transaction)).scalar_one()
    total_customers = db.execute(select(func.count()).select_from(Customer)).scalar_one()
    total_terminals = db.execute(select(func.count()).select_from(Terminal)).scalar_one()
    total_risk_scores = db.execute(select(func.count()).select_from(RiskScore)).scalar_one()
    total_alerts = db.execute(select(func.count()).select_from(Alert)).scalar_one()

    customer_states = _latest_customer_states(db)
    terminal_states = _latest_terminal_states(db)
    customers_at_risk = sum(1 for _, state, _ in customer_states if state in _AT_RISK_STATES)
    terminals_at_risk = sum(1 for _, state, _ in terminal_states if state in _AT_RISK_STATES)

    risk_exposure = db.execute(
        select(func.coalesce(func.sum(Transaction.tx_amount), 0.0))
        .select_from(RiskScore)
        .join(Transaction, Transaction.transaction_id == RiskScore.transaction_id)
        .where(RiskScore.unified_risk_level.in_([HIGH, CRITICAL]))
    ).scalar_one()

    return schemas.OverviewStats(
        total_transactions=total_transactions,
        total_customers=total_customers,
        total_terminals=total_terminals,
        total_risk_scores=total_risk_scores,
        total_alerts=total_alerts,
        risk_level_counts=_counts(db, RiskScore.unified_risk_level),
        alert_action_counts=_counts(db, Alert.recommended_action),
        alert_status_counts=_counts(db, Alert.status),
        customers_at_risk=customers_at_risk,
        terminals_at_risk=terminals_at_risk,
        risk_exposure_amount=float(risk_exposure),
    )


@router.get("/risk-activity", response_model=list[schemas.RiskActivityPoint])
def get_risk_activity(
    days: int = Query(30, ge=1, le=183),
    db: Session = Depends(get_db),
) -> list[schemas.RiskActivityPoint]:
    """Daily severity-2 ("elevated") counts per component, for the most recent `days`
    days of data actually present (not wall-clock "today" -- this is a frozen
    historical dataset, so "recent" is relative to the data's own max tx_datetime,
    matching how Replay already frames its position)."""
    rows = db.execute(
        text(
            """
            WITH bounds AS (SELECT max(tx_datetime) AS max_dt FROM transactions)
            SELECT
                date_trunc('day', t.tx_datetime) AS day,
                sum((rs.transaction_risk_severity = 2)::int) AS transaction_high,
                sum((rs.customer_risk_severity = 2)::int) AS customer_high,
                sum((rs.terminal_risk_severity = 2)::int) AS terminal_high,
                sum((rs.unified_risk_level IN ('HIGH', 'CRITICAL'))::int) AS elevated_transactions,
                count(*) AS total_scored
            FROM risk_scores rs
            JOIN transactions t ON t.transaction_id = rs.transaction_id, bounds
            WHERE t.tx_datetime > bounds.max_dt - (:days || ' days')::interval
            GROUP BY 1
            ORDER BY 1
            """
        ),
        {"days": days},
    ).all()
    return [
        schemas.RiskActivityPoint(
            date=row.day.date(),
            transaction_high=row.transaction_high,
            customer_high=row.customer_high,
            terminal_high=row.terminal_high,
            elevated_transactions=row.elevated_transactions,
            total_scored=row.total_scored,
        )
        for row in rows
    ]


@router.get("/recent-activity", response_model=list[schemas.ReplayItemOut])
def get_recent_activity(
    limit: int = Query(20, ge=1, le=100),
    levels: str | None = Query(
        None,
        description="Comma-separated unified_risk_level values to restrict to, e.g. 'HIGH,CRITICAL'. "
        "Unfiltered by default.",
    ),
    db: Session = Depends(get_db),
) -> list[schemas.ReplayItemOut]:
    """The most recently (chronologically last) scored transactions, across every
    risk level by default -- not alerts-only, so a quiet ALLOW/LOW transaction is
    visible in the feed too, matching what actually happened rather than only what
    escalated. Reverse of GET /replay/transactions' ordering (Dev Plan Sec 22's replay
    stream is oldest-first by design); this is index-bounded (ORDER BY tx_datetime DESC
    LIMIT n on an indexed column), not a full-table scan.

    `levels` exists because elevated transactions are a small minority of the stream
    (~1.5%) -- a client that wants "the last N HIGH/CRITICAL events" (e.g. the Command
    Center's Recent High Risk panel) needs that filtered server-side; sampling the last
    N transactions overall and filtering client-side can legitimately return zero rows.
    """
    stmt = (
        select(Transaction, RiskScore, Alert)
        .join(RiskScore, RiskScore.transaction_id == Transaction.transaction_id)
        .outerjoin(Alert, Alert.transaction_id == Transaction.transaction_id)
    )
    if levels:
        wanted = [lvl.strip().upper() for lvl in levels.split(",") if lvl.strip()]
        stmt = stmt.where(RiskScore.unified_risk_level.in_(wanted))
    stmt = stmt.order_by(Transaction.tx_datetime.desc(), Transaction.transaction_id.desc()).limit(limit)
    rows = db.execute(stmt).all()
    return [
        schemas.ReplayItemOut(
            transaction=schemas.TransactionOut.model_validate(tx),
            risk_score=schemas.RiskScoreOut.model_validate(risk),
            alert=schemas.AlertSummaryOut.model_validate(alert) if alert is not None else None,
        )
        for tx, risk, alert in rows
    ]


_NEIGHBOR_LIMIT = 8
_MAX_FOCUS_PER_TYPE = 2


def _top_focus_terminals(db: Session, limit: int) -> list[tuple[int, str, int]]:
    rows = db.execute(
        text(
            """
            SELECT terminal_id, terminal_risk_state, terminal_risk_severity FROM (
                SELECT DISTINCT ON (rs.terminal_id)
                    rs.terminal_id, rs.terminal_risk_state, rs.terminal_risk_severity, t.tx_datetime
                FROM risk_scores rs
                JOIN transactions t ON t.transaction_id = rs.transaction_id
                ORDER BY rs.terminal_id, t.tx_datetime DESC
            ) latest
            WHERE terminal_risk_state IN ('HIGH_RISK', 'RISK_RISING')
            ORDER BY terminal_risk_severity DESC, tx_datetime DESC
            LIMIT :limit
            """
        ),
        {"limit": limit},
    ).all()
    return [(r.terminal_id, r.terminal_risk_state, r.terminal_risk_severity) for r in rows]


def _top_focus_customers(db: Session, limit: int) -> list[tuple[int, str, int]]:
    rows = db.execute(
        text(
            """
            SELECT customer_id, customer_risk_state, customer_risk_severity FROM (
                SELECT DISTINCT ON (rs.customer_id)
                    rs.customer_id, rs.customer_risk_state, rs.customer_risk_severity, t.tx_datetime
                FROM risk_scores rs
                JOIN transactions t ON t.transaction_id = rs.transaction_id
                ORDER BY rs.customer_id, t.tx_datetime DESC
            ) latest
            WHERE customer_risk_state IN ('HIGH_RISK', 'RISK_RISING')
            ORDER BY customer_risk_severity DESC, tx_datetime DESC
            LIMIT :limit
            """
        ),
        {"limit": limit},
    ).all()
    return [(r.customer_id, r.customer_risk_state, r.customer_risk_severity) for r in rows]


def _terminal_neighbors(db: Session, terminal_id: int, limit: int) -> list[tuple[int, int]]:
    """Real customers who transacted with this terminal, most recent first -- an
    actual relationship (Dev Plan: "do not fabricate relationships"), never inferred."""
    rows = db.execute(
        text(
            """
            SELECT customer_id, count(*) AS weight, max(tx_datetime) AS last_tx
            FROM transactions
            WHERE terminal_id = :tid
            GROUP BY customer_id
            ORDER BY last_tx DESC
            LIMIT :limit
            """
        ),
        {"tid": terminal_id, "limit": limit},
    ).all()
    return [(r.customer_id, r.weight) for r in rows]


def _customer_neighbors(db: Session, customer_id: int, limit: int) -> list[tuple[int, int]]:
    rows = db.execute(
        text(
            """
            SELECT terminal_id, count(*) AS weight, max(tx_datetime) AS last_tx
            FROM transactions
            WHERE customer_id = :cid
            GROUP BY terminal_id
            ORDER BY last_tx DESC
            LIMIT :limit
            """
        ),
        {"cid": customer_id, "limit": limit},
    ).all()
    return [(r.terminal_id, r.weight) for r in rows]


def _states_for_customers(db: Session, ids: list[int]) -> dict[int, tuple[str | None, int | None]]:
    if not ids:
        return {}
    rows = db.execute(
        text(
            """
            SELECT DISTINCT ON (rs.customer_id) rs.customer_id, rs.customer_risk_state, rs.customer_risk_severity
            FROM risk_scores rs
            JOIN transactions t ON t.transaction_id = rs.transaction_id
            WHERE rs.customer_id = ANY(:ids)
            ORDER BY rs.customer_id, t.tx_datetime DESC
            """
        ),
        {"ids": ids},
    ).all()
    return {r.customer_id: (r.customer_risk_state, r.customer_risk_severity) for r in rows}


def _states_for_terminals(db: Session, ids: list[int]) -> dict[int, tuple[str | None, int | None]]:
    if not ids:
        return {}
    rows = db.execute(
        text(
            """
            SELECT DISTINCT ON (rs.terminal_id) rs.terminal_id, rs.terminal_risk_state, rs.terminal_risk_severity
            FROM risk_scores rs
            JOIN transactions t ON t.transaction_id = rs.transaction_id
            WHERE rs.terminal_id = ANY(:ids)
            ORDER BY rs.terminal_id, t.tx_datetime DESC
            """
        ),
        {"ids": ids},
    ).all()
    return {r.terminal_id: (r.terminal_risk_state, r.terminal_risk_severity) for r in rows}


def _live_window_network(db: Session, live_window: int) -> schemas.NetworkGraph:
    """Rolling recent-activity graph: nodes/edges derived ONLY from the most recent
    `live_window` transactions -- real customer<->terminal pairs from those exact
    rows, never the full history. Reuses _states_for_customers/_states_for_terminals
    (the same behavioral-state lookups the default graph mode already uses) rather
    than inventing a second way to read current state."""
    window_rows = db.execute(
        select(Transaction.transaction_id, Transaction.customer_id, Transaction.terminal_id)
        .order_by(Transaction.tx_datetime.desc(), Transaction.transaction_id.desc())
        .limit(live_window)
    ).all()

    if not window_rows:
        return schemas.NetworkGraph(nodes=[], edges=[], focus_ids=[], latest_transaction_id=None)

    latest_transaction_id = window_rows[0].transaction_id
    newest_customer_id = window_rows[0].customer_id
    newest_terminal_id = window_rows[0].terminal_id

    pair_weights: dict[tuple[int, int], int] = {}
    customer_ids: set[int] = set()
    terminal_ids: set[int] = set()
    for row in window_rows:
        customer_ids.add(row.customer_id)
        terminal_ids.add(row.terminal_id)
        key = (row.customer_id, row.terminal_id)
        pair_weights[key] = pair_weights.get(key, 0) + 1

    cust_states = _states_for_customers(db, list(customer_ids))
    term_states = _states_for_terminals(db, list(terminal_ids))

    nodes: dict[str, schemas.NetworkNode] = {}
    for cid in customer_ids:
        node_id = f"customer:{cid}"
        state, severity = cust_states.get(cid, (None, None))
        nodes[node_id] = schemas.NetworkNode(
            id=node_id,
            entity_type="customer",
            entity_id=cid,
            risk_state=state,
            risk_severity=severity,
            is_focus=cid == newest_customer_id,
        )
    for tid in terminal_ids:
        node_id = f"terminal:{tid}"
        state, severity = term_states.get(tid, (None, None))
        nodes[node_id] = schemas.NetworkNode(
            id=node_id,
            entity_type="terminal",
            entity_id=tid,
            risk_state=state,
            risk_severity=severity,
            is_focus=tid == newest_terminal_id,
        )

    edges = [
        schemas.NetworkEdge(source=f"customer:{cid}", target=f"terminal:{tid}", weight=w)
        for (cid, tid), w in pair_weights.items()
    ]
    focus_ids = [f"customer:{newest_customer_id}", f"terminal:{newest_terminal_id}"]

    return schemas.NetworkGraph(nodes=list(nodes.values()), edges=edges, focus_ids=focus_ids, latest_transaction_id=latest_transaction_id)


@router.get("/network", response_model=schemas.NetworkGraph)
def get_entity_network(
    focus_type: str | None = Query(None, pattern="^(customer|terminal)$"),
    focus_id: int | None = Query(None),
    live_window: int | None = Query(
        None,
        ge=1,
        le=500,
        description="Restrict the graph to the most recent N transactions overall "
        "(by tx_datetime desc, tie-broken by transaction_id desc) -- reuses the same "
        "'last N transactions' convention GET /stats/recent-activity already uses. "
        "This is the Simulated Recent Operational Stream's rolling live-operations "
        "view (mrs.data.recent_stream / mrs.live.simulate): nodes are every customer/ "
        "terminal that appears in that window, edges are their real transaction "
        "counts within it, and the response's latest_transaction_id names the single "
        "newest one so a client can highlight it as newly arrived. Takes precedence "
        "over focus_type/focus_id when given (a different mode, not combined with "
        "the investigation view in this first cut).",
    ),
    db: Session = Depends(get_db),
) -> schemas.NetworkGraph:
    """A real, bounded neighborhood graph (Dev Plan: signature Entity Risk Network).

    Default (no params): the 1-2 most severe currently at-risk terminals plus the
    1-2 most severe currently at-risk customers become "focus" hubs. Passing
    focus_type/focus_id centers the graph on one specific entity instead (the
    Command Center's click-to-investigate interaction). Passing live_window switches
    to the rolling recent-activity view instead of either (see that parameter's own
    description). Every edge is a real customer<->terminal pair derived from actual
    transactions -- nothing inferred, in any mode.
    """
    if live_window is not None:
        return _live_window_network(db, live_window)

    focus_terminals: list[tuple[int, str, int]] = []
    focus_customers: list[tuple[int, str, int]] = []

    if focus_type == "terminal" and focus_id is not None:
        states = _states_for_terminals(db, [focus_id])
        state, severity = states.get(focus_id, (None, None))
        focus_terminals = [(focus_id, state, severity)]
    elif focus_type == "customer" and focus_id is not None:
        states = _states_for_customers(db, [focus_id])
        state, severity = states.get(focus_id, (None, None))
        focus_customers = [(focus_id, state, severity)]
    else:
        focus_terminals = _top_focus_terminals(db, _MAX_FOCUS_PER_TYPE)
        focus_customers = _top_focus_customers(db, _MAX_FOCUS_PER_TYPE)

    nodes: dict[str, schemas.NetworkNode] = {}
    edges: list[schemas.NetworkEdge] = []
    focus_ids: list[str] = []

    for tid, state, severity in focus_terminals:
        node_id = f"terminal:{tid}"
        nodes[node_id] = schemas.NetworkNode(
            id=node_id, entity_type="terminal", entity_id=tid, risk_state=state, risk_severity=severity, is_focus=True
        )
        focus_ids.append(node_id)
        for cust_id, weight in _terminal_neighbors(db, tid, _NEIGHBOR_LIMIT):
            edges.append(schemas.NetworkEdge(source=f"customer:{cust_id}", target=node_id, weight=weight))

    for cid, state, severity in focus_customers:
        node_id = f"customer:{cid}"
        nodes[node_id] = schemas.NetworkNode(
            id=node_id, entity_type="customer", entity_id=cid, risk_state=state, risk_severity=severity, is_focus=True
        )
        focus_ids.append(node_id)
        for term_id, weight in _customer_neighbors(db, cid, _NEIGHBOR_LIMIT):
            edges.append(schemas.NetworkEdge(source=node_id, target=f"terminal:{term_id}", weight=weight))

    neighbor_customer_ids = {int(e.source.split(":")[1]) for e in edges if e.source.startswith("customer:")}
    neighbor_customer_ids -= {c for c, _, _ in focus_customers}
    neighbor_terminal_ids = {int(e.target.split(":")[1]) for e in edges if e.target.startswith("terminal:")}
    neighbor_terminal_ids -= {t for t, _, _ in focus_terminals}

    cust_states = _states_for_customers(db, list(neighbor_customer_ids))
    for cid in neighbor_customer_ids:
        node_id = f"customer:{cid}"
        state, severity = cust_states.get(cid, (None, None))
        nodes.setdefault(
            node_id,
            schemas.NetworkNode(
                id=node_id, entity_type="customer", entity_id=cid, risk_state=state, risk_severity=severity, is_focus=False
            ),
        )

    term_states = _states_for_terminals(db, list(neighbor_terminal_ids))
    for tid in neighbor_terminal_ids:
        node_id = f"terminal:{tid}"
        state, severity = term_states.get(tid, (None, None))
        nodes.setdefault(
            node_id,
            schemas.NetworkNode(
                id=node_id, entity_type="terminal", entity_id=tid, risk_state=state, risk_severity=severity, is_focus=False
            ),
        )

    return schemas.NetworkGraph(nodes=list(nodes.values()), edges=edges, focus_ids=focus_ids)


def _terminals_at_risk_candidates(db: Session, limit: int) -> list[tuple[int, str, int, dt.datetime]]:
    rows = db.execute(
        text(
            """
            SELECT terminal_id, terminal_risk_state, terminal_risk_severity, tx_datetime FROM (
                SELECT DISTINCT ON (rs.terminal_id)
                    rs.terminal_id, rs.terminal_risk_state, rs.terminal_risk_severity, t.tx_datetime
                FROM risk_scores rs
                JOIN transactions t ON t.transaction_id = rs.transaction_id
                ORDER BY rs.terminal_id, t.tx_datetime DESC
            ) latest
            WHERE terminal_risk_state IN ('HIGH_RISK', 'RISK_RISING')
            ORDER BY terminal_risk_severity DESC, tx_datetime DESC
            LIMIT :limit
            """
        ),
        {"limit": limit},
    ).all()
    return [(r.terminal_id, r.terminal_risk_state, r.terminal_risk_severity, r.tx_datetime) for r in rows]


@router.get("/terminals-at-risk", response_model=list[schemas.EntityAtRiskRow])
def get_terminals_at_risk(
    limit: int = Query(8, ge=1, le=25),
    db: Session = Depends(get_db),
) -> list[schemas.EntityAtRiskRow]:
    """Currently elevated terminals (Dev Plan Sec 15 View 2), ranked by severity, each
    with a real recent-vs-baseline behavioral deviation -- the "High Deviation
    Terminals" Command Center panel. Never uses tx_fraud (ground truth); current_rate/
    baseline_rate are computed from this system's own terminal_risk_severity."""
    candidates = _terminals_at_risk_candidates(db, limit)
    rates = entity_deviation_rates(db, "terminal", [c[0] for c in candidates])

    return [
        schemas.EntityAtRiskRow(
            entity_type="terminal",
            entity_id=tid,
            risk_state=state,
            risk_severity=severity,
            current_rate=rates.get(tid, {}).get("current_rate", 0.0),
            baseline_rate=rates.get(tid, {}).get("baseline_rate"),
            recent_transaction_count=int(rates.get(tid, {}).get("current_count", 0)),
            last_activity=last_activity,
        )
        for tid, state, severity, last_activity in candidates
    ]
