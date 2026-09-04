---
name: Merchant Risk Sentinel
colors:
  surface: '#111316'
  surface-dim: '#111316'
  surface-bright: '#37393d'
  surface-container-lowest: '#0c0e11'
  surface-container-low: '#1a1c1f'
  surface-container: '#1e2023'
  surface-container-high: '#282a2d'
  surface-container-highest: '#333538'
  on-surface: '#e2e2e6'
  on-surface-variant: '#c5c6cb'
  inverse-surface: '#e2e2e6'
  inverse-on-surface: '#2f3034'
  outline: '#8f9195'
  outline-variant: '#45474a'
  surface-tint: '#c4c6cd'
  primary: '#c4c6cd'
  on-primary: '#2d3136'
  primary-container: '#a4a7ad'
  on-primary-container: '#393d42'
  inverse-primary: '#5b5f64'
  secondary: '#c4c6cd'
  on-secondary: '#2e3036'
  secondary-container: '#46494f'
  on-secondary-container: '#b6b8bf'
  tertiary: '#d3c4b9'
  on-tertiary: '#382f27'
  tertiary-container: '#b2a49a'
  on-tertiary-container: '#443a33'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e0e2e9'
  primary-fixed-dim: '#c4c6cd'
  on-primary-fixed: '#181c20'
  on-primary-fixed-variant: '#43474c'
  secondary-fixed: '#e1e2e9'
  secondary-fixed-dim: '#c4c6cd'
  on-secondary-fixed: '#191c21'
  on-secondary-fixed-variant: '#44474c'
  tertiary-fixed: '#f0e0d5'
  tertiary-fixed-dim: '#d3c4b9'
  on-tertiary-fixed: '#221a13'
  on-tertiary-fixed-variant: '#4f453d'
  background: '#111316'
  on-background: '#e2e2e6'
  surface-variant: '#333538'
typography:
  score-display:
    fontFamily: IBM Plex Sans
    fontSize: 48px
    fontWeight: '600'
    lineHeight: 48px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: IBM Plex Sans
    fontSize: 24px
    fontWeight: '500'
    lineHeight: 32px
  headline-md:
    fontFamily: IBM Plex Sans
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 24px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-caps:
    fontFamily: IBM Plex Sans
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.06em
  data-mono:
    fontFamily: monospace
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  headline-lg-mobile:
    fontFamily: IBM Plex Sans
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 28px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  sidebar-width: 64px
  sidebar-expanded: 220px
  gutter: 1.5rem
  stack-compact: 0.5rem
  stack-default: 1rem
  container-padding: 2rem
---

## Brand & Style
The design system is engineered for high-stakes operational environments where clarity and focus are paramount. It follows a **Minimalist-Operational** aesthetic, prioritizing data density and rapid pattern recognition over decorative elements. 

The personality is clinical, objective, and authoritative. By utilizing a deep, monochromatic foundation, the UI recedes to the background, allowing evidence-based risk indicators to emerge with clarity. The emotional response is one of calm control, reducing the cognitive load on fraud analysts during high-velocity decision-making. 

Visual weight is distributed through tonal layering rather than aggressive borders or shadows. The design avoids the "SaaS-blue" trope, opting instead for a charcoal-based palette that reduces eye strain during long shifts in low-light environments.

## Colors
This design system utilizes a specialized "Obsidian Neutral" palette. The background is a specific charcoal (#0B0C0E), providing more depth and less "vibration" than pure black. 

**Semantic Application:**
- **Risk Tiers:** Colors are muted and desaturated. They should only be used for status indicators, risk bars, and data points—never for decorative UI flourishes or primary action buttons.
- **Action Hierarchy:** Primary actions use Soft White text on a muted gray background to maintain a low-profile aesthetic.
- **Borders:** Use `#292D33` for structural definition. Avoid high-contrast separators; rely on the tonal shift between surfaces whenever possible.

## Typography
The typographic system combines the utilitarian structure of **IBM Plex Sans** for headers and data labels with the high legibility of **Inter** for body content.

- **Risk Scores:** Use the `score-display` role for large numeric indicators. These should feel like instrumentation.
- **Metadata Labels:** All small supporting labels (e.g., table headers, timestamp prefixes) must use `label-caps` to distinguish them from actionable data.
- **Data Tables:** Use `data-mono` for transaction IDs, IP addresses, and terminal codes to ensure character alignment and fast scanning of alphanumeric strings.

## Layout & Spacing
The layout follows a **Fixed Sidebar + Fluid Workspace** model. The sidebar is intentionally narrow and "quiet" to keep the analyst's focus on the center of the screen.

**Structural Rules:**
- **Grid:** Use a 12-column grid for the main workspace. 
- **Dividers:** Use 1px solid lines (#292D33) instead of nested containers. Elements should feel like they are part of a single unified sheet rather than a collection of floating boxes.
- **Density:** High information density is encouraged. Vertical rhythm should be tight (8px increments), but horizontal margins between disparate data points should be generous to prevent "wall of text" fatigue.
- **Breakpoints:** 
  - Mobile (< 768px): Single column, hidden sidebar (drawer).
  - Tablet (768px - 1280px): Collapsed sidebar, 2-column data layout.
  - Desktop (> 1280px): 3-column analysis view (Entity | Timeline | Decision).

## Elevation & Depth
In this design system, depth is communicated through **Tonal Layering** rather than shadows. 

- **Level 0 (Background):** #0B0C0E – The base canvas.
- **Level 1 (Surface):** #111316 – The primary workspace area.
- **Level 2 (Panels):** #17191D – Detailed information modules or side-panels.
- **Level 3 (Pop-overs/Modals):** #1D2025 – Floating elements. These are the only components permitted to have a subtle, 20% opacity black shadow to provide separation from the workspace.

Avoid glassmorphism or blurs, as they introduce unnecessary visual noise and GPU overhead in data-heavy views.

## Shapes
The shape language is rigid and professional. We use a **Soft (4px)** corner radius for almost all components to maintain an organized, grid-like appearance while slightly softening the "brutalist" edge.

- **Interactive Elements:** Buttons and inputs use `rounded` (4px).
- **Cards/Containers:** Use `rounded-lg` (8px) sparingly for major section containers.
- **Network Nodes:** Use a strict geometric code. Customers/Entities are circles; Terminals/Hardware are squares. This allows analysts to distinguish entity types by silhouette alone.

## Components

**Buttons & Inputs:**
- **Primary Action:** Solid #A4A7AD background with #0B0C0E text. No gradients.
- **Secondary Action:** Ghost style with #292D33 border and #F2F2F0 text.
- **Inputs:** Darker background (#0B0C0E) with a subtle #292D33 border. On focus, the border color changes to #F2F2F0 (not blue).

**Data Visualization:**
- **Risk Bars:** Horizontal linear bars. Background is #292D33; the fill color corresponds to the semantic risk tier.
- **Timeline Markers:** A vertical 1px line with small circular nodes. Use the risk colors only for nodes that represent flagged events; all other events (logins, updates) should be neutral #6F737A.
- **Entity Network:** Use thin (1px) lines in #292D33. Active paths or selected connections should glow slightly with a 1px #F2F2F0 stroke.

**Decision Console:**
- The "AI Analyst" layer should be styled as a "System Log" or "Terminal Output." Use a monospace font and a subtly different background tint (#121418) to indicate it is a synthesized interpretation layer rather than raw database evidence.