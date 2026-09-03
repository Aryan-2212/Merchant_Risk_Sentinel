/** Human-readable finding for a behavioral state -- shared by terminal and customer
 * pages since mrs.behavioral.terminal/customer use the identical five-state vocabulary
 * (NORMAL/RISK_RISING/HIGH_RISK/RECOVERY/INSUFFICIENT_HISTORY). Used both for
 * AlertDetail's per-signal "why risk increased" bullets (which only ever see
 * RISK_RISING/RECOVERY/HIGH_RISK, since those are the only states that can appear as a
 * contributing signal) and CustomerDetail's risk-summary banner (which must also
 * describe NORMAL/INSUFFICIENT_HISTORY gracefully, since most customers are calm). */
export function behavioralFinding(subject: "Terminal" | "Customer", state: string | null | undefined): string {
  switch (state) {
    case "HIGH_RISK":
      return `${subject} behavioral state is currently HIGH RISK.`;
    case "RISK_RISING":
      return `${subject} behavioral state is trending toward elevated risk (RISK RISING).`;
    case "RECOVERY":
      return `${subject} is in a post-incident recovery window, still classified as elevated risk.`;
    case "NORMAL":
      return `${subject} behavioral state is currently NORMAL. No elevated risk signals in the recent window.`;
    case "INSUFFICIENT_HISTORY":
      return `Insufficient transaction history to assess this ${subject.toLowerCase()}'s behavioral baseline yet.`;
    default:
      return `${subject} behavioral state: ${state ?? "unknown"}.`;
  }
}
