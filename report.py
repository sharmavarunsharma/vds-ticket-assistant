"""
report.py — Weekly report generator
Analyzes ticket DataFrame and generates structured weekly summary.
Can be triggered manually or automatically on Fridays.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd


# ─── Report Generation ─────────────────────────────────────────────────────────

def generate_weekly_report(df: pd.DataFrame, output_path: str = "weekly_report.txt") -> dict:
    """
    Generate a weekly report from the ticket DataFrame.
    Returns a report dict and saves a text file.
    """
    now = datetime.now()
    week_start = now - timedelta(days=7)

    # ── Ticket counts ──────────────────────────────────────────────────────────
    total = len(df)

    # Status analysis
    status_col = "status" if "status" in df.columns else None
    open_tickets = 0
    closed_tickets = 0
    in_progress = 0

    if status_col:
        status_lower = df[status_col].str.lower().fillna("")
        closed_statuses = ["done", "closed", "resolved", "completed", "fixed"]
        open_statuses = ["open", "new", "to do", "backlog", "reopened"]
        inprog_statuses = ["in progress", "in review", "investigation", "waiting"]

        closed_tickets = status_lower.isin(closed_statuses).sum()
        open_tickets = status_lower.isin(open_statuses).sum()
        in_progress = status_lower.isin(inprog_statuses).sum()

        # Fallback: anything not closed is open
        if closed_tickets + open_tickets + in_progress == 0:
            closed_tickets = status_lower.str.contains("close|done|resolv|fix", na=False).sum()
            open_tickets = total - closed_tickets
    else:
        open_tickets = total  # Assume all open if no status column

    # Resolution rate
    resolution_rate = round((closed_tickets / total * 100), 1) if total > 0 else 0

    # ── Priority breakdown ─────────────────────────────────────────────────────
    priority_breakdown = {}
    if "priority" in df.columns:
        priority_breakdown = df["priority"].value_counts().to_dict()

    # ── Top issues by keyword frequency ───────────────────────────────────────
    all_text = " ".join(
        (df.get("summary", pd.Series([])).fillna("") + " " +
         df.get("description", pd.Series([])).fillna("")).tolist()
    ).lower()

    # Common IT issue categories
    issue_categories = {
        "VPN Issues": ["vpn", "virtual private network", "cisco vpn"],
        "SSH/Access": ["ssh", "access denied", "permission", "unauthorized"],
        "Cloud/AWS": ["aws", "ec2", "s3", "cloud", "terraform"],
        "Database": ["database", "db", "sql", "query", "connection"],
        "Performance": ["slow", "performance", "timeout", "latency", "hang"],
        "Installation": ["install", "setup", "configure", "deployment"],
        "Network": ["network", "dns", "firewall", "proxy", "connectivity"],
        "Authentication": ["login", "password", "ldap", "sso", "authentication"],
    }

    top_issues = {}
    for category, keywords in issue_categories.items():
        count = sum(all_text.count(kw) for kw in keywords)
        if count > 0:
            top_issues[category] = count

    # Sort by frequency
    top_issues = dict(sorted(top_issues.items(), key=lambda x: x[1], reverse=True))

    # ── Suggested Actions ─────────────────────────────────────────────────────
    suggested_actions = []

    if open_tickets > 10:
        suggested_actions.append(f"⚠️ {open_tickets} open tickets — consider triaging and bulk-assigning to reduce backlog")

    if resolution_rate < 50:
        suggested_actions.append(f"📉 Resolution rate is {resolution_rate}% — review blockers and escalate stale tickets")

    if "VPN Issues" in top_issues and top_issues["VPN Issues"] > 3:
        suggested_actions.append("🌐 Multiple VPN issues detected — schedule a review with Network Team; consider runbook update")

    if "SSH/Access" in top_issues and top_issues["SSH/Access"] > 3:
        suggested_actions.append("🔐 Recurring access/SSH issues — audit IAM policies and SSH key rotation schedule")

    if "Performance" in top_issues and top_issues["Performance"] > 2:
        suggested_actions.append("🐢 Performance issues flagged — request infra capacity review")

    if not suggested_actions:
        suggested_actions.append("✅ No major patterns detected — operations appear stable this week")

    # ── Assemble report dict ───────────────────────────────────────────────────
    report = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "report_period": f"{week_start.strftime('%Y-%m-%d')} → {now.strftime('%Y-%m-%d')}",
        "total_tickets": total,
        "open_tickets": int(open_tickets),
        "closed_tickets": int(closed_tickets),
        "in_progress": int(in_progress),
        "resolution_rate": resolution_rate,
        "priority_breakdown": priority_breakdown,
        "top_issues": top_issues,
        "suggested_actions": suggested_actions,
    }

    # ── Write text report ──────────────────────────────────────────────────────
    _write_text_report(report, output_path)

    return report


def _write_text_report(report: dict, path: str):
    """Format and write the weekly report to a .txt file."""
    lines = []
    lines.append("=" * 65)
    lines.append("       VDS JIRA TICKET ASSISTANT — WEEKLY REPORT")
    lines.append("=" * 65)
    lines.append(f"  Generated : {report['generated_at']}")
    lines.append(f"  Period    : {report['report_period']}")
    lines.append("=" * 65)

    lines.append("\n📊 TICKET SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total Tickets      : {report['total_tickets']}")
    lines.append(f"  Open               : {report['open_tickets']}")
    lines.append(f"  Closed / Resolved  : {report['closed_tickets']}")
    lines.append(f"  In Progress        : {report['in_progress']}")
    lines.append(f"  Resolution Rate    : {report['resolution_rate']}%")

    if report.get("priority_breakdown"):
        lines.append("\n🏷️  PRIORITY BREAKDOWN")
        lines.append("-" * 40)
        for priority, count in report["priority_breakdown"].items():
            lines.append(f"  {priority:<20} : {count}")

    if report.get("top_issues"):
        lines.append("\n🔍 TOP ISSUE CATEGORIES")
        lines.append("-" * 40)
        for category, count in list(report["top_issues"].items())[:8]:
            bar = "█" * min(count, 20)
            lines.append(f"  {category:<22} : {bar} ({count})")

    lines.append("\n💡 SUGGESTED ACTIONS")
    lines.append("-" * 40)
    for action in report["suggested_actions"]:
        lines.append(f"  {action}")

    lines.append("\n" + "=" * 65)
    lines.append("  Report generated by VDS Jira Ticket Assistant")
    lines.append("  Ahead.com | EWS & DevOps ServiceDesk | VDS Project")
    lines.append("=" * 65)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print(f"Warning: Could not write report file: {e}")


def format_report_for_display(report: dict) -> str:
    """Return a markdown-formatted string for Streamlit display."""
    md = f"""### 📅 Report Period: `{report['report_period']}`
*Generated: {report['generated_at']}*

---

#### 📊 Ticket Summary
| Metric | Value |
|--------|-------|
| Total Tickets | **{report['total_tickets']}** |
| Open | 🔴 {report['open_tickets']} |
| Closed / Resolved | ✅ {report['closed_tickets']} |
| In Progress | 🔄 {report['in_progress']} |
| Resolution Rate | **{report['resolution_rate']}%** |
"""

    if report.get("priority_breakdown"):
        md += "\n---\n#### 🏷️ Priority Breakdown\n"
        for p, c in report["priority_breakdown"].items():
            md += f"- **{p}**: {c} tickets\n"

    if report.get("top_issues"):
        md += "\n---\n#### 🔍 Top Issue Categories\n"
        for cat, cnt in list(report["top_issues"].items())[:6]:
            md += f"- **{cat}**: {cnt} occurrences\n"

    if report.get("suggested_actions"):
        md += "\n---\n#### 💡 Suggested Actions\n"
        for action in report["suggested_actions"]:
            md += f"{action}\n\n"

    return md
