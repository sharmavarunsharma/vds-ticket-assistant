"""
rules.py — Rule-based ticket routing and escalation engine
Analyzes ticket text and returns routing decisions, priority flags, and suggested actions.
"""

import re
from typing import Optional


# ─── Rule Definitions ──────────────────────────────────────────────────────────

RULES = [
    # Format: (pattern, team, action, priority, color, icon)
    {
        "keywords": ["vpn", "virtual private network", "cisco vpn", "pulse secure", "globalprotect", "anyconnect"],
        "team": "Network Team",
        "action": "Route to Network Team for VPN configuration/troubleshooting",
        "priority": "Medium",
        "color": "#3b82f6",  # blue
        "icon": "🌐",
        "queue": "NETWORK-QUEUE",
        "sla": "4 hours"
    },
    {
        "keywords": ["ssh", "secure shell", "putty", "openssh", "ssh key", "ssh tunnel", "ssh access", "port 22"],
        "team": "Infra Team",
        "action": "Route to Infra Team for SSH/server access issues",
        "priority": "Medium",
        "color": "#8b5cf6",  # purple
        "icon": "🖥️",
        "queue": "INFRA-QUEUE",
        "sla": "4 hours"
    },
    {
        "keywords": ["urgent", "critical", "production down", "p1", "blocker", "outage", "down", "not working", "broken", "emergency", "asap", "immediately"],
        "team": "On-Call / L2 Escalation",
        "action": "ESCALATE IMMEDIATELY — Production impact detected",
        "priority": "Highest",
        "color": "#ef4444",  # red
        "icon": "🚨",
        "queue": "ESCALATION-QUEUE",
        "sla": "30 minutes"
    },
    {
        "keywords": ["aws", "ec2", "s3", "lambda", "cloud", "terraform", "kubernetes", "k8s", "docker", "container"],
        "team": "Cloud/DevOps Team",
        "action": "Route to Cloud/DevOps Team for infrastructure issues",
        "priority": "Medium",
        "color": "#f59e0b",  # amber
        "icon": "☁️",
        "queue": "DEVOPS-QUEUE",
        "sla": "4 hours"
    },
    {
        "keywords": ["database", "db", "sql", "postgres", "mysql", "oracle", "mongo", "redis", "query", "slow query", "connection pool"],
        "team": "Database Team",
        "action": "Route to Database Team for DB performance/access issues",
        "priority": "Medium",
        "color": "#06b6d4",  # cyan
        "icon": "🗄️",
        "queue": "DB-QUEUE",
        "sla": "4 hours"
    },
    {
        "keywords": ["permission", "access denied", "unauthorized", "403", "401", "role", "ldap", "active directory", "ad group", "iam"],
        "team": "Identity & Access Management (IAM) Team",
        "action": "Route to IAM Team for access/permission issues",
        "priority": "Medium",
        "color": "#10b981",  # emerald
        "icon": "🔐",
        "queue": "IAM-QUEUE",
        "sla": "8 hours"
    },
    {
        "keywords": ["high", "p2", "important", "major", "severe"],
        "team": "L2 Support",
        "action": "Escalate to L2 — High priority ticket",
        "priority": "High",
        "color": "#f97316",  # orange
        "icon": "⚠️",
        "queue": "L2-QUEUE",
        "sla": "2 hours"
    },
]

# Default fallback rule
DEFAULT_RULE = {
    "team": "General Support Queue",
    "action": "Assign to general queue — no specific routing rule matched",
    "priority": "Low",
    "color": "#6b7280",  # gray
    "icon": "📋",
    "queue": "GENERAL-QUEUE",
    "sla": "1 business day"
}


# ─── Rule Engine ───────────────────────────────────────────────────────────────

def apply_rules(ticket_text: str) -> dict:
    """
    Apply rule engine to ticket text.
    Returns routing decision with team, action, priority, and SLA.
    """
    if not ticket_text:
        return {**DEFAULT_RULE, "matched_keywords": [], "all_matches": []}

    text_lower = ticket_text.lower()
    matched_rules = []

    for rule in RULES:
        matched_kw = [kw for kw in rule["keywords"] if kw in text_lower]
        if matched_kw:
            matched_rules.append({**rule, "matched_keywords": matched_kw})

    if not matched_rules:
        return {**DEFAULT_RULE, "matched_keywords": [], "all_matches": []}

    # Sort by priority: Highest > High > Medium > Low
    priority_order = {"Highest": 0, "High": 1, "Medium": 2, "Low": 3}
    matched_rules.sort(key=lambda r: priority_order.get(r["priority"], 99))

    # Return highest priority match
    best = matched_rules[0]
    return {
        "team": best["team"],
        "action": best["action"],
        "priority": best["priority"],
        "color": best["color"],
        "icon": best["icon"],
        "queue": best["queue"],
        "sla": best["sla"],
        "matched_keywords": best["matched_keywords"],
        "all_matches": [
            {"team": r["team"], "keywords": r["matched_keywords"], "priority": r["priority"]}
            for r in matched_rules[1:]  # Secondary matches
        ]
    }


def get_priority_badge(priority: str) -> str:
    """Return a color-coded priority label for display."""
    badges = {
        "Highest": "🔴 HIGHEST",
        "High": "🟠 HIGH",
        "Medium": "🟡 MEDIUM",
        "Low": "🟢 LOW"
    }
    return badges.get(priority, "⚪ UNKNOWN")


def generate_routing_comment(routing: dict, ticket_summary: str) -> str:
    """Generate a ready-to-paste Jira routing comment."""
    kw_str = ", ".join(routing.get("matched_keywords", []))
    secondary = routing.get("all_matches", [])
    secondary_str = ""
    if secondary:
        secondary_str = "\n\nℹ️ *Additional routing signals detected:*\n" + "\n".join(
            [f"  - {m['team']} (keywords: {', '.join(m['keywords'])})" for m in secondary]
        )

    return f"""🤖 *[AUTO-ROUTING ENGINE]*

Ticket: _{ticket_summary[:100]}_

*Routing Decision:* {routing['icon']} Assigned to *{routing['team']}*
*Priority:* {get_priority_badge(routing['priority'])}
*Queue:* {routing['queue']}
*SLA:* Resolve within {routing['sla']}

*Reason:* Keywords detected → _{kw_str if kw_str else 'Default routing'}_
*Suggested Action:* {routing['action']}{secondary_str}

_This comment was generated automatically by the VDS Ticket Assistant._"""
