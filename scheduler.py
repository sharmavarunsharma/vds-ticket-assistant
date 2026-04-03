"""
scheduler.py — Automated task scheduling using APScheduler
Runs weekly report generation every Friday at a specified time.
Safe to import — does not auto-start scheduler unless explicitly called.
"""

import os
import threading
from datetime import datetime
from typing import Optional, Callable

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    print("⚠️  APScheduler not installed. Scheduled jobs disabled.")

# ─── Global Scheduler Instance ─────────────────────────────────────────────────

_scheduler: Optional[object] = None
_scheduler_lock = threading.Lock()


# ─── Scheduler Setup ───────────────────────────────────────────────────────────

def start_scheduler(report_callback: Callable, report_kwargs: dict = None):
    """
    Start the background scheduler with a weekly Friday report job.
    
    Args:
        report_callback: Function to call for report generation
        report_kwargs: Keyword arguments to pass to the callback
    """
    global _scheduler

    if not SCHEDULER_AVAILABLE:
        return {"status": "error", "message": "APScheduler not installed"}

    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            return {"status": "already_running", "message": "Scheduler already active"}

        _scheduler = BackgroundScheduler()

        # Every Friday at 18:00 IST (12:30 UTC)
        _scheduler.add_job(
            func=_run_report_job,
            trigger=CronTrigger(
                day_of_week="fri",
                hour=12,
                minute=30,
                timezone="UTC"
            ),
            kwargs={"callback": report_callback, "kwargs": report_kwargs or {}},
            id="weekly_report",
            name="Weekly Ticket Report (Fridays)",
            replace_existing=True,
            misfire_grace_time=3600  # 1 hour grace if server was down
        )

        _scheduler.start()

        return {
            "status": "started",
            "message": "Scheduler started — weekly report every Friday at 18:00 IST",
            "next_run": get_next_run_time()
        }


def stop_scheduler():
    """Gracefully stop the scheduler."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
            _scheduler = None
            return {"status": "stopped"}
        return {"status": "not_running"}


def get_scheduler_status() -> dict:
    """Return current scheduler status and next run info."""
    if not SCHEDULER_AVAILABLE:
        return {"running": False, "reason": "APScheduler not installed"}

    if _scheduler is None or not _scheduler.running:
        return {"running": False, "next_run": None}

    return {
        "running": True,
        "next_run": get_next_run_time(),
        "jobs": [
            {"id": j.id, "name": j.name, "next_run": str(j.next_run_time)}
            for j in _scheduler.get_jobs()
        ]
    }


def get_next_run_time() -> Optional[str]:
    """Get the next scheduled run time as a string."""
    if not _scheduler or not _scheduler.running:
        return None
    try:
        jobs = _scheduler.get_jobs()
        if jobs:
            return str(jobs[0].next_run_time)
    except Exception:
        pass
    return None


def _run_report_job(callback: Callable, kwargs: dict):
    """Internal job runner with error handling."""
    try:
        print(f"[SCHEDULER] Running weekly report job at {datetime.now()}")
        callback(**kwargs)
        print(f"[SCHEDULER] Weekly report completed at {datetime.now()}")
    except Exception as e:
        print(f"[SCHEDULER] Error in weekly report job: {e}")


# ─── Manual Trigger ────────────────────────────────────────────────────────────

def run_now(callback: Callable, kwargs: dict = None) -> dict:
    """
    Manually trigger the report job immediately (useful for testing).
    """
    try:
        callback(**(kwargs or {}))
        return {"status": "success", "message": "Report generated successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─── Scheduler Info for UI ─────────────────────────────────────────────────────

def get_schedule_info() -> dict:
    """Return human-readable schedule information for display in UI."""
    return {
        "schedule": "Every Friday at 18:00 IST",
        "output_file": "weekly_report.txt",
        "timezone": "Asia/Kolkata (IST)",
        "description": "Generates weekly ticket analysis: totals, open/closed, top issues, suggested actions",
        "available": SCHEDULER_AVAILABLE
    }
