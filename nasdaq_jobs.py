"""
NASDAQ Background Job Runner
Handles long-running scans with session persistence
"""

import os
import json
import tempfile
import threading
import uuid
import logging
from datetime import datetime, timezone, timedelta
from nasdaq_scan import build_scan_report

logger = logging.getLogger(__name__)

ET = timezone(timedelta(hours=-4))
JOB_DIR = os.path.join(tempfile.gettempdir(), "global_alpha_jobs")

# Ensure job directory exists
os.makedirs(JOB_DIR, exist_ok=True)


def _get_job_path(job_id: str) -> str:
    """Get the file path for a job."""
    return os.path.join(JOB_DIR, f"{job_id}.json")


def _read_job_file(job_id: str) -> dict:
    """Read job status file."""
    path = _get_job_path(job_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read job {job_id}: {e}")
        return {}


def _write_job_file(job_id: str, data: dict) -> None:
    """Write job status file."""
    path = _get_job_path(job_id)
    try:
        with open(path, "w") as f:
            json.dump(data, f, default=str)
    except Exception as e:
        logger.error(f"Failed to write job {job_id}: {e}")


def start_scan_job(symbols: list) -> str:
    """
    Start a background scan job.
    Returns job_id for tracking.
    """
    job_id = str(uuid.uuid4())[:8]

    # Initialize job file
    job_data = {
        "job_id": job_id,
        "status": "PENDING",
        "symbols_count": len(symbols),
        "started_at": datetime.now(ET).isoformat(),
        "progress": 0,
        "message": "Initializing scan...",
        "result": None,
        "error": None,
    }
    _write_job_file(job_id, job_data)

    # Start background thread
    def _run_scan():
        try:
            job_data["status"] = "RUNNING"
            job_data["message"] = "Downloading data..."
            _write_job_file(job_id, job_data)

            # Run the full scan
            result = build_scan_report(symbols, progress_cb=None)

            job_data["status"] = "COMPLETE"
            job_data["progress"] = 100
            job_data["message"] = "Scan complete"
            job_data["result"] = {
                "momentum_df_shape": result.get("momentum_df", pd.DataFrame()).shape,
                "shortlist_count": len(result.get("shortlist_df", pd.DataFrame())),
                "regime": result.get("regime", {}),
            }
            job_data["completed_at"] = datetime.now(ET).isoformat()

        except Exception as e:
            job_data["status"] = "FAILED"
            job_data["error"] = str(e)
            logger.error(f"Job {job_id} failed: {e}")

        finally:
            _write_job_file(job_id, job_data)

    thread = threading.Thread(target=_run_scan, daemon=True)
    thread.start()

    return job_id


def get_job_status(job_id: str) -> dict:
    """Get current status of a job."""
    return _read_job_file(job_id)


def get_job_result(job_id: str) -> dict | None:
    """Get the full result if job is complete."""
    job_data = _read_job_file(job_id)

    if job_data.get("status") == "COMPLETE":
        # Re-run the scan to get fresh result
        # (In production, would cache the result object)
        return job_data.get("result")

    return None


def list_active_jobs() -> list:
    """List all active/recent jobs."""
    try:
        files = os.listdir(JOB_DIR)
        jobs = []
        for f in files:
            if f.endswith(".json"):
                job_id = f[:-5]
                job_data = _read_job_file(job_id)
                if job_data and job_data.get("status") in ["PENDING", "RUNNING", "COMPLETE"]:
                    jobs.append({
                        "job_id": job_id,
                        "status": job_data.get("status"),
                        "started_at": job_data.get("started_at"),
                        "message": job_data.get("message"),
                    })
        return sorted(jobs, key=lambda x: x.get("started_at", ""), reverse=True)
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        return []


def cleanup_old_jobs(days: int = 1) -> None:
    """Clean up job files older than N days."""
    try:
        cutoff = datetime.now(ET) - timedelta(days=days)
        for f in os.listdir(JOB_DIR):
            if f.endswith(".json"):
                path = os.path.join(JOB_DIR, f)
                file_time = datetime.fromtimestamp(os.path.getmtime(path), ET)
                if file_time < cutoff:
                    os.remove(path)
    except Exception as e:
        logger.error(f"Failed to cleanup jobs: {e}")


# Import pandas for type hints
import pandas as pd
