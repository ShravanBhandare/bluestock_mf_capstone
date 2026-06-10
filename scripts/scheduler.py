"""
Daily Scheduled ETL and Metrics Run Scheduler.

This script runs a background loop to trigger the ETL and metrics computation
daily at a specified hour (default: 8 PM), allowing automated updates of the database.
"""

import time
import datetime
import os
import subprocess
import sys
import pathlib
import logging

# Set up logging configuration
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [Scheduler] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def run_daily_job():
    """
    Triggers the ETL pipeline and the metrics calculation scripts as subprocesses.
    """
    logger.info("Starting scheduled daily run...")
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    etl_script = base_dir / "scripts" / "etl_pipeline.py"
    metrics_script = base_dir / "scripts" / "compute_metrics.py"
    
    # 1. Run ETL
    logger.info(f"Triggering ETL pipeline: {etl_script}")
    etl_process = subprocess.run([sys.executable, str(etl_script)], capture_output=True, text=True)
    if etl_process.returncode == 0:
        logger.info("ETL executed successfully.")
    else:
        logger.error(f"ETL Execution FAILED:\n{etl_process.stderr}")
        return
        
    # 2. Run Metrics Calculation
    logger.info(f"Triggering Metrics calculation: {metrics_script}")
    metrics_process = subprocess.run([sys.executable, str(metrics_script)], capture_output=True, text=True)
    if metrics_process.returncode == 0:
        logger.info("Financial metrics updated successfully.")
    else:
        logger.error(f"Metrics updates FAILED:\n{metrics_process.stderr}")
        return
        
    logger.info("Scheduled job completed successfully.")

def start_scheduler():
    """
    Runs the scheduling loop, waking up at 8 PM daily to run the analytics pipeline.
    """
    target_hour = 20  # 8 PM
    target_minute = 0
    
    logger.info(f"ETL Scheduler active. Target run time: Daily at {target_hour:02d}:{target_minute:02d} (8 PM)")
    
    # Initial load check
    logger.info("Running initial database refresh...")
    run_daily_job()
    
    while True:
        now = datetime.datetime.now()
        target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        
        if now >= target_time:
            # If target time has already passed today, set to tomorrow at 8 PM
            target_time += datetime.timedelta(days=1)
            
        wait_seconds = (target_time - now).total_seconds()
        logger.info(f"Next run scheduled for: {target_time}")
        logger.info(f"Sleeping for {wait_seconds / 3600:.2f} hours...")
        
        # Sleep in chunks to allow keyboard interrupts or signals
        chunk = 60
        while wait_seconds > 0:
            time.sleep(min(chunk, wait_seconds))
            wait_seconds -= chunk
            
        run_daily_job()

if __name__ == "__main__":
    try:
        start_scheduler()
    except KeyboardInterrupt:
        logger.info("Scheduler terminated by user.")
        sys.exit(0)
