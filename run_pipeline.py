"""
Bluestock Mutual Fund Analytics Master Pipeline Orchestrator.

This script coordinates the sequential execution of the data pipeline:
1. Ingestion & Database Schema Set-up (etl_pipeline)
2. Quantitative Risk & Return Ratios (compute_metrics)
3. Machine Learning Segmentations (train_clustering)
4. Advanced Financial Analytics & Cohorts (run_day6_tasks)

Usage:
    python run_pipeline.py
"""

import sys
import time
import logging

# Set up logging for orchestration step-by-step
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Orchestrator] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("==========================================================")
    logger.info("Starting Bluestock Fintech Mutual Fund Analytics Pipeline")
    logger.info("==========================================================")
    start_time = time.time()
    
    try:
        # Step 1: Ingest Data
        logger.info("Step 1: Running ETL Pipeline (CSV ingestion & database schema setup)...")
        from scripts import etl_pipeline
        etl_pipeline.run_etl()
        logger.info("ETL Pipeline completed successfully.\n")
        
        # Step 2: Compute Metrics
        logger.info("Step 2: Running Quantitative Metrics Engine...")
        from scripts import compute_metrics
        compute_metrics.compute_all_metrics()
        logger.info("Metrics calculation completed successfully.\n")
        
        # Step 3: Train Clustering
        logger.info("Step 3: Training K-Means Investor Persona Models...")
        from scripts import train_clustering
        train_clustering.train()
        logger.info("ML persona clustering completed successfully.\n")
        
        # Step 4: Advanced Risk & Cohort Reports
        logger.info("Step 4: Running Day 6 Advanced Financial Analytics...")
        from scripts import run_day6_tasks
        run_day6_tasks.run_day6()
        logger.info("Day 6 Advanced Analytics completed successfully.\n")
        
        elapsed_time = time.time() - start_time
        logger.info("==========================================================")
        logger.info(f"Pipeline executed successfully in {elapsed_time:.2f} seconds!")
        logger.info("All quantitative metrics, ML personas, and reports are up-to-date.")
        logger.info("You can now run 'streamlit run dashboard/app.py' to launch the UI.")
        logger.info("==========================================================")
        
    except Exception as e:
        logger.error("==========================================================")
        logger.error(f"Pipeline execution failed due to an error: {str(e)}")
        logger.error("Please resolve the error and restart the pipeline.")
        logger.error("==========================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
