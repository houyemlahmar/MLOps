"""
Simple scheduler for ML pipeline - Alternative to Airflow
Run: python scheduler.py
"""
import schedule
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)

PROJECT_ROOT = Path(__file__).parent
logger = logging.getLogger(__name__)


def run_pipeline():
    """Execute the ML pipeline"""
    logger.info("=" * 80)
    logger.info(f"Starting pipeline run at {datetime.now()}")
    logger.info("=" * 80)
    
    try:
        # 1. Data validation and preprocessing
        logger.info("Step 1: Data validation and preprocessing")
        subprocess.run(
            ["python", "src/data.py"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✓ Data preprocessing completed")
        
        # 2. Feature engineering
        logger.info("Step 2: Feature engineering")
        subprocess.run(
            ["python", "src/features.py"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✓ Feature engineering completed")
        
        # 3. Model training
        logger.info("Step 3: Model training")
        subprocess.run(
            ["python", "src/train_final.py"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✓ Model training completed")
        
        # 4. Model evaluation
        logger.info("Step 4: Model evaluation")
        result = subprocess.run(
            ["python", "src/eval.py"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✓ Model evaluation completed")
        logger.info(result.stdout)
        
        # 5. Data drift monitoring
        logger.info("Step 5: Data drift monitoring")
        subprocess.run(
            ["python", "src/monitor.py"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✓ Drift monitoring completed")
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed at step: {e.cmd}")
        logger.error(f"Error: {e.stderr}")
        logger.error("=" * 80)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error("=" * 80)


def main():
    """Main scheduler"""
    logger.info("ML Pipeline Scheduler Started")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    # Schedule daily at 2:00 AM
    schedule.every().day.at("02:00").do(run_pipeline)
    
    # Optional: Schedule every hour for testing
    # schedule.every().hour.do(run_pipeline)
    
    # Optional: Run immediately on startup
    # run_pipeline()
    
    logger.info("Scheduler configured. Press Ctrl+C to stop.")
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()
