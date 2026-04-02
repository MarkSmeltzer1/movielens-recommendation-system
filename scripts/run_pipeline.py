import subprocess
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPTS_DIR = "scripts/data_prep"
STEPS = [
    "step1_genre_onehot.py",
    "step2_data_quality_checks.py",
    "step3_feature_selection.py",
    "step4_time_based_split.py"
]

def run_step(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
        
    logger.info(f"Running {script_name}...")
    try:
        # Run using the same python interpreter
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        logger.info(f"{script_name} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        return False

def main():
    logger.info("Starting Data Preparation Pipeline...")
    
    for step in STEPS:
        if not run_step(step):
            logger.error("Pipeline failed. Stopping.")
            sys.exit(1)
            
    logger.info("Pipeline finished successfully! Data is ready in data/processed/")
    
    # Upload to S3 (Backup & Versioning)
    BUCKET_NAME = "movielens-data-7741bd4d"
    logger.info(f"Uploading processed data to s3://{BUCKET_NAME}/processed/...")
    try:
        subprocess.run(["aws", "s3", "cp", "data/processed/", f"s3://{BUCKET_NAME}/processed/", "--recursive"], check=True)
        logger.info("Upload complete! ☁️")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upload to S3: {e}")

if __name__ == "__main__":
    main()
