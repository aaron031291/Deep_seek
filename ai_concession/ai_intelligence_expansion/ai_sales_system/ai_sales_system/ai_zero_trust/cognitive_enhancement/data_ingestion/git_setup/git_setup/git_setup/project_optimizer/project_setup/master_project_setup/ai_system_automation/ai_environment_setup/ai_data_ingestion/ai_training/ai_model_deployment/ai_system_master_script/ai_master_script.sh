#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import time
import json
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/master_automation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("master_automation")

# Define script paths
SCRIPTS = {
    "setup": "./scripts/setup_env.sh",
    "data_ingestion": "./scripts/data_ingestion.py",
    "ai_training": "./scripts/ai_training.py",
    "model_deployment": "./scripts/model_deployment.py",
    "health_check": "./scripts/health_check.sh",
    "backup_recovery": "./scripts/backup_recovery.sh",
    "logging_monitor": "./scripts/logging_monitor.py",
    "security_audit": "./scripts/security_audit.py",
    "cleanup": "./scripts/cleanup_maintenance.sh"
}

def ensure_script_permissions():
    """Ensure all scripts have execution permissions"""
    logger.info("Setting execution permissions for all scripts...")
    for script_name, script_path in SCRIPTS.items():
        try:
            if os.path.exists(script_path):
                subprocess.run(["chmod", "+x", script_path], check=True)
                logger.info(f"Set execution permission for {script_path}")
            else:
                logger.warning(f"Script {script_path} does not exist")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set permissions for {script_path}: {e}")
            return False
    return True

def create_directory_structure():
    """Create necessary directory structure if it doesn't exist"""
    directories = [
        "./scripts",
        "./data/raw",
        "./data/processed",
        "./data/backup",
        "./models/trained",
        "./models/deployed",
        "./models/archived",
        "./logs",
        "./config",
        "./temp"
    ]
    
    logger.info("Creating directory structure...")
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False
    return True

def run_script(script_name, timeout=None, args=None):
    """Run a script and return success status"""
    script_path = SCRIPTS.get(script_name)
    if not script_path:
        logger.error(f"Unknown script: {script_name}")
        return False
        
    if not os.path.exists(script_path):
        logger.error(f"Script does not exist: {script_path}")
        return False
    
    start_time = time.time()
    logger.info(f"Starting {script_name} script...")
    
    command = [script_path]
    if args:
        command.extend(args)
    
    try:
        process = subprocess.run(
            command,
            timeout=timeout,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Log output
        if process.stdout:
            for line in process.stdout.splitlines():
                logger.info(f"{script_name} output: {line}")
        
        execution_time = time.time() - start_time
        logger.info(f"Successfully completed {script_name} in {execution_time:.2f} seconds")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"{script_name} timed out after {timeout} seconds")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"{script_name} failed with exit code {e.returncode}")
        if e.stderr:
            for line in e.stderr.splitlines():
                logger.error(f"{script_name} error: {line}")
        return False
    except Exception as e:
        logger.error(f"Error running {script_name}: {e}")
        return False

def save_pipeline_status(status, start_time, end_time, failed_steps=None):
    """Save the pipeline execution status to a JSON file"""
    status_data = {
        "pipeline_run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status": status,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "failed_steps": failed_steps or []
    }
    
    try:
        os.makedirs("logs/pipeline_status", exist_ok=True)
        status_file = f"logs/pipeline_status/run_{status_data['pipeline_run_id']}.json"
        
        with open(status_file, "w") as f:
            json.dump(status_data, f, indent=4)
            
        # Also update the latest status file
        with open("logs/pipeline_status/latest_run.json", "w") as f:
            json.dump(status_data, f, indent=4)
            
        logger.info(f"Pipeline status saved to {status_file}")
    except Exception as e:
        logger.error(f"Failed to save pipeline status: {e}")

def run_pipeline(args):
    """Run the complete automation pipeline"""
    start_time = datetime.now()
    failed_steps = []
    
    # Create directory structure
    if not create_directory_structure():
        logger.error("Failed to create directory structure. Exiting.")
        save_pipeline_status("FAILED", start_time, datetime.now(), ["directory_setup"])
        return False
    
    # Ensure script permissions
    if not ensure_script_permissions():
        logger.error("Failed to set script permissions. Exiting.")
        save_pipeline_status("FAILED", start_time, datetime.now(), ["permissions_setup"])
        return False
    
    # Define the pipeline steps in order
    pipeline_steps = []
    
    # Always run setup first
    pipeline_steps.append(("setup", None))
    
    # Add health check early
    pipeline_steps.append(("health_check", None))
    
    # Add security audit
    pipeline_steps.append(("security_audit", None))
    
    # Data and model pipeline
    if not args.skip_data:
        pipeline_steps.append(("data_ingestion", None))
    
    if not args.skip_training:
        pipeline_steps.append(("ai_training", None))
    
    if not args.skip_deployment and not args.skip_training:
        deploy_args = ["--environment", args.environment] if args.environment else None
        pipeline_steps.append(("model_deployment", deploy_args))
    
    # Add monitoring
    pipeline_steps.append(("logging_monitor", None))
    
    # Add backup if requested
    if args.backup:
        pipeline_steps.append(("backup_recovery", ["--backup-only"]))
    
    # Always run cleanup last
    pipeline_steps.append(("cleanup", None))
    
    # Execute pipeline steps
    for step_name, step_args in pipeline_steps:
        if not run_script(step_name, timeout=args.timeout, args=step_args):
            logger.error(f"Pipeline step {step_name} failed")
            failed_steps.append(step_name)
            
            if args.fail_fast:
                logger.error("Stopping pipeline due to fail-fast option")
                break
    
    # Determine overall status
    end_time = datetime.now()
    status = "SUCCESS" if not failed_steps else "PARTIAL_SUCCESS" if len(failed_steps) < len(pipeline_steps) else "FAILED"
    
    # Save pipeline status
    save_pipeline_status(status, start_time, end_time, failed_steps)
    
    # Log summary
    logger.info(f"Pipeline completed with status: {status}")
    logger.info(f"Total execution time: {(end_time - start_time).total_seconds():.2f} seconds")
    
    if failed_steps:
        logger.info(f"Failed steps: {', '.join(failed_steps)}")
    
    return len(failed_steps) == 0

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AI System Automation Pipeline")
    
    parser.add_argument("--skip-data", action="store_true", help="Skip data ingestion step")
    parser.add_argument("--skip-training", action="store_true", help="Skip AI model training step")
    parser.add_argument("--skip-deployment", action="store_true", help="Skip model deployment step")
    parser.add_argument("--environment", choices=["staging", "production"], help="Deployment environment")
    parser.add_argument("--backup", action="store_true", help="Perform backup")
    parser.add_argument("--fail-fast", action="store_true", help="Stop pipeline on first failure")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout for each step in seconds")
    
    return parser.parse_args()

def main():
    """Main function to run the automation pipeline"""
    # Parse arguments
    args = parse_arguments()
    
    logger.info("Starting AI System Automation Pipeline")
    logger.info(f"Command line arguments: {args}")
    
    # Run the pipeline
    success = run_pipeline(args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
