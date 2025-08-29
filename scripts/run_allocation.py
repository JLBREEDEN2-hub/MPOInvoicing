#!/usr/bin/env python3
"""
Main script for running the invoice allocation process.
This is the primary entry point for the GitLab CI/CD pipeline.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.etl.pipeline import ETLPipeline
from src.allocation.allocation_engine import AllocationEngine
from src.utils.file_handlers import FileHandler


def setup_logging(config: dict):
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    
    # Create logs directory
    logs_path = Path(config.get('logs_path', 'output/logs'))
    logs_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    if log_config.get('console_handler', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_config.get('file_handler', True):
        log_file = logs_path / f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        sys.exit(1)


def load_program_config(program: str, config_dir: Path) -> dict:
    """Load program-specific configuration."""
    # Configuration is now consolidated in the main config file
    config_path = config_dir / f"{program}.yaml"
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load program config for {program}: {e}")
    
    return {}


def find_data_to_process(base_path: Path, program: str = None) -> list:
    """
    Find data that needs to be processed.
    
    Returns:
        List of (program, year, month) tuples to process
    """
    data_to_process = []
    raw_path = base_path / "data" / "raw"
    
    if not raw_path.exists():
        return data_to_process
    
    # If program is specified, only process that program
    if program:
        programs_to_check = [program]
    else:
        # Find all programs
        mpo_path = raw_path / "mpo"
        invoice_path = raw_path / "invoices"
        
        programs_to_check = set()
        if mpo_path.exists():
            programs_to_check.update([p.name for p in mpo_path.iterdir() if p.is_dir()])
        if invoice_path.exists():
            programs_to_check.update([p.name for p in invoice_path.iterdir() if p.is_dir()])
    
    # Find year/month combinations with data
    for prog in programs_to_check:
        for data_type in ['mpo', 'invoices']:
            type_path = raw_path / data_type / prog
            if not type_path.exists():
                continue
            
            for year_dir in type_path.iterdir():
                if not year_dir.is_dir() or not year_dir.name.isdigit():
                    continue
                
                year = int(year_dir.name)
                
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir() or not month_dir.name.isdigit():
                        continue
                    
                    month = int(month_dir.name)
                    
                    # Check if there are files to process
                    files = list(month_dir.glob("*"))
                    if files:
                        data_tuple = (prog, year, month)
                        if data_tuple not in data_to_process:
                            data_to_process.append(data_tuple)
    
    return sorted(data_to_process)


def process_allocation(program: str, year: int, month: int, config: dict, 
                      program_config: dict, logger: logging.Logger) -> bool:
    """
    Process allocation for a specific program/year/month.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing allocation for {program} {year}/{month:02d}")
    
    try:
        # Initialize components
        etl_pipeline = ETLPipeline(config)
        allocation_engine = AllocationEngine({**config, **program_config})
        file_handler = FileHandler(config.get('base_path', '.'))
        
        # Step 1: Run ETL pipeline
        logger.info("Running ETL pipeline...")
        processed_data, metadata = etl_pipeline.run(program, year, month)
        
        if processed_data.empty:
            logger.warning(f"No data to process for {program} {year}/{month:02d}")
            return True
        
        logger.info(f"ETL pipeline completed: {len(processed_data)} rows processed")
        
        # Step 2: Run allocation engine
        logger.info("Running allocation engine...")
        allocation_results = allocation_engine.allocate(processed_data)
        
        if allocation_results.empty:
            logger.warning(f"No allocations generated for {program} {year}/{month:02d}")
            return True
        
        logger.info(f"Allocation completed: {len(allocation_results)} allocation records")
        
        # Step 3: Save results
        logger.info("Saving results...")
        output_path = file_handler.save_results(allocation_results, program, year, month)
        logger.info(f"Results saved to: {output_path}")
        
        # Step 4: Generate summary
        summary = allocation_engine.get_allocation_summary()
        logger.info(f"Allocation summary: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {program} {year}/{month:02d}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Invoice Allocation Process")
    parser.add_argument(
        "--config", 
        type=Path, 
        default="config/focusedfox.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--program", 
        type=str,
        help="Specific program to process (default: all programs)"
    )
    parser.add_argument(
        "--year", 
        type=int,
        help="Specific year to process"
    )
    parser.add_argument(
        "--month", 
        type=int,
        help="Specific month to process"
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess existing data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting Invoice Allocation Process")
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Create output directory
    output_path = Path(config.get('output_path', 'output'))
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find data to process
    if args.program and args.year and args.month:
        # Process specific program/year/month
        data_to_process = [(args.program, args.year, args.month)]
    else:
        # Find all data to process
        data_to_process = find_data_to_process(project_dir, args.program)
    
    if not data_to_process:
        logger.warning("No data found to process")
        return 0
    
    logger.info(f"Found {len(data_to_process)} datasets to process: {data_to_process}")
    
    if args.dry_run:
        logger.info("Dry run mode - would process the following:")
        for program, year, month in data_to_process:
            logger.info(f"  - {program} {year}/{month:02d}")
        return 0
    
    # Process each dataset
    success_count = 0
    
    for program, year, month in data_to_process:
        # Load program-specific configuration (now same as main config for focusedfox)
        if program == 'focusedfox':
            program_config = config
        else:
            program_config = load_program_config(program, args.config.parent)
        
        # Process allocation
        if process_allocation(program, year, month, config, program_config, logger):
            success_count += 1
        else:
            logger.error(f"Failed to process {program} {year}/{month:02d}")
    
    # Summary
    logger.info(f"Processing complete: {success_count}/{len(data_to_process)} successful")
    
    if success_count == len(data_to_process):
        logger.info("All datasets processed successfully")
        return 0
    else:
        logger.error(f"{len(data_to_process) - success_count} datasets failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())