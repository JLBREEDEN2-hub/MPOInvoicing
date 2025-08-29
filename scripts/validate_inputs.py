#!/usr/bin/env python3
"""
Input validation script for the GitLab CI/CD pipeline.
Validates uploaded MPO and Invoice files before processing.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.etl.mpo_processor import MPOProcessor
from src.etl.invoice_processor import InvoiceProcessor
from src.utils.file_handlers import FileHandler


def setup_logging(level: str = "INFO"):
    """Setup basic logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_file_structure(base_path: Path, logger: logging.Logger) -> dict:
    """
    Validate that the file structure follows expected patterns.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'files_found': {
            'mpo': [],
            'invoice': []
        }
    }
    
    raw_path = base_path / "data" / "raw"
    
    if not raw_path.exists():
        validation_results['errors'].append(f"Raw data directory not found: {raw_path}")
        validation_results['is_valid'] = False
        return validation_results
    
    # Check for MPO files
    mpo_path = raw_path / "mpo"
    if mpo_path.exists():
        for program_dir in mpo_path.iterdir():
            if not program_dir.is_dir():
                continue
                
            for year_dir in program_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                    
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir():
                        continue
                    
                    # Find Excel files
                    excel_files = list(month_dir.glob("*.xlsx"))
                    for file_path in excel_files:
                        validation_results['files_found']['mpo'].append({
                            'path': str(file_path),
                            'program': program_dir.name,
                            'year': year_dir.name,
                            'month': month_dir.name,
                            'size': file_path.stat().st_size
                        })
    
    # Check for Invoice files
    invoice_path = raw_path / "invoices"
    if invoice_path.exists():
        for program_dir in invoice_path.iterdir():
            if not program_dir.is_dir():
                continue
                
            for year_dir in program_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                    
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir():
                        continue
                    
                    # Find CSV and Excel files
                    data_files = list(month_dir.glob("*.csv")) + list(month_dir.glob("*.xlsx"))
                    for file_path in data_files:
                        validation_results['files_found']['invoice'].append({
                            'path': str(file_path),
                            'program': program_dir.name,
                            'year': year_dir.name,
                            'month': month_dir.name,
                            'size': file_path.stat().st_size
                        })
    
    # Check if we found any files
    total_files = len(validation_results['files_found']['mpo']) + len(validation_results['files_found']['invoice'])
    
    if total_files == 0:
        validation_results['warnings'].append("No data files found for processing")
    else:
        logger.info(f"Found {len(validation_results['files_found']['mpo'])} MPO files")
        logger.info(f"Found {len(validation_results['files_found']['invoice'])} Invoice files")
    
    return validation_results


def validate_mpo_file(file_info: dict, config: dict, logger: logging.Logger) -> dict:
    """
    Validate a single MPO file.
    
    Returns:
        Dictionary with validation results
    """
    file_path = Path(file_info['path'])
    logger.info(f"Validating MPO file: {file_path.name}")
    
    validation_results = {
        'file': file_path.name,
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # Load program configuration
        program_config = {}
        config_path = Path("config/programs") / f"{file_info['program']}.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                program_config = yaml.safe_load(f)
        
        # Initialize processor
        processor = MPOProcessor(program_config)
        
        # Load and validate file
        df = processor.load_mpo_data(file_path)
        
        # Run processor validation
        processor_results = processor.validate(df)
        
        # Merge results
        validation_results['is_valid'] = processor_results['is_valid']
        validation_results['errors'].extend(processor_results['errors'])
        validation_results['warnings'].extend(processor_results['warnings'])
        validation_results['stats'] = processor_results['stats']
        
        logger.info(f"MPO file validation complete: {len(df)} rows, {validation_results['stats'].get('total_funding', 0):,.2f} total funding")
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Error validating MPO file: {str(e)}")
        logger.error(f"Error validating MPO file {file_path.name}: {e}")
    
    return validation_results


def validate_invoice_file(file_info: dict, config: dict, logger: logging.Logger) -> dict:
    """
    Validate a single Invoice file.
    
    Returns:
        Dictionary with validation results
    """
    file_path = Path(file_info['path'])
    logger.info(f"Validating Invoice file: {file_path.name}")
    
    validation_results = {
        'file': file_path.name,
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # Initialize processor
        processor = InvoiceProcessor()
        
        # Load file
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Process and validate
        processed_df = processor.process(df)
        processor_results = processor.validate(processed_df)
        
        # Merge results
        validation_results['is_valid'] = processor_results['is_valid']
        validation_results['errors'].extend(processor_results['errors'])
        validation_results['warnings'].extend(processor_results['warnings'])
        validation_results['stats'] = processor_results['stats']
        
        logger.info(f"Invoice file validation complete: {len(processed_df)} rows, {validation_results['stats'].get('total_billed', 0):,.2f} total billed")
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Error validating Invoice file: {str(e)}")
        logger.error(f"Error validating Invoice file {file_path.name}: {e}")
    
    return validation_results


def generate_validation_report(all_results: dict, output_path: Path):
    """Generate validation report."""
    report_lines = [
        "# Input Validation Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        f"- Total files validated: {len(all_results['mpo_results']) + len(all_results['invoice_results'])}",
        f"- MPO files: {len(all_results['mpo_results'])}",
        f"- Invoice files: {len(all_results['invoice_results'])}",
        f"- Overall valid: {all_results['overall_valid']}",
        ""
    ]
    
    # MPO Results
    if all_results['mpo_results']:
        report_lines.extend([
            "## MPO File Results",
            ""
        ])
        
        for result in all_results['mpo_results']:
            status = "✅ VALID" if result['is_valid'] else "❌ INVALID"
            report_lines.append(f"- **{result['file']}**: {status}")
            
            if result['errors']:
                for error in result['errors']:
                    report_lines.append(f"  - Error: {error}")
            
            if result['warnings']:
                for warning in result['warnings']:
                    report_lines.append(f"  - Warning: {warning}")
            
            if result['stats']:
                stats = result['stats']
                report_lines.extend([
                    f"  - Total Funding: ${stats.get('total_funding', 0):,.2f}",
                    f"  - CLINs: {stats.get('total_clins', 0)}",
                    f"  - SLINs: {stats.get('total_slins', 0)}"
                ])
            report_lines.append("")
    
    # Invoice Results
    if all_results['invoice_results']:
        report_lines.extend([
            "## Invoice File Results",
            ""
        ])
        
        for result in all_results['invoice_results']:
            status = "✅ VALID" if result['is_valid'] else "❌ INVALID"
            report_lines.append(f"- **{result['file']}**: {status}")
            
            if result['errors']:
                for error in result['errors']:
                    report_lines.append(f"  - Error: {error}")
            
            if result['warnings']:
                for warning in result['warnings']:
                    report_lines.append(f"  - Warning: {warning}")
            
            if result['stats']:
                stats = result['stats']
                report_lines.extend([
                    f"  - Total Billed: ${stats.get('total_billed', 0):,.2f}",
                    f"  - Invoices: {stats.get('unique_invoices', 0)}",
                    f"  - Employees: {stats.get('unique_employees', 0)}"
                ])
            report_lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate input files for allocation processing")
    parser.add_argument(
        "--path",
        type=Path,
        default=".",
        help="Base path to search for files"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="config/focusedfox.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="validation_report.md",
        help="Output path for validation report"
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with error code if validation fails"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting input validation")
    
    # Load configuration
    config = {}
    if args.config.exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Change to base path
    os.chdir(args.path)
    
    # Validate file structure
    logger.info("Validating file structure...")
    structure_results = validate_file_structure(Path("."), logger)
    
    if not structure_results['is_valid']:
        logger.error("File structure validation failed")
        for error in structure_results['errors']:
            logger.error(f"  - {error}")
        if args.fail_on_error:
            return 1
    
    # Validate individual files
    all_results = {
        'structure_results': structure_results,
        'mpo_results': [],
        'invoice_results': [],
        'overall_valid': True
    }
    
    # Validate MPO files
    logger.info("Validating MPO files...")
    for file_info in structure_results['files_found']['mpo']:
        result = validate_mpo_file(file_info, config, logger)
        all_results['mpo_results'].append(result)
        if not result['is_valid']:
            all_results['overall_valid'] = False
    
    # Validate Invoice files
    logger.info("Validating Invoice files...")
    for file_info in structure_results['files_found']['invoice']:
        result = validate_invoice_file(file_info, config, logger)
        all_results['invoice_results'].append(result)
        if not result['is_valid']:
            all_results['overall_valid'] = False
    
    # Generate report
    logger.info(f"Generating validation report: {args.output}")
    generate_validation_report(all_results, args.output)
    
    # Write environment variables for GitLab CI
    with open("validation.env", "w") as f:
        f.write(f"VALIDATION_STATUS={'PASS' if all_results['overall_valid'] else 'FAIL'}\n")
        f.write(f"MPO_FILES_COUNT={len(all_results['mpo_results'])}\n")
        f.write(f"INVOICE_FILES_COUNT={len(all_results['invoice_results'])}\n")
    
    # Summary
    logger.info(f"Validation complete: {'PASS' if all_results['overall_valid'] else 'FAIL'}")
    
    if args.fail_on_error and not all_results['overall_valid']:
        logger.error("Validation failed - exiting with error code")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())