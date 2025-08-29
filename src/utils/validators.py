"""
Data validation utilities for the Invoice Allocation System.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data files and configurations."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize validator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def validate_mpo_data(self, df: pd.DataFrame, program_config: Dict) -> Dict[str, Any]:
        """
        Validate MPO data against program configuration.
        
        Args:
            df: MPO DataFrame
            program_config: Program-specific configuration
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_columns = program_config.get('validation', {}).get('required_mpo_columns', [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['errors'].append(f"Missing required MPO columns: {missing_columns}")
            validation_results['is_valid'] = False
        
        # Check for excluded CLINs
        excluded_clins = program_config.get('excluded_clins', [])
        if 'CLIN' in df.columns and excluded_clins:
            excluded_found = df[df['CLIN'].isin([str(c) for c in excluded_clins])]
            if not excluded_found.empty:
                validation_results['warnings'].append(
                    f"Found {len(excluded_found)} rows with excluded CLINs: {excluded_clins}"
                )
        
        # Check for zero funding if configured to exclude
        processing_config = program_config.get('processing', {})
        if processing_config.get('exclude_zero_funding', True) and 'Accum Total' in df.columns:
            zero_funding = df[df['Accum Total'] == 0]
            if not zero_funding.empty:
                validation_results['warnings'].append(
                    f"Found {len(zero_funding)} rows with zero funding (will be excluded)"
                )
        
        # Generate statistics
        validation_results['stats'] = {
            'total_rows': len(df),
            'unique_clins': df['CLIN'].nunique() if 'CLIN' in df.columns else 0,
            'unique_slins': df['SLIN'].nunique() if 'SLIN' in df.columns else 0,
            'total_funding': df['Accum Total'].sum() if 'Accum Total' in df.columns else 0,
            'zero_funding_rows': len(df[df['Accum Total'] == 0]) if 'Accum Total' in df.columns else 0
        }
        
        return validation_results
    
    def validate_invoice_data(self, df: pd.DataFrame, program_config: Dict) -> Dict[str, Any]:
        """
        Validate Invoice data against program configuration.
        
        Args:
            df: Invoice DataFrame
            program_config: Program-specific configuration
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_columns = program_config.get('validation', {}).get('required_invoice_columns', [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['warnings'].append(f"Missing invoice columns: {missing_columns}")
        
        # Check for fee descriptions
        fee_descriptions = program_config.get('fee_descriptions', [])
        labor_col = program_config.get('column_mappings', {}).get('invoice', {}).get('labor_type', 'BILL_FM_LN_LBL')
        
        if labor_col in df.columns and fee_descriptions:
            fee_rows = df[df[labor_col].str.upper().isin([f.upper() for f in fee_descriptions])]
            if not fee_rows.empty:
                validation_results['warnings'].append(
                    f"Found {len(fee_rows)} fee rows (will be excluded): {fee_descriptions}"
                )
        
        # Check for null descriptions
        if labor_col in df.columns:
            null_descriptions = df[df[labor_col].isna() | (df[labor_col] == '')]
            if not null_descriptions.empty:
                validation_results['warnings'].append(
                    f"Found {len(null_descriptions)} rows with null labor descriptions (will be excluded)"
                )
        
        # Generate statistics
        amount_col = program_config.get('column_mappings', {}).get('invoice', {}).get('amount', 'Billed Amt + Burdens')
        employee_col = program_config.get('column_mappings', {}).get('invoice', {}).get('employee_id', 'Empl/Vendor ID')
        
        validation_results['stats'] = {
            'total_rows': len(df),
            'unique_invoices': df['Invoice ID'].nunique() if 'Invoice ID' in df.columns else 0,
            'unique_employees': df[employee_col].nunique() if employee_col in df.columns else 0,
            'total_amount': df[amount_col].sum() if amount_col in df.columns else 0,
            'credit_invoices': len(df[df[amount_col] < 0]) if amount_col in df.columns else 0,
            'null_descriptions': len(df[df[labor_col].isna() | (df[labor_col] == '')]) if labor_col in df.columns else 0
        }
        
        return validation_results
    
    def validate_program_config(self, program_name: str, config_dir: Path) -> Dict[str, Any]:
        """
        Validate program configuration file.
        
        Args:
            program_name: Name of the program
            config_dir: Directory containing configuration files
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'config': {}
        }
        
        config_file = config_dir / "programs" / f"{program_name}.yaml"
        
        if not config_file.exists():
            validation_results['errors'].append(f"Configuration file not found: {config_file}")
            validation_results['is_valid'] = False
            return validation_results
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            validation_results['config'] = config
            
            # Validate required configuration sections
            required_sections = ['program_name', 'excluded_clins', 'fee_descriptions']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                validation_results['warnings'].append(f"Missing configuration sections: {missing_sections}")
            
            # Validate column mappings
            if 'column_mappings' in config:
                mappings = config['column_mappings']
                if 'mpo' not in mappings or 'invoice' not in mappings:
                    validation_results['warnings'].append("Missing column mappings for mpo or invoice")
            
            # Validate validation rules
            if 'validation' in config:
                val_config = config['validation']
                if 'required_mpo_columns' not in val_config:
                    validation_results['warnings'].append("Missing required_mpo_columns in validation config")
                if 'required_invoice_columns' not in val_config:
                    validation_results['warnings'].append("Missing required_invoice_columns in validation config")
            
            logger.info(f"Configuration validation completed for {program_name}")
            
        except Exception as e:
            validation_results['errors'].append(f"Error loading configuration: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    def validate_file_structure(self, base_path: Path, program: str, year: int, month: int) -> Dict[str, Any]:
        """
        Validate that required file structure exists.
        
        Args:
            base_path: Base directory path
            program: Program name
            year: Year
            month: Month
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'found_files': {
                'mpo': [],
                'invoice': []
            }
        }
        
        # Check MPO directory
        mpo_dir = base_path / "data" / "raw" / "mpo" / program / str(year) / f"{month:02d}"
        if mpo_dir.exists():
            mpo_files = list(mpo_dir.glob("*.xlsx"))
            validation_results['found_files']['mpo'] = [str(f) for f in mpo_files]
            if not mpo_files:
                validation_results['warnings'].append(f"No MPO files found in {mpo_dir}")
        else:
            validation_results['warnings'].append(f"MPO directory does not exist: {mpo_dir}")
        
        # Check Invoice directory
        invoice_dir = base_path / "data" / "raw" / "invoices" / program / str(year) / f"{month:02d}"
        if invoice_dir.exists():
            invoice_files = list(invoice_dir.glob("*.xlsx")) + list(invoice_dir.glob("*.csv"))
            validation_results['found_files']['invoice'] = [str(f) for f in invoice_files]
            if not invoice_files:
                validation_results['warnings'].append(f"No Invoice files found in {invoice_dir}")
        else:
            validation_results['warnings'].append(f"Invoice directory does not exist: {invoice_dir}")
        
        # Check if we have any data to process
        total_files = len(validation_results['found_files']['mpo']) + len(validation_results['found_files']['invoice'])
        if total_files == 0:
            validation_results['errors'].append("No data files found to process")
            validation_results['is_valid'] = False
        
        return validation_results


def validate_inputs(mpo_file: Optional[Path], invoice_file: Optional[Path], 
                   config: Dict) -> Tuple[bool, List[str]]:
    """
    Quick validation function for input files and configuration.
    
    Args:
        mpo_file: Path to MPO file (can be None)
        invoice_file: Path to Invoice file (can be None)
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Validate files exist
    if mpo_file and not mpo_file.exists():
        errors.append(f"MPO file not found: {mpo_file}")
    
    if invoice_file and not invoice_file.exists():
        errors.append(f"Invoice file not found: {invoice_file}")
    
    # Validate configuration
    if not config:
        errors.append("No configuration provided")
    
    required_config_keys = ['program_name', 'excluded_clins', 'fee_descriptions']
    missing_keys = [key for key in required_config_keys if key not in config]
    if missing_keys:
        errors.append(f"Missing configuration keys: {missing_keys}")
    
    return len(errors) == 0, errors