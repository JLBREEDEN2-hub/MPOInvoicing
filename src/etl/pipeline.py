"""
ETL Pipeline for processing MPO and Invoice data from GitLab repository.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml

from .mpo_processor import MPOProcessor
from .invoice_processor import InvoiceProcessor
from ..utils.file_handlers import FileHandler

logger = logging.getLogger(__name__)


class ETLPipeline:
    """Main ETL pipeline for processing uploaded data from GitLab."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize ETL Pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.file_handler = FileHandler(self.config.get('base_path', '.'))
        self.mpo_processor = MPOProcessor()
        self.invoice_processor = InvoiceProcessor()
        self.processed_data = {}
        
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def run(self, program: str, year: int, month: int) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete ETL pipeline for a specific program/period.
        
        Args:
            program: Program name (e.g., 'focusedfox')
            year: Year to process
            month: Month to process
            
        Returns:
            Tuple of (processed_dataframe, metadata_dict)
        """
        logger.info(f"Starting ETL pipeline for {program} {year}/{month:02d}")
        
        try:
            # Step 1: Extract - Find and load files
            files = self.file_handler.find_files(program, year, month)
            
            if not files['mpo'] and not files['invoice']:
                raise ValueError(f"No data files found for {program} {year}/{month:02d}")
            
            # Step 2: Transform - Process MPO data
            mpo_data = self._process_mpo_files(files['mpo'])
            
            # Step 3: Transform - Process Invoice data
            invoice_data = self._process_invoice_files(files['invoice'])
            
            # Step 4: Load - Combine and prepare final dataset
            combined_data = self._combine_data(mpo_data, invoice_data)
            
            # Step 5: Generate metadata
            metadata = self._generate_metadata(
                program, year, month, 
                len(files['mpo']), len(files['invoice']), 
                len(combined_data)
            )
            
            logger.info(f"ETL pipeline completed successfully for {program} {year}/{month:02d}")
            return combined_data, metadata
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise
    
    def _process_mpo_files(self, mpo_files: List[Path]) -> pd.DataFrame:
        """Process all MPO Excel files."""
        all_mpo_data = []
        
        for file_path in mpo_files:
            logger.info(f"Processing MPO file: {file_path.name}")
            
            # Read Excel file
            df = self.file_handler.read_excel(file_path)
            
            # Process with MPO processor
            processed_df = self.mpo_processor.process(df)
            processed_df['source_file'] = file_path.name
            processed_df['file_type'] = 'MPO'
            
            all_mpo_data.append(processed_df)
        
        if all_mpo_data:
            return pd.concat(all_mpo_data, ignore_index=True)
        return pd.DataFrame()
    
    def _process_invoice_files(self, invoice_files: List[Path]) -> pd.DataFrame:
        """Process all Invoice files (CSV or Excel)."""
        all_invoice_data = []
        
        for file_path in invoice_files:
            logger.info(f"Processing Invoice file: {file_path.name}")
            
            # Read file based on extension
            if file_path.suffix.lower() == '.csv':
                df = self.file_handler.read_csv(file_path)
            else:
                df = self.file_handler.read_excel(file_path)
            
            # Process with Invoice processor
            processed_df = self.invoice_processor.process(df)
            processed_df['source_file'] = file_path.name
            processed_df['file_type'] = 'Invoice'
            
            all_invoice_data.append(processed_df)
        
        if all_invoice_data:
            return pd.concat(all_invoice_data, ignore_index=True)
        return pd.DataFrame()
    
    def _combine_data(self, mpo_data: pd.DataFrame, 
                     invoice_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine MPO and Invoice data for allocation processing.
        
        Args:
            mpo_data: Processed MPO data
            invoice_data: Processed Invoice data
            
        Returns:
            Combined DataFrame ready for allocation
        """
        logger.info("Combining MPO and Invoice data...")
        
        # If one dataset is empty, return the other
        if mpo_data.empty:
            return invoice_data
        if invoice_data.empty:
            return mpo_data
        
        # Identify common columns for merging
        common_columns = list(set(mpo_data.columns) & set(invoice_data.columns))
        
        # Key columns for merging (prioritized)
        key_columns = ['CLIN', 'SLIN', 'InvoiceID', 'Project_Code']
        merge_keys = [col for col in key_columns if col in common_columns]
        
        if merge_keys:
            # Merge on common keys
            combined = pd.merge(
                invoice_data,
                mpo_data,
                on=merge_keys,
                how='outer',
                suffixes=('', '_mpo'),
                indicator=True
            )
            combined['merge_status'] = combined['_merge'].map({
                'left_only': 'Invoice_Only',
                'right_only': 'MPO_Only',
                'both': 'Matched'
            })
            combined.drop('_merge', axis=1, inplace=True)
        else:
            # No common keys, concatenate
            combined = pd.concat([invoice_data, mpo_data], 
                                ignore_index=True, sort=False)
            combined['merge_status'] = 'Concatenated'
        
        logger.info(f"Combined dataset contains {len(combined)} rows")
        return combined
    
    def _generate_metadata(self, program: str, year: int, month: int,
                          mpo_count: int, invoice_count: int,
                          total_rows: int) -> Dict:
        """Generate metadata for the ETL run."""
        return {
            'program': program,
            'year': year,
            'month': month,
            'processing_date': datetime.now().isoformat(),
            'mpo_files_processed': mpo_count,
            'invoice_files_processed': invoice_count,
            'total_rows_processed': total_rows,
            'pipeline_version': '1.0.0'
        }
    
    def validate_output(self, df: pd.DataFrame) -> bool:
        """
        Validate the processed data before saving.
        
        Args:
            df: Processed DataFrame to validate
            
        Returns:
            True if validation passes
        """
        required_columns = [
            'CLIN', 'SLIN', 'InvoiceID', 'AvailableFunding',
            'Billed_Amt_Burdens', 'InvDescription'
        ]
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            return False
        
        # Check for empty dataset
        if df.empty:
            logger.warning("Processed dataset is empty")
            return False
        
        # Check for critical nulls
        critical_nulls = df[['CLIN', 'InvoiceID']].isnull().sum()
        if critical_nulls.any():
            logger.warning(f"Critical columns contain nulls: {critical_nulls.to_dict()}")
        
        return True