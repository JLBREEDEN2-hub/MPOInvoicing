"""
MPO Data Processor for handling MPO Excel uploads from GitLab.
Replicates Power Query MPOSheet transformation logic.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MPOProcessor:
    """Process MPO Excel files with Power Query transformation logic."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize MPO Processor with configuration.
        
        Args:
            config: Configuration dictionary with excluded CLINs and fee descriptions
        """
        self.config = config or {}
        self.excluded_clins = self.config.get('excluded_clins', [])
        self.fee_descriptions = self.config.get('fee_descriptions', [
            'FEE', 'AWARD FEE', 'INCENTIVE FEE', 'BASE FEE', 'FIXED FEE'
        ])
        
        # Required columns from MPO sheet
        self.required_columns = [
            'CLIN', 'SLIN', 'Description', 'Accum Total', 'InvoiceStartDate',
            'Prior ITD Bill', 'Current ITD Bill', 'Funding', 'Remaining Funding'
        ]
    
    def load_mpo_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load and validate MPO Excel file.
        
        Args:
            file_path: Path to MPO Excel file
            
        Returns:
            Validated DataFrame
        """
        logger.info(f"Loading MPO data from {file_path.name}")
        
        try:
            df = pd.read_excel(file_path)
            return self._validate_columns(df)
        except Exception as e:
            logger.error(f"Error loading MPO file: {e}")
            raise
    
    def process_mpo_sheet(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Replicate Power Query MPOSheet transformation logic.
        
        Args:
            df: Raw MPO DataFrame
            
        Returns:
            Tuple of (processed DataFrame, grouped dictionary by CLIN)
        """
        logger.info(f"Processing MPO sheet with {len(df)} rows")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Step 1: Transform CLIN to text
        processed_df['CLIN'] = processed_df['CLIN'].astype(str).str.strip()
        
        # Step 2: Duplicate and split Description column
        processed_df['Description_Copy'] = processed_df['Description']
        
        # Check if Description contains dash separator
        has_dash = processed_df['Description'].str.contains('-', na=False).any()
        
        if has_dash:
            # Split on dash if present
            split_result = processed_df['Description_Copy'].str.split('-', n=1, expand=True)
            processed_df['Description_Part1'] = split_result[0].fillna('')
            processed_df['Description_Part2'] = split_result[1].fillna('')
        else:
            # No dash separator, use full description as Part1
            processed_df['Description_Part1'] = processed_df['Description_Copy'].fillna('')
            processed_df['Description_Part2'] = ''
        
        # Step 3: Filter rows based on business rules
        initial_rows = len(processed_df)
        
        # Convert fee descriptions to uppercase for comparison
        fee_descriptions_upper = [desc.upper() for desc in self.fee_descriptions]
        
        # Apply filters
        processed_df = processed_df[
            # Exclude fee-related descriptions
            (~processed_df['Description_Part2'].str.upper().str.strip().isin(fee_descriptions_upper)) &
            # Exclude specific CLINs
            (~processed_df['CLIN'].isin([str(x) for x in self.excluded_clins])) &
            # CLIN must not be null
            (processed_df['CLIN'].notna()) &
            (processed_df['CLIN'] != 'nan') &
            # Accum Total must not be zero
            (processed_df['Accum Total'] != 0)
        ]
        
        filtered_rows = initial_rows - len(processed_df)
        logger.info(f"Filtered out {filtered_rows} rows based on business rules")
        
        # Step 4: Sort by CLIN and SLIN
        if 'SLIN' in processed_df.columns:
            processed_df['SLIN'] = processed_df['SLIN'].astype(str).str.strip()
            processed_df = processed_df.sort_values(['CLIN', 'SLIN'])
        else:
            processed_df = processed_df.sort_values('CLIN')
            processed_df['SLIN'] = ''
        
        # Step 5: Add MPO index
        processed_df.reset_index(drop=True, inplace=True)
        processed_df['MPO_Index'] = range(len(processed_df))
        
        # Step 6: Rename and select columns
        column_mapping = {
            'Accum Total': 'AvailableFunding',
            'Prior ITD Bill': 'PriorITDBill',
            'Current ITD Bill': 'CurrentITDBill',
            'Funding': 'TotalFunding',
            'Remaining Funding': 'RemainingFunding'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in processed_df.columns:
                processed_df[new_col] = processed_df[old_col]
        
        # Select columns to keep
        columns_to_keep = [
            'CLIN', 'SLIN', 'AvailableFunding', 'InvoiceStartDate', 
            'MPO_Index', 'Description', 'PriorITDBill', 'CurrentITDBill',
            'TotalFunding', 'RemainingFunding'
        ]
        
        # Only keep columns that exist
        columns_to_keep = [col for col in columns_to_keep if col in processed_df.columns]
        processed_df = processed_df[columns_to_keep]
        
        # Step 7: Add month-year column
        if 'InvoiceStartDate' in processed_df.columns:
            processed_df['InvoiceStartDate'] = pd.to_datetime(processed_df['InvoiceStartDate'], errors='coerce')
            processed_df['MPO_Month_Year'] = processed_df['InvoiceStartDate'].dt.strftime('%Y-%m')
        else:
            processed_df['MPO_Month_Year'] = datetime.now().strftime('%Y-%m')
        
        # Step 8: Add processing metadata
        processed_df['Has_MPO'] = 'Yes'
        processed_df['MPO_ProcessedDate'] = pd.Timestamp.now()
        
        # Step 9: Group by CLIN for nested structure
        grouped = self._create_nested_structure(processed_df)
        
        logger.info(f"MPO processing complete: {len(processed_df)} rows retained")
        
        return processed_df, grouped
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing method for compatibility with ETL pipeline.
        
        Args:
            df: Raw MPO DataFrame
            
        Returns:
            Processed MPO DataFrame
        """
        processed_df, _ = self.process_mpo_sheet(df)
        return processed_df
    
    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that required columns exist in the DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
        """
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to find alternative column names
            alternative_found = []
            for col in missing_columns:
                alternatives = self._find_alternative_column(df.columns, col)
                if alternatives:
                    # Use the first alternative found
                    df.rename(columns={alternatives[0]: col}, inplace=True)
                    alternative_found.append(col)
            
            # Update missing columns list
            missing_columns = [col for col in missing_columns if col not in alternative_found]
            
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                # Create default columns for missing ones
                for col in missing_columns:
                    if col in ['Accum Total', 'Prior ITD Bill', 'Current ITD Bill', 
                              'Funding', 'Remaining Funding']:
                        df[col] = 0
                    elif col == 'InvoiceStartDate':
                        df[col] = pd.NaT
                    else:
                        df[col] = ''
        
        return df
    
    def _find_alternative_column(self, columns: List[str], target: str) -> List[str]:
        """
        Find alternative column names that might match the target.
        
        Args:
            columns: List of available column names
            target: Target column name
            
        Returns:
            List of potential alternative column names
        """
        alternatives = []
        target_lower = target.lower().replace(' ', '').replace('_', '')
        
        for col in columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if col_lower == target_lower or target_lower in col_lower:
                alternatives.append(col)
        
        return alternatives
    
    def _create_nested_structure(self, df: pd.DataFrame) -> Dict:
        """
        Create nested dictionary structure grouped by CLIN.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with CLIN as key and list of records as value
        """
        grouped = {}
        
        for clin in df['CLIN'].unique():
            clin_data = df[df['CLIN'] == clin]
            grouped[clin] = clin_data.to_dict('records')
        
        return grouped
    
    def calculate_funding_summary(self, df: pd.DataFrame) -> Dict:
        """
        Calculate funding summary statistics.
        
        Args:
            df: Processed MPO DataFrame
            
        Returns:
            Dictionary with funding statistics
        """
        summary = {
            'total_funding': df['AvailableFunding'].sum() if 'AvailableFunding' in df.columns else 0,
            'total_clins': df['CLIN'].nunique() if 'CLIN' in df.columns else 0,
            'total_slins': df['SLIN'].nunique() if 'SLIN' in df.columns else 0,
            'funding_by_clin': {},
            'zero_funding_clins': []
        }
        
        if 'CLIN' in df.columns and 'AvailableFunding' in df.columns:
            # Funding by CLIN
            funding_by_clin = df.groupby('CLIN')['AvailableFunding'].sum()
            summary['funding_by_clin'] = funding_by_clin.to_dict()
            
            # CLINs with zero funding
            zero_funding = funding_by_clin[funding_by_clin == 0]
            summary['zero_funding_clins'] = zero_funding.index.tolist()
        
        return summary
    
    def validate(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate processed MPO data.
        
        Args:
            df: Processed MPO DataFrame
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for required columns
        missing_critical = []
        for col in ['CLIN', 'AvailableFunding']:
            if col not in df.columns:
                missing_critical.append(col)
        
        if missing_critical:
            validation_results['errors'].append(
                f"Missing critical columns: {missing_critical}"
            )
            validation_results['is_valid'] = False
        
        # Check for duplicate CLIN/SLIN combinations
        if 'CLIN' in df.columns and 'SLIN' in df.columns:
            duplicates = df[df.duplicated(subset=['CLIN', 'SLIN'], keep=False)]
            if not duplicates.empty:
                validation_results['warnings'].append(
                    f"Found {len(duplicates)} duplicate CLIN/SLIN combinations"
                )
        
        # Check for negative funding
        if 'AvailableFunding' in df.columns:
            negative_funding = df[df['AvailableFunding'] < 0]
            if not negative_funding.empty:
                validation_results['warnings'].append(
                    f"Found {len(negative_funding)} rows with negative funding"
                )
        
        # Generate statistics
        validation_results['stats'] = self.calculate_funding_summary(df)
        
        return validation_results