"""
File handling utilities for the Invoice Allocation System.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FileHandler:
    """Handle file operations for invoice and MPO data."""
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize FileHandler with base path.
        
        Args:
            base_path: Base directory path for the project
        """
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
    
    def find_files(self, program: str, year: int, month: int, 
                   file_type: str = "all") -> Dict[str, List[Path]]:
        """
        Find files for a specific program, year, and month.
        
        Args:
            program: Program name (e.g., 'focusedfox')
            year: Year (e.g., 2025)
            month: Month (1-12)
            file_type: 'mpo', 'invoice', or 'all'
            
        Returns:
            Dictionary with categorized file paths
        """
        files = {
            'mpo': [],
            'invoice': [],
            'all': []
        }
        
        # Format month with leading zero
        month_str = f"{month:02d}"
        
        # Find MPO files
        if file_type in ['mpo', 'all']:
            mpo_path = self.raw_path / 'mpo' / program / str(year) / month_str
            if mpo_path.exists():
                files['mpo'] = list(mpo_path.glob("*.xlsx"))
                files['all'].extend(files['mpo'])
                logger.info(f"Found {len(files['mpo'])} MPO files for {program} {year}/{month_str}")
        
        # Find invoice files
        if file_type in ['invoice', 'all']:
            invoice_path = self.raw_path / 'invoices' / program / str(year) / month_str
            if invoice_path.exists():
                # Look for both CSV and Excel files
                csv_files = list(invoice_path.glob("*.csv"))
                xlsx_files = list(invoice_path.glob("*.xlsx"))
                files['invoice'] = csv_files + xlsx_files
                files['all'].extend(files['invoice'])
                logger.info(f"Found {len(files['invoice'])} invoice files for {program} {year}/{month_str}")
        
        return files
    
    def read_excel(self, file_path: Union[str, Path], 
                   sheet_name: Optional[Union[str, int]] = None) -> pd.DataFrame:
        """
        Read Excel file with error handling.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index to read
            
        Returns:
            DataFrame with Excel data
        """
        file_path = Path(file_path)
        
        try:
            if sheet_name is not None:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            logger.info(f"Successfully read {len(df)} rows from {file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path.name}: {e}")
            raise
    
    def read_csv(self, file_path: Union[str, Path], 
                 encoding: Optional[str] = None) -> pd.DataFrame:
        """
        Read CSV file with automatic encoding detection.
        
        Args:
            file_path: Path to CSV file
            encoding: Optional encoding specification
            
        Returns:
            DataFrame with CSV data
        """
        file_path = Path(file_path)
        
        encodings = [encoding] if encoding else ['utf-8', 'latin-1', 'cp1252']
        
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                logger.info(f"Successfully read {len(df)} rows from {file_path.name} using {enc} encoding")
                return df
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not read {file_path.name} with any supported encoding")
    
    def save_results(self, df: pd.DataFrame, program: str, year: int, 
                    month: int, suffix: str = "allocated") -> Path:
        """
        Save processed results to the processed directory.
        
        Args:
            df: DataFrame to save
            program: Program name
            year: Year
            month: Month
            suffix: File suffix (default: 'allocated')
            
        Returns:
            Path to saved file
        """
        # Create output directory if it doesn't exist
        output_dir = self.processed_path / program / str(year) / f"{month:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{program}_{year}{month:02d}_{suffix}_{timestamp}.csv"
        output_path = output_dir / filename
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        
        return output_path
    
    def get_latest_processed(self, program: str, year: int, 
                           month: int) -> Optional[Path]:
        """
        Get the most recent processed file for a program/year/month.
        
        Args:
            program: Program name
            year: Year
            month: Month
            
        Returns:
            Path to latest processed file or None if not found
        """
        processed_dir = self.processed_path / program / str(year) / f"{month:02d}"
        
        if not processed_dir.exists():
            return None
        
        files = list(processed_dir.glob("*.csv"))
        if not files:
            return None
        
        # Return the most recent file
        return max(files, key=lambda p: p.stat().st_mtime)