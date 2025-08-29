"""
Invoice Data Processor for handling Invoice Excel/CSV uploads from GitLab.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class InvoiceProcessor:
    """Process Invoice Detail files uploaded to GitLab repository."""
    
    def __init__(self):
        """Initialize Invoice Processor."""
        self.required_columns = [
            'Project String', 'Organization ID', 'Account ID', 'Invoice ID',
            'Invoice Date', 'BILL_FM_LN_LBL', 'Empl/Vendor ID', 
            'Billed Amt + Burdens', 'CLIN'
        ]
        self.numeric_columns = [
            'Billable Regular Hours', 'Transaction Amount', 
            'Total Burdens', 'Billed Amt + Burdens'
        ]
        self.date_columns = [
            'Invoice Date', 'Invoice Start Date', 'Invoice End Date',
            'Timesheet End Date'
        ]
    
    def process_invoices(self, df: pd.DataFrame, mpo_grouped: Optional[Dict] = None) -> pd.DataFrame:
        """
        Replicate Power Query Output transformation logic.
        
        Args:
            df: Raw Invoice DataFrame
            mpo_grouped: Optional MPO grouped data for reference
            
        Returns:
            Processed Invoice DataFrame
        """
        logger.info(f"Processing Invoice data with {len(df)} rows using Power Query logic")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Step 1: Apply Power Query column mapping
        column_mapping = {
            'Project String': 'Project_Code',
            'BILL_FM_LN_LBL': 'InvDescription',
            'Billed Amt + Burdens': 'Billed_Amt_Burdens',
            'Invoice ID': 'Actual_Invoice_ID',
            'Empl/Vendor ID': 'Empl_VendorID',
            'Empl/Vendor Name': 'Empl_VendorName',
            'Invoice Date': 'InvoiceDate',
            'Invoice End Date': 'InvoiceEndDate',
            'Invoice Start Date': 'InvoiceStartDate',
            'PLC Description': 'PLC_Description',
            'Account ID': 'Account_ID',
            'Organization ID': 'Org_ID',
            'Timesheet End Date': 'Timesheet_End_Date',
            'Billable Regular Hours': 'Billable_Regular_Hours',
            'Transaction Amount': 'Transaction_Amount',
            'Total Burdens': 'Total_Burdens',
            'COMMENTS': 'Comments'
        }
        
        processed_df = processed_df.rename(columns=column_mapping)
        
        # Step 2: Filter out null descriptions and Fee entries
        initial_rows = len(processed_df)
        processed_df = processed_df[
            (processed_df['InvDescription'].notna()) &
            (processed_df['InvDescription'] != '') &
            (~processed_df['InvDescription'].str.upper().str.strip().isin(['FEE']))
        ]
        filtered_rows = initial_rows - len(processed_df)
        logger.info(f"Filtered out {filtered_rows} rows (null descriptions and fees)")
        
        # Step 3: Add invoice index
        processed_df.reset_index(drop=True, inplace=True)
        processed_df['InvoiceID'] = range(1, len(processed_df) + 1)
        
        # Step 4: Convert dates to strings for processing
        date_columns = ['InvoiceDate', 'InvoiceEndDate', 'InvoiceStartDate', 'Timesheet_End_Date']
        for col in date_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
                # Keep as datetime for processing but add string version
                processed_df[f'{col}_str'] = processed_df[col].dt.strftime('%Y-%m-%d')
        
        # Step 5: Add month-year for matching
        if 'InvoiceStartDate' in processed_df.columns:
            processed_df['Invoice_Month_Year'] = processed_df['InvoiceStartDate'].dt.strftime('%Y-%m')
        else:
            processed_df['Invoice_Month_Year'] = pd.Timestamp.now().strftime('%Y-%m')
        
        # Step 6: Convert numeric columns
        numeric_columns = ['Billed_Amt_Burdens', 'Billable_Regular_Hours', 
                          'Transaction_Amount', 'Total_Burdens']
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
        
        # Step 7: Add additional processing from original logic
        processed_df = self._add_power_query_fields(processed_df)
        
        logger.info(f"Invoice processing complete: {len(processed_df)} rows")
        return processed_df
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing method for compatibility with ETL pipeline.
        
        Args:
            df: Raw Invoice DataFrame
            
        Returns:
            Processed Invoice DataFrame
        """
        return self.process_invoices(df)
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistency."""
        column_mapping = {
            # Standard mappings for variations
            'Project_String': 'Project String',
            'ProjectString': 'Project String',
            'Project Code': 'Project String',
            'Org_ID': 'Organization ID',
            'OrgID': 'Organization ID',
            'Account_ID': 'Account ID',
            'AccountID': 'Account ID',
            'Invoice_ID': 'Invoice ID',
            'InvoiceID': 'Invoice ID',
            'Invoice_Date': 'Invoice Date',
            'InvoiceDate': 'Invoice Date',
            'Labor_Type': 'BILL_FM_LN_LBL',
            'InvDescription': 'BILL_FM_LN_LBL',
            'Empl_VendorID': 'Empl/Vendor ID',
            'Empl_VendorName': 'Empl/Vendor Name',
            'Empl/VendorID': 'Empl/Vendor ID',
            'Empl/VendorName': 'Empl/Vendor Name',
            'Timesheet_End_Date': 'Timesheet End Date',
            'TimesheetEndDate': 'Timesheet End Date',
            'Billable_Regular_Hours': 'Billable Regular Hours',
            'BillableRegularHours': 'Billable Regular Hours',
            'Transaction_Amount': 'Transaction Amount',
            'TransactionAmount': 'Transaction Amount',
            'Total_Burdens': 'Total Burdens',
            'TotalBurdens': 'Total Burdens',
            'Billed_Amt_Burdens': 'Billed Amt + Burdens',
            'BilledAmtBurdens': 'Billed Amt + Burdens',
            'PLC_Description': 'PLC Description',
            'PLCDescription': 'PLC Description'
        }
        
        # Apply mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Create internal standardized names for processing
        df['InvoiceID'] = df['Invoice ID'].copy() if 'Invoice ID' in df.columns else None
        df['InvoiceDate'] = df['Invoice Date'].copy() if 'Invoice Date' in df.columns else None
        df['Empl_VendorID'] = df['Empl/Vendor ID'].copy() if 'Empl/Vendor ID' in df.columns else None
        df['Empl_VendorName'] = df['Empl/Vendor Name'].copy() if 'Empl/Vendor Name' in df.columns else None
        df['Billed_Amt_Burdens'] = df['Billed Amt + Burdens'].copy() if 'Billed Amt + Burdens' in df.columns else 0
        df['InvDescription'] = df['BILL_FM_LN_LBL'].copy() if 'BILL_FM_LN_LBL' in df.columns else ''
        df['Timesheet_End_Date'] = df['Timesheet End Date'].copy() if 'Timesheet End Date' in df.columns else None
        df['Project Code'] = df['Project String'].copy() if 'Project String' in df.columns else ''
        df['Account_ID'] = df['Account ID'].copy() if 'Account ID' in df.columns else ''
        df['Org_ID'] = df['Organization ID'].copy() if 'Organization ID' in df.columns else ''
        df['Transaction_Amount'] = df['Transaction Amount'].copy() if 'Transaction Amount' in df.columns else 0
        df['Total_Burdens'] = df['Total Burdens'].copy() if 'Total Burdens' in df.columns else 0
        df['Billable_Regular_Hours'] = df['Billable Regular Hours'].copy() if 'Billable Regular Hours' in df.columns else 0
        df['PLC_Description'] = df['PLC Description'].copy() if 'PLC Description' in df.columns else ''
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate Invoice data."""
        # Convert date columns
        for col in self.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['Billable_Regular_Hours', 'Transaction_Amount', 
                       'Total_Burdens', 'Billed_Amt_Burdens']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Ensure CLIN is string
        if 'CLIN' in df.columns:
            df['CLIN'] = df['CLIN'].astype(str).str.strip()
            df['CLIN'] = df['CLIN'].replace(['nan', 'None', ''], pd.NA)
        
        # Add SLIN if not present (may be derived from CLIN or separate)
        if 'SLIN' not in df.columns:
            # Try to extract SLIN from CLIN if it contains both
            if 'CLIN' in df.columns:
                df['SLIN'] = df['CLIN'].str.extract(r'\.(\d+)$', expand=False)
                df['CLIN'] = df['CLIN'].str.extract(r'^(\d+)', expand=False)
            else:
                df['SLIN'] = pd.NA
        else:
            df['SLIN'] = df['SLIN'].astype(str).str.strip()
            df['SLIN'] = df['SLIN'].replace(['nan', 'None', ''], pd.NA)
        
        # Ensure InvoiceID is integer
        if 'InvoiceID' in df.columns:
            df['InvoiceID'] = pd.to_numeric(df['InvoiceID'], errors='coerce')
            df['InvoiceID'] = df['InvoiceID'].fillna(0).astype(int)
        
        # Remove rows with no Invoice ID
        if 'InvoiceID' in df.columns:
            initial_rows = len(df)
            df = df[df['InvoiceID'] > 0]
            removed_rows = initial_rows - len(df)
            if removed_rows > 0:
                logger.warning(f"Removed {removed_rows} rows with invalid Invoice ID")
        
        return df
    
    def _process_labor_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and standardize labor type classifications."""
        if 'InvDescription' not in df.columns:
            df['InvDescription'] = 'UNKNOWN'
        
        # Store original descriptions
        df['Original_InvDescription'] = df['InvDescription'].copy()
        
        # Clean and normalize labor types
        df['InvDescription_Clean'] = df['InvDescription'].str.strip()
        df['InvDescription_normalized'] = df['InvDescription_Clean'].str.upper()
        
        # Create labor priority mapping
        labor_priority_map = {
            'DIRECT LABOR': 1,
            'DL OVERTIME': 2,
            'CONS/SUBS/EQUIPMENT': 3,
            'CONSULTANTS': 3,
            'SUBCONTRACTORS': 3,
            'EQUIPMENT': 3,
            'ODC': 4,
            'OTHER DIRECT COSTS': 4
        }
        
        # Apply priority mapping
        df['labor_priority'] = df['InvDescription_normalized'].map(labor_priority_map)
        
        # Handle variations
        df.loc[df['InvDescription_normalized'].str.contains('DIRECT LABOR', na=False), 'labor_priority'] = 1
        df.loc[df['InvDescription_normalized'].str.contains('OVERTIME', na=False), 'labor_priority'] = 2
        df.loc[df['InvDescription_normalized'].str.contains('CONS|SUB|EQUIP', na=False), 'labor_priority'] = 3
        
        # Fill unmapped with default priority
        df['labor_priority'] = df['labor_priority'].fillna(99)
        
        return df
    
    def _calculate_totals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate invoice totals and aggregations."""
        # Calculate actual invoice totals by Invoice ID
        if 'InvoiceID' in df.columns and 'Billed_Amt_Burdens' in df.columns:
            invoice_totals = df.groupby('InvoiceID')['Billed_Amt_Burdens'].sum()
            df['Invoice_Total'] = df['InvoiceID'].map(invoice_totals)
        
        # Calculate hours totals
        if 'InvoiceID' in df.columns and 'Billable_Regular_Hours' in df.columns:
            hours_totals = df.groupby('InvoiceID')['Billable_Regular_Hours'].sum()
            df['Invoice_Total_Hours'] = df['InvoiceID'].map(hours_totals)
        
        # Identify credit invoices
        df['Is_Credit'] = df['Billed_Amt_Burdens'] < 0
        
        # Calculate burden rate
        if 'Total_Burdens' in df.columns and 'Transaction_Amount' in df.columns:
            df['Burden_Rate'] = np.where(
                df['Transaction_Amount'] != 0,
                df['Total_Burdens'] / df['Transaction_Amount'],
                0
            )
        
        return df
    
    def _add_power_query_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fields that replicate Power Query transformation logic.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            DataFrame with Power Query fields added
        """
        # Store original descriptions (critical for allocation logic)
        df['Original_InvDescription'] = df['InvDescription'].copy()
        
        # Clean and normalize descriptions
        df['InvDescription_Clean'] = df['InvDescription'].str.strip()
        df['InvDescription_normalized'] = df['InvDescription_Clean'].str.upper()
        
        # Apply labor priority mapping (matches original allocation logic)
        labor_priority_map = {
            'DIRECT LABOR': 1,
            'DL OVERTIME': 2,
            'CONS/SUBS/EQUIPMENT': 3,
            'CONSULTANTS': 3,
            'SUBCONTRACTORS': 3,
            'EQUIPMENT': 3,
            'ODC': 4,
            'OTHER DIRECT COSTS': 4
        }
        
        df['labor_priority'] = df['InvDescription_normalized'].map(labor_priority_map)
        
        # Handle variations in labor descriptions
        df.loc[df['InvDescription_normalized'].str.contains('DIRECT LABOR', na=False), 'labor_priority'] = 1
        df.loc[df['InvDescription_normalized'].str.contains('OVERTIME', na=False), 'labor_priority'] = 2
        df.loc[df['InvDescription_normalized'].str.contains('CONS|SUB|EQUIP', na=False), 'labor_priority'] = 3
        
        # Fill unmapped with default priority
        df['labor_priority'] = df['labor_priority'].fillna(99)
        
        # Add MPO tracking field
        df['Has_MPO'] = 'No'  # Will be updated when merged with MPO data
        
        # Add processing metadata
        df['Processed_Date'] = pd.Timestamp.now()
        df['Data_Source'] = 'Invoice'
        
        # Initialize allocation tracking fields
        df['Allocated_Amount'] = 0
        df['Remaining_To_Allocate'] = df['Billed_Amt_Burdens'].copy()
        df['Allocation_Status'] = 'Pending'
        
        # Ensure CLIN and SLIN fields exist and are properly formatted
        if 'CLIN' not in df.columns:
            df['CLIN'] = ''
        else:
            df['CLIN'] = df['CLIN'].astype(str).str.strip()
        
        if 'SLIN' not in df.columns:
            # Try to extract from CLIN if it contains both
            if 'CLIN' in df.columns:
                df['SLIN'] = df['CLIN'].str.extract(r'\.(\d+)$', expand=False)
                df['CLIN'] = df['CLIN'].str.extract(r'^(\d+)', expand=False)
            else:
                df['SLIN'] = ''
        else:
            df['SLIN'] = df['SLIN'].astype(str).str.strip()
        
        return df
    
    def _add_tracking_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add tracking and audit fields."""
        # Add processing timestamp
        df['Processed_Date'] = pd.Timestamp.now()
        
        # Add unique identifier for each row
        df['Row_ID'] = range(1, len(df) + 1)
        
        # Add actual invoice ID for tracking
        if 'InvoiceID' in df.columns:
            df['Actual_Invoice_ID'] = df['InvoiceID'].astype(str)
        
        # Add data source indicator
        df['Data_Source'] = 'Invoice'
        
        # Initialize allocation fields
        df['Allocated_Amount'] = 0
        df['Remaining_To_Allocate'] = df['Billed_Amt_Burdens'].copy()
        df['Allocation_Status'] = 'Pending'
        
        # Add comments field for tracking
        df['Comments'] = ''
        
        return df
    
    def validate(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate processed Invoice data.
        
        Args:
            df: Processed Invoice DataFrame
            
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
        missing_columns = []
        for col in self.required_columns:
            if col not in df.columns and col.replace(' ', '_') not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            validation_results['warnings'].append(
                f"Missing columns: {missing_columns}"
            )
        
        # Check for duplicates
        if 'InvoiceID' in df.columns and 'Empl_VendorID' in df.columns:
            duplicates = df[df.duplicated(
                subset=['InvoiceID', 'Empl_VendorID', 'Timesheet_End_Date'], 
                keep=False
            )]
            if not duplicates.empty:
                validation_results['warnings'].append(
                    f"Found {len(duplicates)} potential duplicate entries"
                )
        
        # Check for negative values in hours
        if 'Billable_Regular_Hours' in df.columns:
            negative_hours = df[df['Billable_Regular_Hours'] < 0]
            if not negative_hours.empty:
                validation_results['warnings'].append(
                    f"Found {len(negative_hours)} rows with negative hours"
                )
        
        # Generate statistics
        validation_results['stats'] = {
            'total_rows': len(df),
            'unique_invoices': df['InvoiceID'].nunique() if 'InvoiceID' in df.columns else 0,
            'unique_employees': df['Empl_VendorID'].nunique() if 'Empl_VendorID' in df.columns else 0,
            'total_billed': df['Billed_Amt_Burdens'].sum() if 'Billed_Amt_Burdens' in df.columns else 0,
            'total_hours': df['Billable_Regular_Hours'].sum() if 'Billable_Regular_Hours' in df.columns else 0,
            'credit_invoices': df['Is_Credit'].sum() if 'Is_Credit' in df.columns else 0
        }
        
        return validation_results