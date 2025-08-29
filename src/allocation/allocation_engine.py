"""
Allocation Engine for processing invoice allocations against MPO funding.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime


logger = logging.getLogger(__name__)


class AllocationEngine:
    """
    Core allocation engine that processes invoices against available MPO funding.
    Implements chronological allocation with labor coupling rules.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize allocation engine.
        
        Args:
            config: Configuration dictionary with allocation rules
        """
        self.config = config or {}
        
        # Labor types that should be coupled
        self.coupled_labor_types = {
            'DIRECT LABOR',
            'DL OVERTIME'
        }
        
        # Initialize tracking structures
        self.reset_tracking()
    
    def reset_tracking(self):
        """Reset all tracking variables for a new allocation run."""
        self.funding_tracker = {}
        self.exhausted_clins = set()
        self.labor_overtime_assignments = {}  # (emp_id, timesheet_date, clin) -> slin
        self.clin_slin_sequence = {}
        self.allocation_sequence = 1
        
    def allocate_invoices(self, invoices_df: pd.DataFrame, mpo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main allocation logic - replaces Power Query Python script.
        
        Args:
            invoices_df: Processed invoice DataFrame
            mpo_df: Processed MPO DataFrame
            
        Returns:
            DataFrame with allocation results
        """
        logger.info(f"Starting allocation process: {len(invoices_df)} invoice rows, {len(mpo_df)} MPO rows")
        
        # Reset tracking for this run
        self.reset_tracking()
        
        # Join invoices with MPO data on CLIN and Month-Year
        merged_df = self._join_with_mpo(invoices_df, mpo_df)
        
        # Sort for chronological processing
        merged_df = merged_df.sort_values(['CLIN', 'SLIN', 'InvoiceDate', 'InvoiceID', 'MPO_Index'])
        
        # Initialize tracking structures
        funding_tracker = {}
        exhausted_clins = set()
        result_rows = []
        absolute_sequence = 1
        clin_slin_sequence = {}
        
        # Group by InvoiceID for processing
        grouped = merged_df.groupby('InvoiceID')
        
        # Process invoices chronologically
        invoice_order = merged_df[['InvoiceID', 'InvoiceDate']].drop_duplicates().sort_values('InvoiceDate')
        ordered_invoice_ids = invoice_order['InvoiceID'].tolist()
        
        for invoice_id in ordered_invoice_ids:
            if invoice_id not in grouped.groups:
                continue
            
            group = grouped.get_group(invoice_id)
            group_results = self._process_invoice_group(
                group, funding_tracker, exhausted_clins, 
                clin_slin_sequence, absolute_sequence
            )
            result_rows.extend(group_results)
            absolute_sequence += len(group_results)
        
        # Create output dataframe
        result_df = pd.DataFrame(result_rows)
        return self._format_output(result_df)
    
    def allocate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compatibility method for ETL pipeline integration.
        
        Args:
            df: Combined DataFrame with MPO and Invoice data
            
        Returns:
            DataFrame with allocation results
        """
        # Separate MPO and Invoice data if combined
        if 'file_type' in df.columns:
            mpo_df = df[df['file_type'] == 'MPO'].copy()
            invoice_df = df[df['file_type'] == 'Invoice'].copy()
        else:
            # Assume all data is invoice data for now
            invoice_df = df.copy()
            mpo_df = pd.DataFrame()  # Empty MPO data
        
        return self.allocate_invoices(invoice_df, mpo_df)
    
    def _join_with_mpo(self, invoices_df: pd.DataFrame, mpo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join invoice data with MPO data on CLIN and Month-Year.
        
        Args:
            invoices_df: Processed invoice DataFrame
            mpo_df: Processed MPO DataFrame
            
        Returns:
            Merged DataFrame
        """
        if mpo_df.empty:
            logger.warning("No MPO data available for joining")
            # Add required MPO columns with defaults
            invoices_df['AvailableFunding'] = 0
            invoices_df['MPO_Index'] = 0
            invoices_df['Accum Total'] = 0
            return invoices_df
        
        # Prepare DataFrames for joining
        invoices_for_join = invoices_df.copy()
        mpo_for_join = mpo_df.copy()
        
        # Ensure consistent column names
        if 'Invoice_Month_Year' not in invoices_for_join.columns:
            if 'InvoiceStartDate' in invoices_for_join.columns:
                invoices_for_join['Invoice_Month_Year'] = pd.to_datetime(
                    invoices_for_join['InvoiceStartDate']
                ).dt.strftime('%Y-%m')
            else:
                invoices_for_join['Invoice_Month_Year'] = pd.Timestamp.now().strftime('%Y-%m')
        
        if 'MPO_Month_Year' not in mpo_for_join.columns:
            if 'InvoiceStartDate' in mpo_for_join.columns:
                mpo_for_join['MPO_Month_Year'] = pd.to_datetime(
                    mpo_for_join['InvoiceStartDate']
                ).dt.strftime('%Y-%m')
            else:
                mpo_for_join['MPO_Month_Year'] = pd.Timestamp.now().strftime('%Y-%m')
        
        # Join on CLIN and Month-Year
        merged_df = pd.merge(
            invoices_for_join,
            mpo_for_join[['CLIN', 'SLIN', 'AvailableFunding', 'MPO_Index', 'MPO_Month_Year']].rename(columns={'AvailableFunding': 'Accum Total'}),
            left_on=['CLIN', 'Invoice_Month_Year'],
            right_on=['CLIN', 'MPO_Month_Year'],
            how='left',
            suffixes=('', '_mpo')
        )
        
        # Fill missing values for unmatched records
        merged_df['Accum Total'] = merged_df['Accum Total'].fillna(0)
        merged_df['MPO_Index'] = merged_df['MPO_Index'].fillna(0)
        merged_df['SLIN_mpo'] = merged_df['SLIN_mpo'].fillna('')
        
        # Use MPO SLIN if available, otherwise use Invoice SLIN
        merged_df['SLIN'] = merged_df['SLIN_mpo'].where(
            merged_df['SLIN_mpo'] != '', 
            merged_df['SLIN']
        )
        
        logger.info(f"Joined data: {len(merged_df)} rows")
        return merged_df
    
    def _process_invoice_group(self, group: pd.DataFrame, funding_tracker: dict, 
                              exhausted_clins: set, clin_slin_sequence: dict, 
                              absolute_sequence: int) -> List[Dict]:
        """
        Process a single invoice group with labor coupling.
        
        Args:
            group: Invoice group DataFrame
            funding_tracker: Funding tracking dictionary
            exhausted_clins: Set of exhausted CLIN/SLIN keys
            clin_slin_sequence: Sequence tracking dictionary
            absolute_sequence: Current absolute sequence number
            
        Returns:
            List of allocation result dictionaries
        """
        invoice_amount = group['Billed_Amt_Burdens'].iloc[0]
        billable_hours_total = group['Billable_Regular_Hours'].iloc[0]
        
        # Group by employee and timesheet for coupling
        person_timesheet_groups = group.groupby(['Empl_VendorID', 'Timesheet_End_Date'])
        
        # Sort groups for processing
        grouped_records = []
        for (emp_id, timesheet_date), person_group in person_timesheet_groups:
            person_group_sorted = person_group.sort_values(['CLIN', 'SLIN'])
            grouped_records.append((emp_id, timesheet_date, person_group_sorted))
        
        grouped_records.sort(key=lambda x: (
            x[2]['CLIN'].iloc[0],
            x[2]['SLIN'].iloc[0] if not x[2]['SLIN'].isna().all() else 'ZZZ',
            x[1],  # Timesheet date
            x[0]   # Employee ID
        ))
        
        remaining_amount = invoice_amount
        remaining_hours = billable_hours_total
        is_credit = invoice_amount < 0
        invoice_allocation_sequence = 1
        allocation_results = []
        
        for emp_id, timesheet_date, person_group in grouped_records:
            if abs(remaining_amount) < 0.01:
                break
            
            for idx, row in person_group.iterrows():
                clin = str(row['CLIN'])
                slin = str(row['SLIN']) if pd.notna(row['SLIN']) else ''
                clin_slin_key = (clin, slin)
                
                # Initialize funding if new CLIN/SLIN
                if clin_slin_key not in funding_tracker:
                    funding_tracker[clin_slin_key] = row.get('Accum Total', 0)
                    clin_slin_sequence[clin_slin_key] = 0
                
                # Skip exhausted CLINs
                if clin_slin_key in exhausted_clins:
                    continue
                
                available_funding = funding_tracker[clin_slin_key]
                pre_allocation_funding = available_funding
                
                # Skip if no funding or nothing to allocate
                if (not is_credit and available_funding <= 0) or abs(remaining_amount) < 0.01:
                    if available_funding <= 0:
                        exhausted_clins.add(clin_slin_key)
                    continue
                
                # Calculate allocation
                if is_credit:
                    allocated_amount = -min(abs(remaining_amount), abs(available_funding)) if available_funding < 0 else remaining_amount
                else:
                    allocated_amount = min(remaining_amount, available_funding)
                
                allocation_ratio = abs(allocated_amount / invoice_amount) if abs(invoice_amount) > 0.01 else 0
                allocated_hours = allocation_ratio * billable_hours_total
                
                # Update trackers
                funding_tracker[clin_slin_key] = funding_tracker[clin_slin_key] - allocated_amount
                
                if funding_tracker[clin_slin_key] <= 0:
                    exhausted_clins.add(clin_slin_key)
                
                remaining_amount = remaining_amount - allocated_amount
                remaining_hours = remaining_hours - allocated_hours
                
                clin_slin_sequence[clin_slin_key] += 1
                sequence_str = f'{clin}-{slin}-{clin_slin_sequence[clin_slin_key]:03d}'
                
                is_split = 'Yes' if abs(allocated_amount) < abs(invoice_amount) else 'No'
                
                # Create allocation record - PRESERVE InvDescription
                allocation_results.append({
                    'Project_Code': row.get('Project_Code', ''),
                    'CLIN': clin,
                    'SLIN': slin,
                    'Invoice_Total': row.get('Accum Total', 0),
                    'Invoice_Amount': invoice_amount,
                    'Allocated_Amount': allocated_amount,
                    'Remaining_Invoice': funding_tracker[clin_slin_key],
                    'Pre_Allocation_Funding': pre_allocation_funding,
                    'Split_Record': is_split,
                    'Is_Credit': 'Yes' if is_credit else 'No',
                    'Invoice_Label': row.get('InvDescription', ''),  # PRESERVE ORIGINAL LABEL
                    'Actual_Invoice_ID': row.get('Actual_Invoice_ID', ''),
                    'Vendor_Name': row.get('Empl_VendorName', ''),
                    'Emp_ID': row.get('Empl_VendorID', ''),
                    'PLC': row.get('PLC', ''),
                    'PLC_Description': row.get('PLC_Description', ''),
                    'Account_ID': row.get('Account_ID', ''),
                    'Org_ID': row.get('Org_ID', ''),
                    'Billable_Regular_Hours': billable_hours_total,
                    'Allocated_Billable_Hours': allocated_hours,
                    'Total_Burdens': row.get('Total_Burdens', 0),
                    'Transaction_Amount': row.get('Transaction_Amount', 0),
                    'Invoice_Date': row.get('InvoiceDate'),
                    'Timesheet_End_Date': row.get('Timesheet_End_Date'),
                    'Comments': row.get('Comments', ''),
                    'Absolute_Sequence': absolute_sequence,
                    'Allocation_Sequence': sequence_str,
                    'Invoice_Seq': invoice_allocation_sequence
                })
                
                absolute_sequence += 1
                invoice_allocation_sequence += 1
                
                if abs(remaining_amount) < 0.01:
                    break
        
        return allocation_results
    
    def _format_output(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the output DataFrame to match expected structure.
        
        Args:
            result_df: Raw allocation results DataFrame
            
        Returns:
            Formatted DataFrame
        """
        if result_df.empty:
            return result_df
        
        # Sort by absolute sequence
        result_df = result_df.sort_values('Absolute_Sequence')
        
        # Add any missing columns with defaults
        expected_columns = {
            'Project_Code': '',
            'CLIN': '',
            'SLIN': '',
            'Invoice_Total': 0,
            'Invoice_Amount': 0,
            'Allocated_Amount': 0,
            'Remaining_Invoice': 0,
            'Pre_Allocation_Funding': 0,
            'Split_Record': 'No',
            'Is_Credit': 'No',
            'Invoice_Label': '',
            'Actual_Invoice_ID': '',
            'Vendor_Name': '',
            'Emp_ID': '',
            'PLC': '',
            'PLC_Description': '',
            'Account_ID': '',
            'Org_ID': '',
            'Billable_Regular_Hours': 0,
            'Allocated_Billable_Hours': 0,
            'Total_Burdens': 0,
            'Transaction_Amount': 0,
            'Invoice_Date': pd.NaT,
            'Timesheet_End_Date': pd.NaT,
            'Comments': '',
            'Absolute_Sequence': 0,
            'Allocation_Sequence': '',
            'Invoice_Seq': 0
        }
        
        for col, default_val in expected_columns.items():
            if col not in result_df.columns:
                result_df[col] = default_val
        
        return result_df
    
    def _prepare_for_allocation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for allocation processing.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Prepared DataFrame
        """
        logger.info("Preparing data for allocation...")
        
        # Ensure required columns exist
        required_columns = {
            'InvoiceID': 0,
            'CLIN': '',
            'SLIN': '',
            'AvailableFunding': 0,
            'Billed_Amt_Burdens': 0,
            'InvDescription': 'UNKNOWN',
            'Empl_VendorID': 'UNKNOWN',
            'Timesheet_End_Date': pd.NaT,
            'Billable_Regular_Hours': 0,
            'InvoiceDate': pd.NaT,
            'Has_MPO': 'No'
        }
        
        for col, default in required_columns.items():
            if col not in df.columns:
                df[col] = default
        
        # Store original descriptions
        df['Original_InvDescription'] = df['InvDescription'].copy()
        
        # Clean and normalize descriptions
        df['InvDescription_Clean'] = df['InvDescription'].str.strip()
        df['InvDescription_normalized'] = df['InvDescription_Clean'].str.upper()
        
        # Apply labor priority
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
        
        df['labor_priority'] = df['InvDescription_normalized'].map(labor_priority_map).fillna(99)
        
        # Handle variations in labor descriptions
        df.loc[df['InvDescription_normalized'].str.contains('DIRECT LABOR', na=False), 'labor_priority'] = 1
        df.loc[df['InvDescription_normalized'].str.contains('OVERTIME', na=False), 'labor_priority'] = 2
        df.loc[df['InvDescription_normalized'].str.contains('CONS|SUB|EQUIP', na=False), 'labor_priority'] = 3
        
        return df
    
    def _sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort DataFrame chronologically for allocation processing.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Chronologically sorted DataFrame
        """
        sort_columns = [
            'InvoiceDate', 'CLIN', 'SLIN', 'MPO_Index', 
            'InvoiceID', 'labor_priority'
        ]
        
        # Only use columns that exist
        existing_sort_columns = [col for col in sort_columns if col in df.columns]
        
        return df.sort_values(existing_sort_columns) if existing_sort_columns else df
    
    def _process_allocations(self, df: pd.DataFrame) -> List[Dict]:
        """
        Process allocations chronologically by invoice.
        
        Args:
            df: Sorted DataFrame ready for processing
            
        Returns:
            List of allocation result dictionaries
        """
        result_rows = []
        
        # Group by InvoiceID for processing
        grouped = df.groupby('InvoiceID')
        
        # Get chronological order of invoices
        invoice_order = df[['InvoiceID', 'InvoiceDate']].drop_duplicates().sort_values('InvoiceDate')
        ordered_invoice_ids = invoice_order['InvoiceID'].tolist()
        
        for invoice_id in ordered_invoice_ids:
            if invoice_id not in grouped.groups or invoice_id == 0:
                continue
            
            invoice_results = self._process_invoice(grouped.get_group(invoice_id))
            result_rows.extend(invoice_results)
        
        return result_rows
    
    def _process_invoice(self, invoice_group: pd.DataFrame) -> List[Dict]:
        """
        Process a single invoice for allocation.
        
        Args:
            invoice_group: All rows for a single invoice
            
        Returns:
            List of allocation results for this invoice
        """
        invoice_id = invoice_group['InvoiceID'].iloc[0]
        invoice_amount = invoice_group['Billed_Amt_Burdens'].iloc[0]
        billable_hours_total = invoice_group['Billable_Regular_Hours'].iloc[0]
        
        # Skip if no SLIN data
        if invoice_group['SLIN'].isna().all() or (invoice_group['SLIN'] == '').all():
            return []
        
        # Process person/timesheet groups
        person_groups = self._group_by_person_timesheet(invoice_group)
        
        # Initialize invoice-level tracking
        remaining_amount = invoice_amount
        remaining_hours = billable_hours_total
        is_credit = invoice_amount < 0
        invoice_seq = 1
        
        invoice_results = []
        
        # Process each person/timesheet group
        for person_group in person_groups:
            # Check if we should continue processing
            if abs(remaining_amount) < 0.01:
                has_labor_overtime = any(person_group['InvDescription_normalized'].isin(['DIRECT LABOR', 'DL OVERTIME']))
                if not has_labor_overtime:
                    break
            
            # Process each row in the person group
            for idx, row in person_group.iterrows():
                if abs(remaining_amount) < 0.01:
                    # Check if remaining rows have labor/overtime
                    remaining_rows = person_group[person_group.index > idx]
                    has_remaining_labor = any(remaining_rows['InvDescription_normalized'].isin(['DIRECT LABOR', 'DL OVERTIME']))
                    if not has_remaining_labor:
                        break
                
                # Process this row
                allocation_result = self._process_allocation_row(
                    row, remaining_amount, remaining_hours, 
                    invoice_amount, billable_hours_total, 
                    is_credit, invoice_seq
                )
                
                if allocation_result:
                    invoice_results.append(allocation_result)
                    
                    # Update remaining amounts
                    remaining_amount -= allocation_result['Allocated_Amount']
                    remaining_hours -= allocation_result['Allocated_Billable_Hours']
                    invoice_seq += 1
        
        return invoice_results
    
    def _group_by_person_timesheet(self, invoice_group: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Group invoice rows by person and timesheet date.
        
        Args:
            invoice_group: Rows for a single invoice
            
        Returns:
            List of person/timesheet group DataFrames, sorted appropriately
        """
        person_groups = []
        
        grouped = invoice_group.groupby(['Empl_VendorID', 'Timesheet_End_Date'])
        
        for (emp_id, timesheet_date), person_group in grouped:
            # Sort by CLIN, SLIN, labor priority
            sorted_group = person_group.sort_values(['CLIN', 'SLIN', 'labor_priority'])
            person_groups.append(sorted_group)
        
        # Sort groups by CLIN, SLIN, timesheet date, employee ID
        person_groups.sort(key=lambda group: (
            group['CLIN'].iloc[0],
            group['SLIN'].iloc[0] if not group['SLIN'].isna().all() else 'ZZZ',
            group['Timesheet_End_Date'].iloc[0] if pd.notna(group['Timesheet_End_Date'].iloc[0]) else pd.Timestamp.min,
            group['Empl_VendorID'].iloc[0]
        ))
        
        return person_groups
    
    def _process_allocation_row(self, row: pd.Series, remaining_amount: float, 
                               remaining_hours: float, invoice_amount: float, 
                               billable_hours_total: float, is_credit: bool, 
                               invoice_seq: int) -> Optional[Dict]:
        """
        Process allocation for a single row.
        
        Args:
            row: Data row to process
            remaining_amount: Remaining invoice amount to allocate
            remaining_hours: Remaining hours to allocate
            invoice_amount: Total invoice amount
            billable_hours_total: Total billable hours
            is_credit: Whether invoice is a credit
            invoice_seq: Sequence number within invoice
            
        Returns:
            Allocation result dictionary or None if no allocation
        """
        clin = str(row['CLIN'])
        slin = str(row['SLIN']) if pd.notna(row['SLIN']) else ''
        
        if not clin or clin == 'nan' or not slin or slin == 'nan':
            return None
        
        # Apply coupling logic
        emp_id = row['Empl_VendorID']
        timesheet_date = row['Timesheet_End_Date']
        inv_description_norm = row['InvDescription_normalized']
        
        original_slin = slin
        slin = self._apply_coupling(
            emp_id, timesheet_date, clin, slin, inv_description_norm
        )
        
        # Initialize funding tracking
        clin_slin_key = (clin, slin)
        if clin_slin_key not in self.funding_tracker:
            self.funding_tracker[clin_slin_key] = row['AvailableFunding']
            self.clin_slin_sequence[clin_slin_key] = 0
        
        available_funding = self.funding_tracker[clin_slin_key]
        pre_allocation_funding = available_funding
        
        # Calculate allocation
        allocated_amount, allocated_hours = self._calculate_allocation(
            remaining_amount, invoice_amount, available_funding, 
            billable_hours_total, is_credit
        )
        
        # Update funding tracker
        if allocated_amount != 0:
            self.funding_tracker[clin_slin_key] -= allocated_amount
            
            if self.funding_tracker[clin_slin_key] <= 0:
                self.exhausted_clins.add(clin_slin_key)
        
        # Update sequence
        self.clin_slin_sequence[clin_slin_key] += 1
        sequence_str = f'{clin}-{slin}-{self.clin_slin_sequence[clin_slin_key]:03d}'
        
        # Determine if this should be included in results
        is_split = 'Yes' if abs(allocated_amount) < abs(invoice_amount) and abs(allocated_amount) > 0.01 else 'No'
        is_coupled = 'Yes' if inv_description_norm in ['DIRECT LABOR', 'DL OVERTIME'] else 'No'
        
        should_include = (
            abs(allocated_amount) > 0.01 or 
            is_coupled == 'Yes' or 
            row.get('Has_MPO', 'No') == 'Yes'
        )
        
        if should_include:
            return self._create_allocation_result(
                row, clin, slin, original_slin, allocated_amount, 
                allocated_hours, pre_allocation_funding, is_split, 
                is_coupled, is_credit, sequence_str, invoice_seq
            )
        
        return None
    
    def _calculate_allocation(self, remaining_amount: float, invoice_amount: float, 
                            available_funding: float, billable_hours_total: float, 
                            is_credit: bool) -> Tuple[float, float]:
        """
        Calculate allocation amounts and hours.
        
        Args:
            remaining_amount: Remaining invoice amount
            invoice_amount: Total invoice amount
            available_funding: Available funding for CLIN/SLIN
            billable_hours_total: Total billable hours
            is_credit: Whether this is a credit invoice
            
        Returns:
            Tuple of (allocated_amount, allocated_hours)
        """
        allocated_amount = 0
        allocated_hours = 0
        
        if abs(remaining_amount) >= 0.01 and (is_credit or available_funding > 0):
            if is_credit:
                allocated_amount = (
                    -min(abs(remaining_amount), abs(available_funding)) 
                    if available_funding < 0 
                    else remaining_amount
                )
            else:
                allocated_amount = min(remaining_amount, available_funding)
            
            # Calculate proportional hours
            if abs(invoice_amount) > 0.01:
                allocation_ratio = abs(allocated_amount / invoice_amount)
                allocated_hours = allocation_ratio * billable_hours_total
        
        return allocated_amount, allocated_hours
    
    def _create_allocation_result(self, row: pd.Series, clin: str, slin: str, 
                                 original_slin: str, allocated_amount: float, 
                                 allocated_hours: float, pre_allocation_funding: float,
                                 is_split: str, is_coupled: str, is_credit: bool,
                                 sequence_str: str, invoice_seq: int) -> Dict:
        """
        Create allocation result dictionary.
        
        Args:
            row: Source data row
            clin: Contract line item number
            slin: Sub-line item number
            original_slin: Original SLIN before coupling
            allocated_amount: Amount allocated
            allocated_hours: Hours allocated
            pre_allocation_funding: Funding before allocation
            is_split: Whether record is split
            is_coupled: Whether record is coupled
            is_credit: Whether invoice is credit
            sequence_str: Allocation sequence string
            invoice_seq: Invoice sequence number
            
        Returns:
            Allocation result dictionary
        """
        return {
            'Project_Code': row.get('Project Code', ''),
            'CLIN': clin,
            'SLIN': slin,
            'Original_SLIN': original_slin if original_slin != slin else slin,
            'Invoice_Total': row.get('AvailableFunding', 0),
            'Invoice_Amount': row.get('Billed_Amt_Burdens', 0),
            'Allocated_Amount': allocated_amount,
            'Remaining_Funding': self.funding_tracker.get((clin, slin), 0),
            'Pre_Allocation_Funding': pre_allocation_funding,
            'Split_Record': is_split,
            'Is_Credit': 'Yes' if is_credit else 'No',
            'Is_Coupled': is_coupled,
            'Labor_Type': row.get('InvDescription_Clean', ''),
            'Invoice': row.get('Original_InvDescription', ''),
            'Debug_Label': row.get('InvDescription', ''),
            'Invoice_ID_Debug': row.get('InvoiceID', 0),
            'Actual_Invoice_ID': row.get('InvoiceID', 0),
            'Vendor_Name': row.get('Empl_VendorName', ''),
            'Emp_ID': row.get('Empl_VendorID', ''),
            'PLC': row.get('PLC', ''),
            'PLC_Description': row.get('PLC_Description', ''),
            'Account_ID': row.get('Account_ID', ''),
            'Org_ID': row.get('Org_ID', ''),
            'Billable_Regular_Hours': row.get('Billable_Regular_Hours', 0),
            'Allocated_Billable_Hours': allocated_hours,
            'Total_Burdens': row.get('Total_Burdens', 0),
            'Transaction_Amount': row.get('Transaction_Amount', 0),
            'Invoice_Date': row.get('InvoiceDate'),
            'Timesheet_End_Date': row.get('Timesheet_End_Date'),
            'Comments': row.get('Comments', ''),
            'Has_MPO': row.get('Has_MPO', 'No'),
            'Absolute_Sequence': self.allocation_sequence,
            'Allocation_Sequence': sequence_str,
            'Invoice_Seq': invoice_seq
        }
    
    def get_allocation_summary(self) -> Dict:
        """
        Get summary of allocation results.
        
        Returns:
            Summary statistics dictionary
        """
        return {
            'total_clins_processed': len(self.funding_tracker),
            'exhausted_clins': len(self.exhausted_clins),
            'labor_assignments': len(self.labor_overtime_assignments),
            'total_allocations': self.allocation_sequence - 1,
            'remaining_funding': {
                f"{clin}-{slin}": funding 
                for (clin, slin), funding in self.funding_tracker.items()
                if funding > 0
            }
        }
    
    # ========== COUPLING LOGIC METHODS (merged from coupling_logic.py) ==========
    
    def _apply_coupling(self, emp_id: str, timesheet_date: str, clin: str, 
                       slin: str, inv_description_norm: str) -> str:
        """
        Apply coupling logic to determine the correct SLIN for labor entries.
        Ensures Direct Labor and DL Overtime for same employee/timesheet/CLIN
        are allocated to the same SLIN.
        
        Args:
            emp_id: Employee/Vendor ID
            timesheet_date: Timesheet end date
            clin: Contract line item number
            slin: Sub-line item number
            inv_description_norm: Normalized invoice description
            
        Returns:
            The SLIN that should be used (may be different from input)
        """
        # Only apply coupling for labor types
        if inv_description_norm not in self.coupled_labor_types:
            return slin
        
        # Create assignment key
        assignment_key = (str(emp_id), str(timesheet_date), str(clin))
        
        if inv_description_norm == 'DL OVERTIME':
            # For overtime, use the SLIN assigned to Direct Labor if it exists
            if assignment_key in self.labor_overtime_assignments:
                assigned_slin = self.labor_overtime_assignments[assignment_key]
                logger.debug(f"Coupling overtime to Direct Labor SLIN: {assigned_slin} for {assignment_key}")
                return assigned_slin
            else:
                # No Direct Labor found yet, use the overtime SLIN and remember it
                self.labor_overtime_assignments[assignment_key] = slin
                logger.debug(f"Recording overtime SLIN: {slin} for {assignment_key}")
                return slin
        
        elif inv_description_norm == 'DIRECT LABOR':
            # For Direct Labor, record the assignment for future overtime entries
            if assignment_key in self.labor_overtime_assignments:
                # There was already an assignment (probably from overtime)
                existing_slin = self.labor_overtime_assignments[assignment_key]
                logger.debug(f"Direct Labor found existing assignment: {existing_slin} for {assignment_key}")
                return existing_slin
            else:
                # New Direct Labor assignment
                self.labor_overtime_assignments[assignment_key] = slin
                logger.debug(f"Recording Direct Labor SLIN: {slin} for {assignment_key}")
                return slin
        
        return slin
    
    def _is_coupled_labor_type(self, inv_description_norm: str) -> bool:
        """
        Check if a labor type should be subject to coupling rules.
        
        Args:
            inv_description_norm: Normalized invoice description
            
        Returns:
            True if the labor type should be coupled
        """
        return inv_description_norm in self.coupled_labor_types