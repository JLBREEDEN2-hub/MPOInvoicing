#!/usr/bin/env python3
"""
Show the expected output structure by creating a sample allocation record.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_sample_allocation_output():
    """Create a sample allocation output to show the expected structure."""
    print("📋 EXPECTED ALLOCATION OUTPUT STRUCTURE")
    print("=" * 60)
    
    # Create sample allocation records based on the system design
    sample_records = [
        {
            'Project_Code': '25524.0003.0001.LABR',
            'CLIN': '2001',
            'SLIN': 'AA',
            'Original_SLIN': 'AA',
            'Invoice_Total': 100000.00,
            'Invoice_Amount': 5000.00,
            'Allocated_Amount': 5000.00,
            'Remaining_Invoice': 95000.00,
            'Pre_Allocation_Funding': 100000.00,
            'Split_Record': 'No',
            'Is_Credit': 'No',
            'Is_Coupled': 'Yes',
            'Labor_Type': 'Direct Labor',
            'Invoice_Label': 'Direct Labor',  # PRESERVED ORIGINAL LABEL
            'Invoice_ID_Debug': 1,
            'Unique_Count': 3,
            'Actual_Invoice_ID': 'INV-0006185914',
            'Vendor_Name': 'ZEBRON, ERIN',
            'Emp_ID': 'HB3267',
            'PLC': 'SPAA',
            'PLC_Description': 'Sr Prgm Admin Asst',
            'Account_ID': '5017-01',
            'Org_ID': '1.25.33795.K6',
            'Billable_Regular_Hours': 40.0,
            'Allocated_Billable_Hours': 40.0,
            'Total_Burdens': 2000.00,
            'Transaction_Amount': 3000.00,
            'Invoice_Date': '2025-05-06',
            'Timesheet_End_Date': '2025-04-30',
            'Comments': '',
            'Has_MPO': 'Yes',
            'Absolute_Sequence': 1,
            'Allocation_Sequence': '2001-AA-001',
            'Invoice_Seq': 1
        },
        {
            'Project_Code': '25524.0003.0001.LABR',
            'CLIN': '2001',
            'SLIN': 'AA',
            'Original_SLIN': 'AA',
            'Invoice_Total': 95000.00,
            'Invoice_Amount': 8000.00,
            'Allocated_Amount': 8000.00,
            'Remaining_Invoice': 87000.00,
            'Pre_Allocation_Funding': 95000.00,
            'Split_Record': 'No',
            'Is_Credit': 'No',
            'Is_Coupled': 'Yes',
            'Labor_Type': 'DL Overtime',
            'Invoice_Label': 'DL Overtime',  # PRESERVED ORIGINAL LABEL
            'Invoice_ID_Debug': 2,
            'Unique_Count': 3,
            'Actual_Invoice_ID': 'INV-0006185915',
            'Vendor_Name': 'ZEBRON, ERIN',
            'Emp_ID': 'HB3267',
            'PLC': 'SPAA',
            'PLC_Description': 'Sr Prgm Admin Asst',
            'Account_ID': '5017-01',
            'Org_ID': '1.25.33795.K6',
            'Billable_Regular_Hours': 50.0,
            'Allocated_Billable_Hours': 50.0,
            'Total_Burdens': 3200.00,
            'Transaction_Amount': 4800.00,
            'Invoice_Date': '2025-05-06',
            'Timesheet_End_Date': '2025-04-30',
            'Comments': '',
            'Has_MPO': 'Yes',
            'Absolute_Sequence': 2,
            'Allocation_Sequence': '2001-AA-002',
            'Invoice_Seq': 2
        },
        {
            'Project_Code': '25524.0003.0001.LABR',
            'CLIN': '2003',
            'SLIN': 'BB',
            'Original_SLIN': 'BB',
            'Invoice_Total': 150000.00,
            'Invoice_Amount': 12000.00,
            'Allocated_Amount': 12000.00,
            'Remaining_Invoice': 138000.00,
            'Pre_Allocation_Funding': 150000.00,
            'Split_Record': 'No',
            'Is_Credit': 'No',
            'Is_Coupled': 'No',
            'Labor_Type': 'Cons/Subs/Equipment',
            'Invoice_Label': 'Cons/Subs/Equipment',  # PRESERVED ORIGINAL LABEL
            'Invoice_ID_Debug': 3,
            'Unique_Count': 3,
            'Actual_Invoice_ID': 'INV-0006185916',
            'Vendor_Name': 'CONTRACTOR ABC',
            'Emp_ID': 'CONT001',
            'PLC': 'CONS',
            'PLC_Description': 'Consultant',
            'Account_ID': '5017-02',
            'Org_ID': '1.25.33795.K7',
            'Billable_Regular_Hours': 0.0,
            'Allocated_Billable_Hours': 0.0,
            'Total_Burdens': 0.00,
            'Transaction_Amount': 12000.00,
            'Invoice_Date': '2025-05-07',
            'Timesheet_End_Date': '2025-04-30',
            'Comments': 'Equipment purchase',
            'Has_MPO': 'Yes',
            'Absolute_Sequence': 3,
            'Allocation_Sequence': '2003-BB-001',
            'Invoice_Seq': 1
        },
        {
            'Project_Code': '25524.0003.0001.LABR',
            'CLIN': '2001',
            'SLIN': 'AB',
            'Original_SLIN': 'AB',
            'Invoice_Total': 75000.00,
            'Invoice_Amount': 15000.00,
            'Allocated_Amount': 10000.00,  # Partial allocation
            'Remaining_Invoice': 65000.00,
            'Pre_Allocation_Funding': 75000.00,
            'Split_Record': 'Yes',  # This is a split record
            'Is_Credit': 'No',
            'Is_Coupled': 'Yes',
            'Labor_Type': 'Direct Labor',
            'Invoice_Label': 'Direct Labor',  # PRESERVED ORIGINAL LABEL
            'Invoice_ID_Debug': 4,
            'Unique_Count': 3,
            'Actual_Invoice_ID': 'INV-0006185917',
            'Vendor_Name': 'WASSBERG, MARK',
            'Emp_ID': 'HB2985',
            'PLC': 'PML1',
            'PLC_Description': 'Program Mission Lead',
            'Account_ID': '5017-01',
            'Org_ID': '1.25.33795.K6',
            'Billable_Regular_Hours': 80.0,
            'Allocated_Billable_Hours': 53.3,  # Proportional allocation
            'Total_Burdens': 6000.00,
            'Transaction_Amount': 9000.00,
            'Invoice_Date': '2025-05-08',
            'Timesheet_End_Date': '2025-04-30',
            'Comments': '',
            'Has_MPO': 'Yes',
            'Absolute_Sequence': 4,
            'Allocation_Sequence': '2001-AB-001',
            'Invoice_Seq': 1
        },
        {
            'Project_Code': '25524.0003.0001.LABR',
            'CLIN': '2001',
            'SLIN': 'AC',
            'Original_SLIN': 'AC',
            'Invoice_Total': 50000.00,
            'Invoice_Amount': -2000.00,  # Credit invoice
            'Allocated_Amount': -2000.00,
            'Remaining_Invoice': 52000.00,
            'Pre_Allocation_Funding': 50000.00,
            'Split_Record': 'No',
            'Is_Credit': 'Yes',  # Credit record
            'Is_Coupled': 'Yes',
            'Labor_Type': 'Direct Labor',
            'Invoice_Label': 'Direct Labor',  # PRESERVED ORIGINAL LABEL
            'Invoice_ID_Debug': 5,
            'Unique_Count': 3,
            'Actual_Invoice_ID': 'INV-0006185918',
            'Vendor_Name': 'SLAFF, ANDREW',
            'Emp_ID': 'HB4202',
            'PLC': 'PML1',
            'PLC_Description': 'Program Mission Lead',
            'Account_ID': '5017-01',
            'Org_ID': '1.25.33795.K6',
            'Billable_Regular_Hours': -20.0,
            'Allocated_Billable_Hours': -20.0,
            'Total_Burdens': -800.00,
            'Transaction_Amount': -1200.00,
            'Invoice_Date': '2025-05-09',
            'Timesheet_End_Date': '2025-04-30',
            'Comments': 'Time correction',
            'Has_MPO': 'Yes',
            'Absolute_Sequence': 5,
            'Allocation_Sequence': '2001-AC-001',
            'Invoice_Seq': 1
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_records)
    
    # Save to output directory
    output_dir = project_root / "data" / "processed" / "focusedfox" / "2025" / "04"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"focusedfox_sample_output_{timestamp}.csv"
    
    df.to_csv(output_file, index=False)
    
    print(f"✅ Sample output created: {output_file}")
    print(f"📊 Contains {len(df)} allocation records")
    
    # Display summary
    print(f"\n📈 SAMPLE OUTPUT SUMMARY")
    print(f"=" * 40)
    print(f"Total Allocated Amount: ${df['Allocated_Amount'].sum():,.2f}")
    print(f"Total Allocated Hours: {df['Allocated_Billable_Hours'].sum():,.1f}")
    print(f"Unique CLINs: {df['CLIN'].nunique()}")
    print(f"Unique SLINs: {df['SLIN'].nunique()}")
    print(f"Split Records: {(df['Split_Record'] == 'Yes').sum()}")
    print(f"Credit Records: {(df['Is_Credit'] == 'Yes').sum()}")
    print(f"Coupled Records: {(df['Is_Coupled'] == 'Yes').sum()}")
    
    # Display records
    print(f"\n📋 ALLOCATION RECORDS")
    print(f"=" * 60)
    
    for i, row in df.iterrows():
        status_flags = []
        if row['Split_Record'] == 'Yes':
            status_flags.append('SPLIT')
        if row['Is_Credit'] == 'Yes':
            status_flags.append('CREDIT')
        if row['Is_Coupled'] == 'Yes':
            status_flags.append('COUPLED')
        
        flags = ' [' + ', '.join(status_flags) + ']' if status_flags else ''
        
        print(f"{i+1}. CLIN {row['CLIN']}-{row['SLIN']}: {row['Invoice_Label'][:25]:25} = ${row['Allocated_Amount']:8,.2f}{flags}")
        print(f"   👤 {row['Vendor_Name'][:30]:30} 📅 {row['Invoice_Date']} 🔢 {row['Allocation_Sequence']}")
    
    # Display all columns
    print(f"\n📝 COMPLETE OUTPUT STRUCTURE ({len(df.columns)} columns)")
    print(f"=" * 60)
    
    for i, col in enumerate(df.columns, 1):
        sample_val = str(df[col].iloc[0])
        if len(sample_val) > 35:
            sample_val = sample_val[:32] + "..."
        print(f"{i:2d}. {col:28} │ {sample_val}")
    
    print(f"\n💡 KEY FEATURES DEMONSTRATED:")
    print(f"   • Original InvDescription preserved in 'Invoice_Label'")
    print(f"   • Chronological processing with 'Absolute_Sequence'")
    print(f"   • Labor coupling (Direct Labor + Overtime same SLIN)")
    print(f"   • Split records when funding insufficient") 
    print(f"   • Credit/negative invoice handling")
    print(f"   • Detailed tracking with CLIN-SLIN-sequence format")
    
    return output_file


def main():
    """Main runner."""
    print("🎯 INVOICE ALLOCATION SYSTEM - EXPECTED OUTPUT")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(project_root)
    
    # Create sample output
    output_file = create_sample_allocation_output()
    
    print(f"\n🎉 Sample output generation complete!")
    print(f"📁 File: {output_file}")
    print(f"\nThis shows the exact structure your system will produce when")
    print(f"processing real MPO and Invoice data with meaningful allocations.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())