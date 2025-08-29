# Invoice Allocation System

A GitLab-based automated system for processing MPO (Monthly Programming Order) and Invoice data, implementing chronological allocation with labor coupling logic.

## 🏗️ Architecture

```
MPO Excel Upload → Invoice Excel Upload → GitLab Repository
                                             ↓
                                    Python ETL Pipeline
                                             ↓
                                   Python Allocation Engine
                                             ↓
                                      CSV Output File
                                             ↓
                                   GitLab Output Folder
                                             ↓
                                 Automated Reports/Archives
```

## 📁 Project Structure

```
invoice-allocation-system/
├── 📂 data/                     # Data organized by program/year/month
│   ├── raw/
│   │   ├── mpo/[program]/[year]/[month]/           # MPO Excel files
│   │   └── invoices/[program]/[year]/[month]/      # Invoice CSV/Excel files
│   └── processed/[program]/[year]/[month]/         # Output CSV files
├── 📂 src/                      # Modular Python codebase
│   ├── etl/                     # ETL Pipeline
│   │   ├── mpo_processor.py     # MPO data processing (Power Query logic)
│   │   ├── invoice_processor.py  # Invoice data processing (Power Query logic)
│   │   └── pipeline.py          # ETL orchestration
│   ├── allocation/              # Allocation Engine
│   │   ├── allocation_engine.py # Core allocation logic
│   │   └── coupling_logic.py    # Labor coupling rules
│   └── utils/                   # Utilities
│       ├── file_handlers.py     # File I/O operations
│       └── validators.py        # Data validation
├── 📂 config/                   # Configuration files
│   └── focusedfox.yaml         # Consolidated configuration
├── 📂 scripts/                  # Execution scripts
│   ├── run_allocation.py       # Main processing script
│   ├── validate_inputs.py      # Input validation
│   └── show_expected_output.py # Sample output structure
├── 📂 tests/                    # Test suite
└── 📋 .gitlab-ci.yml            # GitLab CI/CD pipeline
```

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# View expected output structure
python scripts/show_expected_output.py
```

### 2. Run Allocation
```bash
# Process specific program/year/month
python scripts/run_allocation.py --program focusedfox --year 2025 --month 4

# Validate inputs only
python scripts/run_allocation.py --program focusedfox --year 2025 --month 4 --validate-only
```

### 3. GitLab CI/CD
The system automatically processes files when uploaded to GitLab:
- **Validation Stage**: Validates uploaded files
- **ETL Stage**: Processes MPO and Invoice data
- **Allocation Stage**: Runs allocation engine
- **Report Stage**: Generates summary reports
- **Archive Stage**: Archives results

## ⚙️ Configuration

### Configuration (`config/focusedfox.yaml`)
```yaml
program_name: "FOCUSEDFOX"
excluded_clins: [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,18,19]
fee_descriptions: ["FEE", "Fee"]
output_format: "csv"
decimal_precision: 2

# Column mappings for this program
column_mappings:
  mpo:
    clin: "CLIN"
    slin: "SLIN"
    funding: "Accum Total"
    start_date: "InvoiceStartDate"
  invoice:
    labor_type: "BILL_FM_LN_LBL"
    employee_id: "Empl/Vendor ID"
    amount: "Billed Amt + Burdens"
```

The configuration file contains all settings including:
- Program-specific settings (excluded CLINs, fee descriptions)
- Column mappings for MPO and Invoice files
- Allocation rules (labor coupling, priorities)
- Validation settings
- Output formatting
- Logging configuration

## 📊 Data Processing Logic

### 1. MPO Processing (Power Query Logic Replication)
- Transform CLIN to text
- Split Description column on '-' delimiter
- Filter out fee descriptions and excluded CLINs
- Sort by CLIN and SLIN
- Add MPO index and month-year columns

### 2. Invoice Processing (Power Query Logic Replication)
- Apply column mappings
- Filter out null descriptions and fee entries
- Add sequential invoice IDs
- Convert dates and numeric columns
- Add month-year for joining

### 3. Allocation Engine
- **Chronological Processing**: Process invoices by date order
- **Labor Coupling**: Direct Labor and DL Overtime for same employee/timesheet/CLIN use same SLIN
- **Funding Tracking**: Track available funding per CLIN/SLIN
- **Split Records**: Handle partial allocations when funding is insufficient

## 🔍 Key Features

### Labor Coupling Logic
- Direct Labor and DL Overtime entries for the same employee/timesheet/CLIN are allocated to the same SLIN
- Preserves relationship between regular time and overtime

### Original Data Preservation
- **InvDescription** labels preserved exactly as uploaded
- No modification of source data during processing
- Debug columns track transformations

### Chronological Allocation
- Invoices processed in strict date order
- Funding allocated on first-come, first-served basis
- Earlier invoices have priority over later ones

### Data Validation
- Input file validation before processing
- Column presence and type checking
- Business rule validation (excluded CLINs, fee descriptions)

## 📈 Output Format

The system generates CSV files with the following structure:
```csv
Project_Code,CLIN,SLIN,Invoice_Amount,Allocated_Amount,Remaining_Invoice,
Split_Record,Is_Credit,Invoice_Label,Vendor_Name,Emp_ID,...
```

### Key Output Columns:
- **Invoice_Label**: Original InvDescription (preserved exactly)
- **Allocated_Amount**: Amount allocated to this CLIN/SLIN
- **Split_Record**: "Yes" if invoice was split across multiple CLINs
- **Is_Credit**: "Yes" for credit/negative invoices
- **Allocation_Sequence**: Unique sequence for tracking

## 🛠️ Development

### Adding New Programs
1. Create configuration file: `config/[program].yaml`
2. Copy structure from focusedfox.yaml as template
3. Update program-specific rules and mappings

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Show expected output structure
python scripts/show_expected_output.py
```

### GitLab Variables
Set these in GitLab CI/CD variables:
- `PROGRAM`: Program name (e.g., "focusedfox")
- `YEAR`: Processing year
- `MONTH`: Processing month

## 📋 File Specifications

### MPO Sheet (Monthly Funding Data)
**Location**: `data/raw/mpo/[program]/[year]/[month]/`
**Required Columns**:
- CLIN, SLIN, Description, Accum Total, InvoiceStartDate
- Prior ITD Bill, Current ITD Bill, Funding, Remaining Funding

### Invoice Details Sheet (Monthly Billing Data)
**Location**: `data/raw/invoices/[program]/[year]/[month]/`
**Required Columns**:
- Project String, Invoice ID, Invoice Date, BILL_FM_LN_LBL
- Empl/Vendor ID, Billed Amt + Burdens, Timesheet End Date
- Billable Regular Hours, Transaction Amount, Total Burdens

## 🔧 Troubleshooting

### Common Issues
1. **Module not found**: Run `pip install -r requirements.txt`
2. **File not found**: Check data is in correct directory structure
3. **Column missing**: Verify file format matches expected columns
4. **Allocation errors**: Check MPO funding availability

### Debug Mode
```bash
# Enable detailed logging
python scripts/run_allocation.py --program focusedfox --year 2025 --month 4 --debug
```

## 📞 Support

For issues and questions:
- Check the GitLab repository issues
- Review configuration files for program-specific settings
- Validate input data format matches specifications

---

**System Version**: 1.0.0  
**Last Updated**: 2025-01-27