"""
Data processing script for RTE production and consumption data.
Extracts dates from the file structure and creates processed parquet files.
"""

import pandas as pd
import re
from pathlib import Path
from datetime import datetime


def process_rte_production_file(input_file, output_file):
    """
    Process RTE production Excel file to extract dates and create a structured dataframe.
    
    Args:
        input_file: Path to the input .xls file
        output_file: Path to the output .parquet file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Reading file: {input_path}")
    
    # Read the file as text (it's actually a tab-separated text file)
    # Try different encodings
    content = None
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            with open(input_path, 'r', encoding=encoding) as f:
                content = f.readlines()
            print(f"Successfully read file with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        raise ValueError("Could not read file with any encoding")
    
    # Pattern to match date lines: "Données de réalisation du DD/MM/YYYY"
    # Handle encoding issues - look for the date pattern directly
    # The date format is always DD/MM/YYYY, so we can search for that pattern
    # and check if the line contains "Donn" and "réalisation" or similar
    date_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})')
    
    # Storage for processed data
    all_rows = []
    current_date = None
    header_columns = None
    
    i = 0
    while i < len(content):
        line = content[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check if this is a date line
        # Look for date pattern and check if line contains "Donn" and "réalisation" keywords
        date_match = date_pattern.search(line)
        if date_match and 'Donn' in line and ('réalisation' in line.lower() or 'realisation' in line.lower()):
            # Extract date
            date_str = date_match.group(1)
            try:
                current_date = datetime.strptime(date_str, '%d/%m/%Y').date()
                print(f"Found date: {current_date}")
            except ValueError:
                print(f"Warning: Could not parse date: {date_str}")
                i += 1
                continue
            
            # Next line should be the header
            i += 1
            if i < len(content):
                header_line = content[i].strip()
                if header_line:
                    # Parse header (tab-separated)
                    header_columns = [col.strip() for col in header_line.split('\t') if col.strip()]
                    # Remove empty strings and clean column names
                    header_columns = [col for col in header_columns if col]
                    print(f"Found header with {len(header_columns)} columns: {header_columns[:5]}...")
                    i += 1
            continue
        
        # If we have a date and header, process data rows
        if current_date is not None and header_columns is not None:
            # Check if this looks like a data row (starts with time pattern like "00:00-01:00")
            if re.match(r'\d{2}:\d{2}-\d{2}:\d{2}', line):
                # Parse tab-separated values
                values = [val.strip() for val in line.split('\t')]
                
                # Create a row dictionary
                row_dict = {'date': current_date}
                
                # Map values to header columns
                for idx, col in enumerate(header_columns):
                    if idx < len(values):
                        val = values[idx]
                        # Handle missing values: '*', '-', empty strings
                        if val == '' or val == '-' or val == '*':
                            row_dict[col] = None
                        else:
                            # Keep as string for now, we'll convert to numeric later
                            row_dict[col] = val
                    else:
                        row_dict[col] = None
                
                all_rows.append(row_dict)
        
        i += 1
    
    # Create DataFrame
    if not all_rows:
        raise ValueError("No data rows found in the file")
    
    print(f"Processed {len(all_rows)} rows")
    df = pd.DataFrame(all_rows)
    
    # Ensure date column is datetime type
    df['date'] = pd.to_datetime(df['date'])
    
    # Convert numeric columns (all except 'date' and 'Heures')
    numeric_columns = [col for col in df.columns if col not in ['date', 'Heures']]
    for col in numeric_columns:
        # Convert to numeric, coercing errors (like '*', '-', etc.) to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Save to parquet
    print(f"\nSaving to: {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"✅ Successfully saved processed data to {output_path}")
    
    return df


def process_rte_consumption_file(input_file, output_file):
    """
    Process RTE consumption Excel file to extract dates and create a structured dataframe.
    
    Args:
        input_file: Path to the input .xls file
        output_file: Path to the output .parquet file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Reading file: {input_path}")
    
    # Read the file as text (it's actually a tab-separated text file)
    # Try different encodings
    content = None
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            with open(input_path, 'r', encoding=encoding) as f:
                content = f.readlines()
            print(f"Successfully read file with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        raise ValueError("Could not read file with any encoding")
    
    # Pattern to match date lines: "Journée du DD/MM/YYYY"
    date_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})')
    
    # Storage for processed data
    all_rows = []
    current_date = None
    header_columns = None
    
    i = 0
    while i < len(content):
        line = content[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check if this is a date line
        # Look for date pattern and check if line contains "Journ" keyword
        date_match = date_pattern.search(line)
        if date_match and 'Journ' in line:
            # Extract date
            date_str = date_match.group(1)
            try:
                current_date = datetime.strptime(date_str, '%d/%m/%Y').date()
                print(f"Found date: {current_date}")
            except ValueError:
                print(f"Warning: Could not parse date: {date_str}")
                i += 1
                continue
            
            # Next line should be the header
            i += 1
            if i < len(content):
                header_line = content[i].strip()
                if header_line:
                    # Parse header (tab-separated)
                    header_columns = [col.strip() for col in header_line.split('\t') if col.strip()]
                    # Remove empty strings and clean column names
                    header_columns = [col for col in header_columns if col]
                    print(f"Found header with {len(header_columns)} columns: {header_columns[:5]}...")
                    i += 1
            continue
        
        # If we have a date and header, process data rows
        if current_date is not None and header_columns is not None:
            # Check if this looks like a data row (starts with time pattern like "00:00" or "00:15")
            if re.match(r'\d{2}:\d{2}', line):
                # Parse tab-separated values
                values = [val.strip() for val in line.split('\t')]
                
                # Create a row dictionary
                row_dict = {'date': current_date}
                
                # Map values to header columns
                for idx, col in enumerate(header_columns):
                    if idx < len(values):
                        val = values[idx]
                        # Handle missing values: '*', '-', empty strings
                        if val == '' or val == '-' or val == '*':
                            row_dict[col] = None
                        else:
                            # Keep as string for now, we'll convert to numeric later
                            row_dict[col] = val
                    else:
                        row_dict[col] = None
                
                all_rows.append(row_dict)
        
        i += 1
    
    # Create DataFrame
    if not all_rows:
        raise ValueError("No data rows found in the file")
    
    print(f"Processed {len(all_rows)} rows")
    df = pd.DataFrame(all_rows)
    
    # Ensure date column is datetime type
    df['date'] = pd.to_datetime(df['date'])
    
    # Convert numeric columns (all except 'date' and 'Heures')
    numeric_columns = [col for col in df.columns if col not in ['date', 'Heures']]
    for col in numeric_columns:
        # Convert to numeric, coercing errors (like '*', '-', etc.) to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Save to parquet
    print(f"\nSaving to: {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"✅ Successfully saved processed data to {output_path}")
    
    return df


def main():
    """Main function to process RTE production and consumption data."""
    # Define paths
    data_dir = Path(__file__).parent / "data" / "rte"
    
    # Process production file
    print("=" * 60)
    print("Processing RTE Production Data")
    print("=" * 60)
    production_input = data_dir / "RealisationDonneesProduction_2023.xls"
    production_output = data_dir / "RealisationDonneesProduction_2023_processed.parquet"
    
    try:
        df_prod = process_rte_production_file(production_input, production_output)
        print(f"\n✅ Production processing complete!")
        print(f"   Input: {production_input}")
        print(f"   Output: {production_output}")
        print(f"   Rows: {len(df_prod)}")
        print(f"   Columns: {len(df_prod.columns)}")
    except Exception as e:
        print(f"❌ Error processing production file: {e}")
        raise
    
    # Process consumption file
    print("\n" + "=" * 60)
    print("Processing RTE Consumption Data")
    print("=" * 60)
    consumption_input = data_dir / "conso_mix_RTE_2023.xls"
    consumption_output = data_dir / "conso_mix_RTE_2023_processed.parquet"
    
    try:
        df_cons = process_rte_consumption_file(consumption_input, consumption_output)
        print(f"\n✅ Consumption processing complete!")
        print(f"   Input: {consumption_input}")
        print(f"   Output: {consumption_output}")
        print(f"   Rows: {len(df_cons)}")
        print(f"   Columns: {len(df_cons.columns)}")
    except Exception as e:
        print(f"❌ Error processing consumption file: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("✅ All processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

