import pandas as pd
import os

# Directory containing the files
directory = '/Users/LijieZhou/Development/data-analysis'

# List all .xlsx files
excel_files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]

for excel_file in excel_files:
    # Full path to the Excel file
    excel_path = os.path.join(directory, excel_file)
    
    # Read the Excel file (first sheet by default)
    df = pd.read_excel(excel_path, engine='openpyxl')
    
    # Create CSV filename
    csv_file = excel_file.replace('.xlsx', '.csv')
    csv_path = os.path.join(directory, csv_file)
    
    # Save to CSV
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"Converted {excel_file} to {csv_file}")