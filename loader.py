# csv_loader.py

import os
import pandas as pd
from db_config import get_engine

# Path where your CSVs are stored
CSV_FOLDER = "csvs"

def load_csvs_to_db():
    engine = get_engine()
    csv_files = [f for f in os.listdir(CSV_FOLDER) if f.endswith(".csv")]

    all_data = []

    for file in csv_files:
        path = os.path.join(CSV_FOLDER, file)
        print(f"⏳ Loading {file}...")

        df = pd.read_csv(path)

        # Normalize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.title()

        # Parse date columns if present
        if 'Shift_Date' in df.columns:
            df['Shift_Date'] = pd.to_datetime(df['Shift_Date'], dayfirst=True)

        # Append to all_data
        all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_sql("production_data", engine, if_exists='replace', index=False)
        print(f"🎉 All {len(csv_files)} files loaded into 'production_data' table ({len(final_df)} rows)")
    else:
        print("⚠️ No CSV files found in the folder.")

if __name__ == "__main__":
    load_csvs_to_db()
