import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import glob
import argparse


def interpolate_csv(input_file, output_file):
    """
    Interpolate a CSV file to have values at every 70-second interval (up to 14,400 seconds).

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path where the interpolated CSV will be saved.
    """
    df = pd.read_csv(input_file)

    required_columns = ['Time (s)', 'T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input (C)']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {input_file}")

    df = df.sort_values('Time (s)')
    original_time = df['Time (s)'].values

    # Set up new time points every 70 seconds up to the maximum time or 14,400 (whichever is smaller)
    max_time = min(original_time[-1], 14400)
    new_time = np.arange(0, max_time + 1, 70)

    # Create interpolation functions
    interp_funcs = {
        col: interp1d(original_time, df[col].values, kind='linear', bounds_error=False, fill_value="extrapolate")
        for col in required_columns if col != 'Time (s)'
    }

    # Generate interpolated values
    new_data = {
        'Time (s)': new_time
    }
    for col, func in interp_funcs.items():
        new_data[col] = func(new_time)

    new_df = pd.DataFrame(new_data)
    new_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate thermal time series CSV files to 70-second intervals.")
    parser.add_argument(
        "--input", "-i",
        default="csvs_unprocessed",
        help="Input directory containing CSV files (default: csvs_unprocessed)"
    )
    parser.add_argument(
        "--output", "-o",
        default="interpolated_csvs(70s)",
        help="Output directory for interpolated CSV files (default: interpolated_csvs(70s))"
    )

    args = parser.parse_args()
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(input_dir, "*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        print(f"Looking for pattern: {pattern}")
    else:
        print(f"Found {len(csv_files)} CSV files")
        for csv_file in csv_files:
            file_name = os.path.basename(csv_file)
            output_file = os.path.join(output_dir, file_name)
            print(f"Processing {csv_file} -> {output_file}")
            try:
                interpolate_csv(csv_file, output_file)
                print(f"  ✅ Successfully interpolated to {output_file}")
            except Exception as e:
                print(f"  ❌ Error processing {csv_file}: {str(e)}")
