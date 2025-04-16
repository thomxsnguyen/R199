import pandas as pd
import os
import glob

def resample_csv(file_path, new_interval="30S"):
    df = pd.read_csv(file_path)
    df["Time (s)"] = pd.to_timedelta(df["Time (s)"], unit='s')
    df.set_index("Time (s)", inplace=True)
    df_resampled = df.resample(new_interval).interpolate(method="linear")
    df_resampled.reset_index(inplace=True)
    df_resampled["Time (s)"] = df_resampled["Time (s)"].dt.total_seconds()
    return df_resampled

def resample_all_in_folder(source_folder, target_folder, new_interval="30S"):
    os.makedirs(target_folder, exist_ok=True)
    file_paths = glob.glob(os.path.join(source_folder, "*.csv"))

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        resampled = resample_csv(file_path, new_interval)
        new_path = os.path.join(target_folder, filename.replace(".csv", "_interp.csv"))
        resampled.to_csv(new_path, index=False)
        print(f"Resampled and saved: {new_path}")

# Run this section to resample your datasets
if __name__ == "__main__":
    resample_all_in_folder("models/data/csvs_interpolated/train", "interp_timesteps/train_interp", new_interval="30S")
    resample_all_in_folder("models/data/csvs_interpolated/validation", "interp_timesteps/validation_interp", new_interval="30S")
    resample_all_in_folder("models/data/csvs_interpolated/testing", "interp_timesteps/testing_interp", new_interval="30S")
