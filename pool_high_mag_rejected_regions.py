import os
import pandas as pd
from tqdm import tqdm

result_dir = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir"

pipeline_run_history_path = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir/pipeline_run_history.csv"

notes_to_keep = ["First all-slide BMA-diff and PBS-diff processing with specimen classification. Begin on 2024-09-16.", "Running BMA-diff Pipeline on H-odd-year slides reported as BMA in part description."]

pipeline_run_history = pd.read_csv(pipeline_run_history_path)

# get the rows with the notes in notes_to_keep
rows = pipeline_run_history[pipeline_run_history["note"].isin(notes_to_keep)]

# only keeps rows where the pipeline is BMA-diff
rows = rows[rows["pipeline"] == "BMA-diff"]

print(f"Found {len(rows)} rows with the notes in notes_to_keep.")

cell_regions_df_dict = {
    "wsi_name": [],
    "pipeline": [],
    "datetime_processed": [],
    "note": [],
    "region_idx": [],
    "region_file_path": [],
    "low_mag_score": [],
    "high_mag_score": [],
    "pseudo_idx": [],
    "low_mag_VoL": [],
    "high_mag_VoL": [],
    "coordinate": []
}

pseudo_idx = 0

for idx, row in tqdm(rows.iterrows(), total=len(rows), desc="Processing Rows"):

    # get the path to the cell regions which are all the .jpg files in the result_dir/pipeline_datetime_processed/focus_regions/high_mag_rejected folder
    high_mag_rejected_dir = os.path.join(result_dir, row["pipeline"] + "_" + row["datetime_processed"], "focus_regions", "high_mag_rejected")

    if not os.path.exists(high_mag_rejected_dir):
        print(f"Folder {high_mag_rejected_dir} does not exist. Skipping...")
        continue

    # get all the file names in the high_mag_rejected_dir
    high_mag_rejected_files = os.listdir(high_mag_rejected_dir)

    low_mag_csv_path = os.path.join(result_dir, row["pipeline"] + "_" + row["datetime_processed"], "focus_regions", "focus_regions_info.csv")
    high_mag_csv_path = os.path.join(result_dir, row["pipeline"]  + "_" + row["datetime_processed"], "focus_regions", "high_mag_focus_regions_info.csv")

    low_mag_df = pd.read_csv(low_mag_csv_path)
    high_mag_df = pd.read_csv(high_mag_csv_path)

    for region_file_name in high_mag_rejected_files:

        # remove the .jpg extension and convert to int to get the region_idx
        region_idx = int(region_file_name[:-4])
        region_idx = region_idx

        # get the path to the region file
        region_file_path = os.path.join(high_mag_rejected_dir, region_file_name)

        # low_mag_score is the row where the column idx is low_mag_idx matches with region_idx, and the value of adequate_confidence_score
        low_mag_score = low_mag_df[low_mag_df["idx"] == region_idx]["adequate_confidence_score"].values[0]

        # high_mag_score is the row where the column idx is high_mag_idx matches with region_idx, and the value of adequate_confidence_score_high_mag
        high_mag_score = high_mag_df[high_mag_df["idx"] == region_idx]["adequate_confidence_score_high_mag"].values[0]

        # get the VoL from the row where the column idx is low_mag_df matches with region_idx
        low_mag_VoL = low_mag_df[low_mag_df["idx"] == region_idx]["VoL"].values[0]

        # get the VoL from the row where the column idx is high_mag_df matches with region_idx
        high_mag_VoL = high_mag_df[high_mag_df["idx"] == region_idx]["VoL_high_mag"].values[0]

        # get the coordinate from the row where the column idx is low_mag_df matches with region_idx
        coordinate = low_mag_df[low_mag_df["idx"] == region_idx]["coordinate"].values[0]

        cell_regions_df_dict["wsi_name"].append(row["wsi_name"])
        cell_regions_df_dict["pipeline"].append(row["pipeline"])
        cell_regions_df_dict["datetime_processed"].append(row["datetime_processed"])
        cell_regions_df_dict["note"].append(row["note"])
        cell_regions_df_dict["region_idx"].append(region_idx)
        cell_regions_df_dict["region_file_path"].append(region_file_path)
        cell_regions_df_dict["low_mag_score"].append(low_mag_score)
        cell_regions_df_dict["high_mag_score"].append(high_mag_score)
        cell_regions_df_dict["pseudo_idx"].append(pseudo_idx)
        cell_regions_df_dict["low_mag_VoL"].append(low_mag_VoL)
        cell_regions_df_dict["high_mag_VoL"].append(high_mag_VoL)
        cell_regions_df_dict["coordinate"].append(coordinate)

        pseudo_idx += 1

cell_regions_df = pd.DataFrame(cell_regions_df_dict)
cell_regions_df.to_csv("/media/hdd3/neo/bma_high_mag_rejected_regions/regions_metadata.csv", index=False)

# print the total number of regions found
print(f"Found {pseudo_idx} regions.")