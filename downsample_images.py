import os
import subprocess
import pandas as pd
from tqdm import tqdm
from image_downsampler import downsample_slide_dzsave_dir


topview_dir = "selected_topviews"
pipeline_run_history_path = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir/pipeline_run_history.csv"

slide_source_dir = "/pesgisipth/NDPI"
save_dir_ndpi = "/media/hdd3/neo/error_slides_ndpi"
save_dir_dzsave = "/media/hdd3/neo/error_slides_dzsave"

already_downsampled_df_path = "/media/hdd3/neo/error_slides_dzsave/already_downsampled.csv"

pipeline_run_history = pd.read_csv(pipeline_run_history_path)

# get the path to all the jpg file names in selected_topviews
all_names = os.listdir(topview_dir)

# only keeps jpg files
all_names = [name for name in all_names if name.endswith(".jpg")]

wsi_names = []
wsi_names_df = {
    "pipeline": [],
    "datetime_processed": [],
    "wsi_name": []
}

def did_we_downsample(dzsave_dir):
    """
    This function checks if the dzsave_dir has been downsampled.
    """

    # check if the dzsave_dir exists
    if not os.path.exists(dzsave_dir):
        return False
    
    # check if the subdir 18_downsampled exists
    if not os.path.exists(os.path.join(dzsave_dir, "18_downsampled")):
        return False
    
    # check to see if every .jpeg file in 18 has a corresponding .jpeg file in 18_downsampled with the same file name
    for image_name in tqdm(os.listdir(os.path.join(dzsave_dir, "18")), desc="Checking if Folder is Fully Processed"):
        if not os.path.exists(os.path.join(dzsave_dir, "18_downsampled", image_name)):
            return False
        
    return True

for name in tqdm(all_names, desc="Getting WSI names"):
    pipeline = name.split("_")[0]
    datetime_processed = name.split("_")[1].split(".")[0]

    # look for the row with pipeline and datetime_processed
    row = pipeline_run_history[(pipeline_run_history["pipeline"] == pipeline) & (pipeline_run_history["datetime_processed"] == datetime_processed)]

    # assert that exactly one row is found
    assert len(row) == 1, f"Expected 1 row, got {len(row)} rows for {name}"

    wsi_name = row["wsi_name"].values[0]

    wsi_names.append(wsi_name)

    wsi_names_df["pipeline"].append(pipeline)
    wsi_names_df["datetime_processed"].append(datetime_processed)
    wsi_names_df["wsi_name"].append(wsi_name)

wsi_names_df = pd.DataFrame(wsi_names_df)
wsi_names_df.to_csv("selected_topviews/wsi_names.csv", index=False)

presaved_df = pd.read_csv(already_downsampled_df_path)

# get the column wsi_name from presaved_df as a list
presaved_wsi_names = presaved_df["wsi_name"].tolist()

wsi_names_to_process = [wsi_name for wsi_name in wsi_names if wsi_name not in presaved_wsi_names]
wsi_names_already_downsampled = [wsi_name for wsi_name in wsi_names if wsi_name in presaved_wsi_names]

print(f"Founds {len(wsi_names)} WSI tiles.")
print(f"Found {len(presaved_wsi_names)} WSI tiles already downsampled.")
print(f"Processing {len(wsi_names_to_process)} WSI tiles...")

already_downsampled_df = {
    "wsi_name": [],
    "dzsave_dir": []
}

for wsi_name in wsi_names_already_downsampled:
    already_downsampled_df["wsi_name"].append(wsi_name)
    already_downsampled_df["dzsave_dir"].append(os.path.join(save_dir_dzsave, wsi_name[:-5]))

for wsi_name in tqdm(wsi_names, desc="Saving WSI Tiles"):
    # remove the .ndpi extension
    wsi_name_no_ext = wsi_name[:-5]

    # create the dzsave directory
    dzsave_dir = os.path.join(save_dir_dzsave, wsi_name_no_ext)

    # check to see if the wsi_name_no_ext folder already exists in save_dir_dzsave
    already_done = did_we_downsample(dzsave_dir)

    if already_done:
        print(f"Folder {wsi_name_no_ext} already fully processed. Skipping...")
        already_downsampled_df["wsi_name"].append(wsi_name)
        already_downsampled_df["dzsave_dir"].append(dzsave_dir)
        already_downsampled_df_to_save = pd.DataFrame(already_downsampled_df)
        already_downsampled_df_to_save.to_csv(already_downsampled_df_path, index=False)
        continue

    # downsample the images
    downsample_slide_dzsave_dir(dzsave_dir)
    already_downsampled_df["wsi_name"].append(wsi_name)
    already_downsampled_df["dzsave_dir"].append(dzsave_dir)

    already_downsampled_df_to_save = pd.DataFrame(already_downsampled_df)
    already_downsampled_df_to_save.to_csv(already_downsampled_df_path, index=False)