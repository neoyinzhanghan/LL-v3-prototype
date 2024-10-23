import os
import subprocess
import pandas as pd
from tqdm import tqdm
from LLRunner.slide_processing.dzsave import dzsave


topview_dir = "selected_topviews"
pipeline_run_history_path = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir/pipeline_run_history.csv"

slide_source_dir = "/pesgisipth/NDPI"
save_dir_ndpi = "/media/hdd3/neo/error_slides_ndpi"
save_dir_dzsave = "/media/hdd3/neo/error_slides_dzsave"

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

for wsi_name in tqdm(wsi_names, desc="Saving WSI Tiles"):

    # remove the .ndpi extension
    wsi_name_no_ext = wsi_name[:-5]

    wsi_name_source_path = os.path.join(slide_source_dir, wsi_name)

    # Run rsync using subprocess, which waits for it to finish
    subprocess.run(["rsync", "-av", wsi_name_source_path, save_dir_ndpi], check=True)

    new_wsi_path = os.path.join(save_dir_ndpi, wsi_name)

    dzsave(wsi_path = new_wsi_path, 
           save_dir = save_dir_dzsave, 
           folder_name = wsi_name_no_ext,
           tile_size = 512, 
           num_cpus = 32,
           region_cropping_batch_size =256)