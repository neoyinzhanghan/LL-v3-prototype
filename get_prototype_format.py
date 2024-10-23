import os
import pandas as pd
from tqdm import tqdm


topview_dir = "selected_topviews"
pipeline_run_history_path = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir/pipeline_run_history.csv"

pipeline_run_history = pd.read_csv(pipeline_run_history_path)

# get the path to all the jpg file names in selected_topviews
all_names = os.listdir(topview_dir)

# only keeps jpg files
all_names = [name for name in all_names if name.endswith(".jpg")]


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

    wsi_names_df["pipeline"].append(pipeline)
    wsi_names_df["datetime_processed"].append(datetime_processed)
    wsi_names_df["wsi_name"].append(wsi_name)

wsi_names_df = pd.DataFrame(wsi_names_df)
wsi_names_df.to_csv("selected_topviews/wsi_names.csv", index=False)

    