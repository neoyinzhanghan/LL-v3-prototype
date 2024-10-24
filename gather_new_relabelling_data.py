import os
import random
import openslide
import pandas as pd
from PIL import Image
from tqdm import tqdm

result_dir = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir"
save_dir = "/media/hdd3/neo/regions_for_greg_to_label"
pipeline_run_history_path = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir/pipeline_run_history.csv"

pipeline_run_history = pd.read_csv(pipeline_run_history_path)

# create the save_dir if it doesn't exist
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "high_mag_selected"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "high_mag_rejected"), exist_ok=True)

# find all the subdirectories in the result_dir that starts with BMA-diff
subdirs = [d for d in os.listdir(result_dir) if d.startswith("BMA-diff")]

# only keep the subdirs that contain further subdirectories focus_regions/high_mag_rejected, and focus_regions/high_mag_unannotated
subdirs = [d for d in subdirs if os.path.exists(os.path.join(result_dir, d, "focus_regions", "high_mag_rejected")) and os.path.exists(os.path.join(result_dir, d, "focus_regions", "high_mag_unannotated"))]

total_num_cells = 3333

cell_metadata = {
    "pseudo_idx": [],
    "data_source_type": [],
    "wsi_name": [],
    "pipeline": [],
    "datetime_processed": [],
    "region_idx": [],
    "region_file_path": [],
    "low_mag_score": [],
    "high_mag_score": [],
    "low_mag_VoL": [],
    "high_mag_VoL": [],
    "coordinate": []
}

pseudo_idx = 0

for i in tqdm(range(total_num_cells), desc="Sampling Cells from BMA result high mag rejected folders"):
    # randomly select a subdir in subdirs
    subdir = random.choice(subdirs)

    data_source_type = "high_mag_rejected"

    # get the list of .jpg images in the subdir/focus_regions/high_mag_rejected folder
    high_mag_rejected_files = os.listdir(os.path.join(result_dir, subdir, "focus_regions", "high_mag_rejected"))
    # make sure the list only contains .jpg files
    high_mag_rejected_files = [f for f in high_mag_rejected_files if f.endswith(".jpg")]
    
    while len(high_mag_rejected_files) == 0:
        subdir = random.choice(subdirs)
        high_mag_rejected_files = os.listdir(os.path.join(result_dir, subdir, "focus_regions", "high_mag_rejected"))
        high_mag_rejected_files = [f for f in high_mag_rejected_files if f.endswith(".jpg")]

    # randomly select a high_mag_rejected_file
    high_mag_rejected_file = random.choice(high_mag_rejected_files)

    # get the region_idx from the file name
    region_idx = int(high_mag_rejected_file[:-4])

    # get the path to the high_mag_rejected_file
    region_file_path = os.path.join(result_dir, subdir, "focus_regions", "high_mag_rejected", high_mag_rejected_file)

    # get the low_mag_score, high_mag_score, low_mag_VoL, high_mag_VoL, and coordinate
    low_mag_csv_path = os.path.join(result_dir, subdir, "focus_regions", "focus_regions_info.csv")
    high_mag_csv_path = os.path.join(result_dir, subdir, "focus_regions", "high_mag_focus_regions_info.csv")    

    low_mag_df = pd.read_csv(low_mag_csv_path)
    high_mag_df = pd.read_csv(high_mag_csv_path)

    low_mag_score = low_mag_df[low_mag_df["idx"] == region_idx]["adequate_confidence_score"].values[0]
    high_mag_score = high_mag_df[high_mag_df["idx"] == region_idx]["adequate_confidence_score_high_mag"].values[0]

    low_mag_VoL = low_mag_df[low_mag_df["idx"] == region_idx]["VoL"].values[0]
    high_mag_VoL = high_mag_df[high_mag_df["idx"] == region_idx]["VoL_high_mag"].values[0]

    coordinate = low_mag_df[low_mag_df["idx"] == region_idx]["coordinate"].values[0]

    # get the wsi_name, pipeline, and datetime_processed
    datetime_processed = subdir.split("_")[1]
    pipeline = subdir.split("_")[0]
    
    row = pipeline_run_history[(pipeline_run_history["pipeline"] == pipeline) & (pipeline_run_history["datetime_processed"] == datetime_processed)]
    wsi_name = row["wsi_name"].values[0]

    cell_metadata["pseudo_idx"].append(pseudo_idx)
    cell_metadata["data_source_type"].append(data_source_type)
    cell_metadata["wsi_name"].append(wsi_name)
    cell_metadata["pipeline"].append(pipeline)
    cell_metadata["datetime_processed"].append(datetime_processed)
    cell_metadata["region_idx"].append(region_idx)
    cell_metadata["region_file_path"].append(region_file_path)
    cell_metadata["low_mag_score"].append(low_mag_score)
    cell_metadata["high_mag_score"].append(high_mag_score)
    cell_metadata["low_mag_VoL"].append(low_mag_VoL)
    cell_metadata["high_mag_VoL"].append(high_mag_VoL)
    cell_metadata["coordinate"].append(coordinate)

    # save the image to save_dir/high_mag_rejected/pseudo_idx.jpg
    image = Image.open(region_file_path)
    image.save(os.path.join(save_dir, "high_mag_rejected", f"{pseudo_idx}.jpg"))

    pseudo_idx += 1


for i in tqdm(range(total_num_cells), desc="Sampling Cells from BMA result high mag unannotated folders"):
   # randomly select a subdir in subdirs
    subdir = random.choice(subdirs)

    data_source_type = "high_mag_rejected"

    # get the list of .jpg images in the subdir/focus_regions/high_mag_rejected folder
    high_mag_rejected_files = os.listdir(os.path.join(result_dir, subdir, "focus_regions", "high_mag_rejected"))
    # make sure the list only contains .jpg files
    high_mag_rejected_files = [f for f in high_mag_rejected_files if f.endswith(".jpg")]
    
    while len(high_mag_rejected_files) == 0:
        subdir = random.choice(subdirs)
        high_mag_rejected_files = os.listdir(os.path.join(result_dir, subdir, "focus_regions", "high_mag_rejected"))
        high_mag_rejected_files = [f for f in high_mag_rejected_files if f.endswith(".jpg")]

    # randomly select a high_mag_rejected_file
    high_mag_rejected_file = random.choice(high_mag_rejected_files)

    # get the region_idx from the file name
    region_idx = int(high_mag_rejected_file[:-4])

    # get the path to the high_mag_rejected_file
    region_file_path = os.path.join(result_dir, subdir, "focus_regions", "high_mag_rejected", high_mag_rejected_file)

    # get the low_mag_score, high_mag_score, low_mag_VoL, high_mag_VoL, and coordinate
    low_mag_csv_path = os.path.join(result_dir, subdir, "focus_regions", "focus_regions_info.csv")
    high_mag_csv_path = os.path.join(result_dir, subdir, "focus_regions", "high_mag_focus_regions_info.csv")    

    low_mag_df = pd.read_csv(low_mag_csv_path)
    high_mag_df = pd.read_csv(high_mag_csv_path)

    low_mag_score = low_mag_df[low_mag_df["idx"] == region_idx]["adequate_confidence_score"].values[0]
    high_mag_score = high_mag_df[high_mag_df["idx"] == region_idx]["adequate_confidence_score_high_mag"].values[0]

    low_mag_VoL = low_mag_df[low_mag_df["idx"] == region_idx]["VoL"].values[0]
    high_mag_VoL = high_mag_df[high_mag_df["idx"] == region_idx]["VoL_high_mag"].values[0]

    coordinate = low_mag_df[low_mag_df["idx"] == region_idx]["coordinate"].values[0]

    # get the wsi_name, pipeline, and datetime_processed
    datetime_processed = subdir.split("_")[1]
    pipeline = subdir.split("_")[0]
    
    row = pipeline_run_history[(pipeline_run_history["pipeline"] == pipeline) & (pipeline_run_history["datetime_processed"] == datetime_processed)]
    wsi_name = row["wsi_name"].values[0]

    cell_metadata["pseudo_idx"].append(pseudo_idx)
    cell_metadata["data_source_type"].append(data_source_type)
    cell_metadata["wsi_name"].append(wsi_name)
    cell_metadata["pipeline"].append(pipeline)
    cell_metadata["datetime_processed"].append(datetime_processed)
    cell_metadata["region_idx"].append(region_idx)
    cell_metadata["region_file_path"].append(region_file_path)
    cell_metadata["low_mag_score"].append(low_mag_score)
    cell_metadata["high_mag_score"].append(high_mag_score)
    cell_metadata["low_mag_VoL"].append(low_mag_VoL)
    cell_metadata["high_mag_VoL"].append(high_mag_VoL)
    cell_metadata["coordinate"].append(coordinate)

    # save the image to save_dir/high_mag_rejected/pseudo_idx.jpg
    image = Image.open(region_file_path)
    image.save(os.path.join(save_dir, "high_mag_rejected", f"{pseudo_idx}.jpg"))

    pseudo_idx += 1



ndpi_slide_folders = [
    "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/BMA_AML",
    "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/BMA_Normal",
    "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/BMA_PCM",
    "/media/hdd3/neo/BMA_MDS_EB1_EB2",
    "/media/hdd3/neo/BMA_MDS_non_EB1_EB2", 
]

# find all the ndpi files in the ndpi_slide_folders
ndpi_paths = []

for ndpi_slide_folder in ndpi_slide_folders:
    ndpi_paths += [os.path.join(ndpi_slide_folder, f) for f in os.listdir(ndpi_slide_folder) if f.endswith(".ndpi")]

for i in tqdm(range(total_num_cells), desc="Sampling Cells from NDPI slides"):
    # randomly select a ndpi_path
    ndpi_path = random.choice(ndpi_paths)

    # open the ndpi file
    slide = openslide.OpenSlide(ndpi_path)

    # randomly sample a coordinate (TL_x, TL_y, BR_x, BR_y) from the slide of tile size 512x512
    TL_x = random.randint(0, slide.dimensions[0] - 512)
    TL_y = random.randint(0, slide.dimensions[1] - 512)
    BR_x = TL_x + 512
    BR_y = TL_y + 512

    coordinate = f"({TL_x},{TL_y},{BR_x},{BR_y})"

    # extract the region from the slide
    region = slide.read_region((TL_x, TL_y), 0, (512, 512))

    # save the region to save_dir/ndpi/pseudo_idx.jpg
    region.save(os.path.join(save_dir, "ndpi", f"{pseudo_idx}.jpg"))

    data_source_type = "ndpi"

    # get the wsi_name, pipeline, and datetime_processed
    wsi_name = os.path.basename(ndpi_path)
    pipeline = "NA"
    datetime_processed = "NA"

    cell_metadata["pseudo_idx"].append(pseudo_idx)
    cell_metadata["data_source_type"].append(data_source_type)
    cell_metadata["wsi_name"].append(wsi_name)
    cell_metadata["pipeline"].append(pipeline)
    cell_metadata["datetime_processed"].append(datetime_processed)
    cell_metadata["region_idx"].append("NA")
    cell_metadata["region_file_path"].append("NA")
    cell_metadata["low_mag_score"].append("NA")
    cell_metadata["high_mag_score"].append("NA")
    cell_metadata["low_mag_VoL"].append("NA")
    cell_metadata["high_mag_VoL"].append("NA")
    cell_metadata["coordinate"].append(coordinate)

    pseudo_idx += 1

cell_metadata_df = pd.DataFrame(cell_metadata)
cell_metadata_df.to_csv(os.path.join(save_dir, "cell_metadata.csv"), index=False)


    