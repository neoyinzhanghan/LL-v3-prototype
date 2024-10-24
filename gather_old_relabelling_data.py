import os
import shutil
import pandas as pd
from tqdm import tqdm


splitted_data_dir = "/media/hdd3/neo/bma_region_clf_data_full_v2_split"

save_data_dir = "/media/hdd3/neo/bma_region_clf_data_v2_pooled"

if os.path.exists(save_data_dir):
    shutil.rmtree(save_data_dir)
os.makedirs(save_data_dir, exist_ok=True)
os.makedirs(os.path.join(save_data_dir, "adequate"), exist_ok=True)
os.makedirs(os.path.join(save_data_dir, "not_adequate"), exist_ok=True)

combined_md = {
    "image_path": [],
    "label": [],
}

# combine the splitted_data_dir/train_metadata.csv, splitted_data_dir/val_metadata.csv, and splitted_data_dir/test_metadata.csv
train_metadata = pd.read_csv(os.path.join(splitted_data_dir, "train_metadata.csv"))
val_metadata = pd.read_csv(os.path.join(splitted_data_dir, "val_metadata.csv"))
test_metadata = pd.read_csv(os.path.join(splitted_data_dir, "test_metadata.csv"))

combined_metadata = pd.concat([train_metadata, val_metadata, test_metadata], ignore_index=True)

for idx, row in tqdm(combined_metadata.iterrows(), total=len(combined_metadata), desc="Combining Region Data"):

    if row["label"] == "adequate":
        file_basename = os.path.basename(row["image_path"])
        new_path = os.path.join(save_data_dir, "adequate", file_basename)
        shutil.copy(row["image_path"], new_path)
    elif row["label"] == "not_adequate":
        file_basename = os.path.basename(row["image_path"])
        new_path = os.path.join(save_data_dir, "not_adequate", file_basename)
        shutil.copy(row["image_path"], new_path)

    combined_md["image_path"].append(new_path)
    combined_md["label"].append(row["label"])

combined_md_df = pd.DataFrame(combined_md)
combined_md_df.to_csv(os.path.join(save_data_dir, "metadata.csv"), index=False)