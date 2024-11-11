import os
import torch
import random
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from BMAHighMagRegionChecker import load_model_checkpoint, predict_images_batch

dzsave_dir = "/media/hdd3/neo/error_slides_dzsave/"

# first get all the subdirectories in dzsave_dir
subdirs = [f.path for f in os.scandir(dzsave_dir) if f.is_dir()]

all_image_paths = []

for subdir in tqdm(subdirs, desc="Gathering Image Paths"):
    # get all the .jpeg files in the subdir
    image_paths = [
        os.path.join(subdir, "18", f)
        for f in os.listdir(os.path.join(subdir, "18"))
        if f.endswith(".jpeg")
    ]

    all_image_paths.extend(image_paths)

print(f"Found {len(all_image_paths)} images.")

# randomly select 2048 images to run the high mag region classifier on
selected_image_paths = random.sample(all_image_paths, 2048)

metadata_dict = {
    "idx": range(len(selected_image_paths)),
    "image_path": selected_image_paths,
}

# save the metadata to a csv file
metadata_df = pd.DataFrame(metadata_dict)
metadata_df.to_csv(
    os.path.join(dzsave_dir, "all_high_mag_image_paths.csv"), index=False
)


class ImagePathDataset(Dataset):
    def __init__(self, metadata_path):
        """
        Args:
            metadata_path: str
                The path to the metadata
            metadata: pd.DataFrame

        """
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)

        # print the number of rows in the metadata at initialization
        print(f"Selected {len(self.metadata)} images loaded into dataloader.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.loc[idx, "image_path"]

        # Load image
        image = Image.open(img_path).convert("RGB")  # Ensure RGB format

        return idx, image


def custom_collate_function(batch):
    """
    This function is used to collate the batch of images and their indices.
    """
    indices, images = zip(*batch)

    return indices, images


# now create a dataloaders for the ImagePathDataset
metadata_path = os.path.join(dzsave_dir, "all_high_mag_image_paths.csv")
dataset = ImagePathDataset(metadata_path)
model_ckpt_path = "/media/hdd3/neo/MODELS/2024-11-07_BMARegionClf-20K/1/version_0/checkpoints/epoch=64-step=21515.ckpt"
model = load_model_checkpoint(model_ckpt_path)
metadata_df = pd.read_csv(metadata_path)
# add a new column to the metadata_df to store the high_mag_score, initialize to missing float
metadata_df["high_mag_score"] = None

# Parameters
batch_size = 256  # Batch size for loading images

# Initialize the DataLoader with specified batch size, number of workers, and custom collate function
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=False,
    collate_fn=custom_collate_function,
)

for idx, images in tqdm(data_loader, desc="Processing Batches"):

    scores = predict_images_batch(model, images)

    # udpate the metadata_df with the scores
    metadata_df.loc[idx, "high_mag_score"] = scores

# remove all the rows with missing high_mag_score
metadata_df = metadata_df.dropna(subset=["high_mag_score"])

# save the metadata_df to a csv file WITH the high_mag_score column  file name should indicate that it has been processed
metadata_df.to_csv(
    os.path.join(dzsave_dir, "all_high_mag_image_paths_processed.csv"), index=False
)

print("Done.")

image_save_dir = "/media/hdd3/neo/region_clf_image_samples"
os.makedirs(image_save_dir, exist_ok=True)


def float_to_str(x):
    return f"{x:.6f}".replace(".", "").lstrip("0")


for idx, row in tqdm(metadata_df.iterrows(), desc="Copying Images"):
    old_image_path = row["image_path"]
    # new image_path should be the first 6 significant digits of the high_mag_score, make sure to remove the decimal
    new_image_path = os.path.join(
        image_save_dir,
        f"{float_to_str(float(row['high_mag_score']))}_{os.path.basename(old_image_path)}",
    )

    shutil.copy(old_image_path, new_image_path)
