import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

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

metadata_dict = {
    "idx": range(len(all_image_paths)),
    "image_path": all_image_paths,
}

# save the metadata to a csv file
metadata_df = pd.DataFrame(metadata_dict)
metadata_df.to_csv(
    os.path.join(dzsave_dir, "all_high_mag_image_paths.csv"), index=False
)

# class ImagePathDataset(Dataset):
#     def __init__(self, image_paths):
#         """
#         Args:
#             image_paths (list of str): List of paths to image files.
#         """
#         self.image_paths = image_paths

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]

#         # Load image
#         image = Image.open(img_path).convert("RGB")  # Ensure RGB format

#         return idx, image, img_path
