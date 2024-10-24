import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from data import LowMagRegionDataset, HighMagRegionDataset
from BMARegionClfManager import load_clf_model, predict_batch
from BMAHighMagRegionChecker import load_model_checkpoint, predict_images_batch
from BMAassumptions import region_clf_ckpt_path, high_mag_region_clf_ckpt_path, high_mag_region_clf_threshold

# Define the dataset path
dataset_path = '/media/hdd3/neo/error_slides_dzsave/H21-9456;S9;MSK9 - 2023-05-19 13.58.34'

# Parameters
batch_size = 256  # Batch size for loading images
num_workers = 8   # Adjust number of workers based on your CPU cores

# Initialize the dataset
dataset = LowMagRegionDataset(dataset_path)

# Custom collate function to handle PIL images and names
def custom_collate_fn(batch):
    # Batch is a list of tuples (PIL image, image_name)
    pil_images, image_names = zip(*batch)
    return list(pil_images), list(image_names)

# Initialize the DataLoader with specified batch size, number of workers, and custom collate function
data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                         shuffle=False, collate_fn=custom_collate_fn)

# Load the model
model = load_clf_model(region_clf_ckpt_path)

# Prepare lists to store results
results = []

# Iterate through the dataset with a DataLoader and progress bar
for pil_images, image_names in tqdm(data_loader, desc="Processing Batches"):
    # Predict batch of images
    scores = predict_batch(pil_images, model)
    # Append the results
    results.extend([{"image_name": name, "low_mag_score": score} for name, score in zip(image_names, scores)])

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv("low_mag_test/region_predictions.csv", index=False)

# sort the results_df by low_mag_score
results_df = results_df.sort_values(by="low_mag_score", ascending=False)

# find image_name that is the top 1000
top_1000 = results_df["image_name"].head(1000).values

# save the top 1000 images to the folder test/top_1000_low_mag, path relative to this script
top_1000_dir = "low_mag_test/top_1000_low_mag"

for image_name in tqdm(top_1000, desc="Saving Top 1000 Images"):
    image_path = os.path.join(dataset_path, "18_downsampled", image_name)
    image = Image.open(image_path)
    image.save(os.path.join(top_1000_dir, image_name))

high_mag_dataset = HighMagRegionDataset(dataset_path, top_1000)

# Initialize the DataLoader with specified batch size, number of workers, and custom collate function
data_loader = DataLoader(high_mag_dataset, batch_size=batch_size, num_workers=num_workers, 
                         shuffle=False, collate_fn=custom_collate_fn)

# Load the high mag region classifier model
high_mag_model = load_model_checkpoint(high_mag_region_clf_ckpt_path)

# Prepare lists to store results
results = []

# Iterate through the dataset with a DataLoader and progress bar
for pil_images, image_names in tqdm(data_loader, desc="Processing Batches"):
    # Predict batch of images
    scores = predict_images_batch(pil_images, high_mag_model)
    # Append the results
    results.extend([{"image_name": name, "high_mag_score": score} for name, score in zip(image_names, scores)])

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv("low_mag_test/high_mag_region_predictions.csv", index=False)

# Filter the results based on the high_mag_region_clf_threshold
selected = results_df[results_df["high_mag_score"] > high_mag_region_clf_threshold]
rejected = results_df[results_df["high_mag_score"] <= high_mag_region_clf_threshold]

# save the selected and rejected images to the folder test/high_mag_selected and test/high_mag_rejected, path relative to this script
selected_dir = "low_mag_test/high_mag_selected"
rejected_dir = "low_mag_test/high_mag_rejected"

for image_name in tqdm(selected["image_name"], desc="Saving Selected Images"):
    image_path = os.path.join(dataset_path, "18", image_name)
    image = Image.open(image_path)
    image.save(os.path.join(selected_dir, image_name))

for image_name in tqdm(rejected["image_name"], desc="Saving Rejected Images"):
    image_path = os.path.join(dataset_path, "18", image_name)
    image = Image.open(image_path)
    image.save(os.path.join(rejected_dir, image_name))