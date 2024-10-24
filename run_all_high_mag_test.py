import os
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from data import AllHighMagRegionDataset
from BMARegionClfManager import load_clf_model, predict_batch
from BMAHighMagRegionChecker import load_model_checkpoint, predict_images_batch
from BMAassumptions import region_clf_ckpt_path, high_mag_region_clf_ckpt_path, high_mag_region_clf_threshold

# Define the dataset path
dataset_path = '/media/hdd3/neo/error_slides_dzsave/H21-9456;S9;MSK9 - 2023-05-19 13.58.34'

# if the folder high_mag_test does not exist, create it, also create the folder high_mag_test/high_mag_selected
os.makedirs("high_mag_test", exist_ok=True) 
os.makedirs("high_mag_test/high_mag_selected", exist_ok=True)

# if the folder high_mag_test does exit, then delete it
if os.path.exists("high_mag_test"):
    shutil.rmtree("high_mag_test")

    # then create the folder high_mag_test
    os.makedirs("high_mag_test", exist_ok=True) 
    os.makedirs("high_mag_test/high_mag_selected", exist_ok=True)


# Parameters
batch_size = 256  # Batch size for loading images
num_workers = 8   # Adjust number of workers based on your CPU cores

# Initialize the dataset
dataset = AllHighMagRegionDataset(dataset_path)

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
    results.extend([{"image_name": name, "high_mag_score": score} for name, score in zip(image_names, scores)])

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv("high_mag_test/region_predictions.csv", index=False)

# find the where the score is greater than high_mag_region_clf_threshold
high_mag_selected = results_df[results_df["high_mag_score"] > high_mag_region_clf_threshold]

# save the high_mag_selected images to the folder high_mag_test/high_mag_selected, path relative to this script
high_mag_selected_dir = "high_mag_test/high_mag_selected"

for image_name in tqdm(high_mag_selected["image_name"], desc="Saving High Mag Selected Images"):
    image_path = os.path.join(dataset_path, "18", image_name)
    image = Image.open(image_path)
    image.save(os.path.join(high_mag_selected_dir, image_name))

# print the total number of high_mag_selected images
print(f"Found {len(high_mag_selected)} high mag selected images.")