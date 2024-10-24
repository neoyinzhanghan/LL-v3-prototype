import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from data import LowMagRegionDataset
from BMARegionClfManager import load_clf_model, predict_batch
from BMAassumptions import region_clf_ckpt_path

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
    results.extend([{"image_name": name, "score": score} for name, score in zip(image_names, scores)])

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv("region_predictions.csv", index=False)
