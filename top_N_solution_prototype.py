import os
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from data import LowMagRegionDataset, HighMagRegionDataset
from BMARegionClfManager import load_clf_model, predict_batch
from BMAHighMagRegionChecker import load_model_checkpoint, predict_images_batch
from BMAassumptions import region_clf_ckpt_path, high_mag_region_clf_ckpt_path, high_mag_region_clf_threshold

def process_dataset(dataset_path):
    # Define the low mag test folder paths within the dataset directory
    low_mag_test_path = os.path.join(dataset_path, "low_mag_test")
    top_1000_dir = os.path.join(low_mag_test_path, "top_1000_low_mag")
    selected_dir = os.path.join(low_mag_test_path, "high_mag_selected")
    rejected_dir = os.path.join(low_mag_test_path, "high_mag_rejected")

    # If the folder low_mag_test exists, delete it first
    if os.path.exists(low_mag_test_path):
        shutil.rmtree(low_mag_test_path)

    os.makedirs(low_mag_test_path, exist_ok=True)

    # Create the required subdirectories
    os.makedirs(top_1000_dir, exist_ok=True)
    os.makedirs(selected_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    # Parameters
    batch_size = 256  # Batch size for loading images
    num_workers = 8   # Adjust number of workers based on your CPU cores

    # Initialize the dataset
    dataset = LowMagRegionDataset(dataset_path)

    # Custom collate function to handle PIL images and names
    def custom_collate_fn(batch):
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
    for pil_images, image_names in tqdm(data_loader, desc="Processing Low Mag Batches"):
        scores = predict_batch(pil_images, model)
        results.extend([{"image_name": name, "low_mag_score": score} for name, score in zip(image_names, scores)])

    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(low_mag_test_path, "region_predictions.csv"), index=False)

    # Sort the results_df by low_mag_score and find the top 1000 images
    results_df = results_df.sort_values(by="low_mag_score", ascending=False)
    top_1000 = results_df["image_name"].head(1000).values

    # Save the top 1000 images to the designated folder
    for image_name in tqdm(top_1000, desc="Saving Top 1000 Low Mag Images"):
        image_path = os.path.join(dataset_path, "18_downsampled", image_name)
        image = Image.open(image_path)
        image.save(os.path.join(top_1000_dir, image_name))

    # Initialize the HighMagRegionDataset
    high_mag_dataset = HighMagRegionDataset(dataset_path, top_1000)

    # Initialize the DataLoader with specified batch size, number of workers, and custom collate function
    data_loader = DataLoader(high_mag_dataset, batch_size=batch_size, num_workers=num_workers, 
                             shuffle=False, collate_fn=custom_collate_fn)

    # Load the high mag region classifier model
    high_mag_model = load_model_checkpoint(high_mag_region_clf_ckpt_path)

    # Prepare lists to store results
    results = []

    # Iterate through the dataset with a DataLoader and progress bar
    for pil_images, image_names in tqdm(data_loader, desc="Processing High Mag Batches"):
        scores = predict_images_batch(high_mag_model, pil_images)
        results.extend([{"image_name": name, "high_mag_score": score} for name, score in zip(image_names, scores)])

    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(low_mag_test_path, "high_mag_region_predictions.csv"), index=False)

    # Filter the results based on the high_mag_region_clf_threshold
    selected = results_df[results_df["high_mag_score"] > high_mag_region_clf_threshold]
    rejected = results_df[results_df["high_mag_score"] <= high_mag_region_clf_threshold]

    # Save the selected and rejected images to their respective folders
    for image_name in tqdm(selected["image_name"], desc="Saving Selected High Mag Images"):
        image_path = os.path.join(dataset_path, "18", image_name)
        image = Image.open(image_path)
        image.save(os.path.join(selected_dir, image_name))

    for image_name in tqdm(rejected["image_name"], desc="Saving Rejected High Mag Images"):
        image_path = os.path.join(dataset_path, "18", image_name)
        image = Image.open(image_path)
        image.save(os.path.join(rejected_dir, image_name))

if __name__ == "__main__":
    import time
    metadata_path = "/media/hdd3/neo/error_slides_dzsave/already_downsampled.csv"
    metadata = pd.read_csv(metadata_path)

    # get the dzsave_dir column in metadata as a list
    dzsave_dirs = metadata["dzsave_dir"].tolist()

    runtime_metadata = {
        "dzsave_dir": [],
        "processing_time": []
    }

    for dzsave_dir in tqdm(dzsave_dirs, desc="Processing Datasets"):
        print(f"Processing {dzsave_dir}...")
        start_time = time.time()
        process_dataset(dzsave_dir)
        end_time = time.time()

        runtime_metadata["dzsave_dir"].append(dzsave_dir)
        runtime_metadata["processing_time"].append(end_time - start_time)

    runtime_metadata_df = pd.DataFrame(runtime_metadata)
    runtime_metadata_df.to_csv("top_N_solution_prototype_runtime_metadata.csv", index=False)