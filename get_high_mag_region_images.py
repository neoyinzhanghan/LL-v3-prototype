import os
import pandas as pd
from tqdm import tqdm

region_metadata_path = "/media/hdd3/neo/bma_high_mag_rejected_regions/regions_metadata.csv"
metadata = pd.read_csv(region_metadata_path)

save_dir  = "/media/hdd3/neo/bma_high_mag_rejected_regions/regions"
os.makedirs(save_dir, exist_ok=True)

