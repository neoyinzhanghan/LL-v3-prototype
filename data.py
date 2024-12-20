import os 
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class LowMagRegionDataset(Dataset):
    """

    === Attributes ===
    slide_prototype_path: str
        The path to the slide prototype folder
    low_mag_regions_path: list 
        The path to the low mag regions images
    
    """
    def __init__(self, slide_prototype_path):
        self.slide_prototype_path = slide_prototype_path

        # get the paths to all the jpeg images in the slide_prototype_path/18_downsampled folder
        self.low_mag_regions_path = [os.path.join(slide_prototype_path, "18_downsampled", name) for name in os.listdir(os.path.join(slide_prototype_path, "18_downsampled")) if name.endswith(".jpeg")]

    def __len__(self):
        return len(self.low_mag_regions_path)

    def __getitem__(self, idx):
        sample = Image.open(self.low_mag_regions_path[idx])

        # get the image file name
        image_name = os.path.basename(self.low_mag_regions_path[idx])

        return sample, image_name
    
class HighMagRegionDataset(Dataset):
    """ 
    === Attributes ===  
    slide_prototype_path: str
        The path to the slide prototype folder
    high_mag_regions_paths: list
        The path to the high mag regions images

    """

    def __init__(self, slide_prototype_path, image_names):
        self.slide_prototype_path = slide_prototype_path

        # get the paths to all the jpeg images in the slide_prototype_path/18
        self.high_mag_regions_paths = [os.path.join(slide_prototype_path, "18", name) for name in image_names]

    def __len__(self):
        return len(self.high_mag_regions_paths)
    
    def __getitem__(self, idx):
        sample = Image.open(self.high_mag_regions_paths[idx])

        # get the image file name
        image_name = os.path.basename(self.high_mag_regions_paths[idx])

        return sample, image_name
    
class AllHighMagRegionDataset(Dataset):
    """ 
    === Attributes ===  
    slide_prototype_path: str
        The path to the slide prototype folder
    high_mag_regions_paths: list
        The path to the high mag regions images

    """

    def __init__(self, slide_prototype_path):
        self.slide_prototype_path = slide_prototype_path

        # get the paths to all the jpeg images in the slide_prototype_path/18
        self.high_mag_regions_paths = [os.path.join(slide_prototype_path, "18", name) for name in os.listdir(os.path.join(slide_prototype_path, "18")) if name.endswith(".jpeg")]

    def __len__(self):
        return len(self.high_mag_regions_paths)
    
    def __getitem__(self, idx):
        sample = Image.open(self.high_mag_regions_paths[idx])

        # get the image file name
        image_name = os.path.basename(self.high_mag_regions_paths[idx])

        return sample, image_name    