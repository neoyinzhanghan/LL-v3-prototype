import os
import openslide
import pandas as pd
from PIL import Image
from tqdm import tqdm

save_dir_ndpi = "/media/hdd3/neo/error_slides_ndpi"
save_dir_dzsave = "/media/hdd3/neo/error_slides_dzsave"

# find all ndpi files in save_dir_ndpi
ndpi_files = [f for f in os.listdir(save_dir_ndpi) if f.endswith(".ndpi")]

for ndpi_file in tqdm(ndpi_files, desc="Extracting Level 7 Images"):

    ndpi_file_no_ext = ndpi_file[:-5]
    ndpi_path = os.path.join(save_dir_ndpi, ndpi_file)

    # open the ndpi file
    slide = openslide.OpenSlide(ndpi_path)

    # get the level 7 dimensions
    level_7_dimensions = slide.level_dimensions[7]

    # extract the level 7 image
    level_7_image = slide.read_region((0, 0), 7, level_7_dimensions) 

    # if rgba, convert to rgb
    if level_7_image.mode == "RGBA":
        level_7_image = level_7_image.convert("RGB")

    # save the level 7 image as a .jpeg file named topview.jpeg in the dzsave_dir
    level_7_image.save(os.path.join(save_dir_dzsave, ndpi_file_no_ext, "topview.jpeg"))

