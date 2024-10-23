import os
import ray
from PIL import Image
from tqdm import tqdm


@ray.remote
class ImageDownsampler:
    def __init__(self, downsampling_factor):
        self.downsample_factor = downsampling_factor

    def downsample(self, image_path):
        image = Image.open(image_path)

        # Downsample the image
        downsampled_image = image.resize((image.width // self.downsample_factor, image.height // self.downsample_factor))

        # the new save directory is the same as the old directory _downsampled, same file name
        save_dir = os.path.dirname(image_path) + "_downsampled"

        # create the save_dir if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # save the downsampled image
        downsampled_image.save(os.path.join(save_dir, os.path.basename(image_path)))
    
    def async_downsample_batch_by_path(self, batch_paths, downsample_factor):
        """ Batch is a list of image paths."""

        for image_path in batch_paths:
            self.downsample(image_path)


def create_list_of_batches_from_list(list, batch_size):
    """
    This function creates a list of batches from a list.

    :param list: a list
    :param batch_size: the size of each batch
    :return: a list of batches

    >>> create_list_of_batches_from_list([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4], [5]]
    >>> create_list_of_batches_from_list([1, 2, 3, 4, 5, 6], 3)
    [[1, 2, 3], [4, 5, 6]]
    >>> create_list_of_batches_from_list([], 3)
    []
    >>> create_list_of_batches_from_list([1, 2], 3)
    [[1, 2]]
    """

    list_of_batches = []

    for i in range(0, len(list), batch_size):
        batch = list[i : i + batch_size]
        list_of_batches.append(batch)

    return list_of_batches

def downsample_images(image_paths, downsample_factor=8, num_downsamplers=32, downsample_batch_size=256):
    # Initialize a list of downsamplers
    downsamplers = [ImageDownsampler.remote(downsample_factor) for _ in range(num_downsamplers)]

    # Create a list of batches from the image_paths
    batches = create_list_of_batches_from_list(image_paths, downsample_batch_size)

    # Downsample the images in parallel
    downsampled_images = []

    with tqdm(total=len(image_paths), desc="Downsampling Images") as pbar:
        for batch in batches:
            # Distribute the batch to the downsamplers
            futures = [downsampler.async_downsample_batch_by_path.remote(batch, downsample_factor) for downsampler in downsamplers]

            # Wait for the results and update the progress bar by the length of the batch
            results = ray.get(futures)
            downsampled_images.extend(results)
            pbar.update(len(batch))

    return downsampled_images

def downsample_slide_dzsave_dir(dzsave_dir):

    # get all the jpeg image paths in the dzsave_dir/18 folder
    image_paths = [os.path.join(dzsave_dir, "18", image_name) for image_name in os.listdir(os.path.join(dzsave_dir, "18")) if image_name.endswith(".jpeg")]

    # downsample the images
    downsample_images(image_paths)

if __name__ == "__main__":
    # Example usage
    dzsave_dir = "/media/hdd3/neo/error_slides_dzsave/H21-9456;S9;MSK9 - 2023-05-19 13.58.34"
    downsample_slide_dzsave_dir(dzsave_dir)