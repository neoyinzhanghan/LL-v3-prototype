from data import LowMagRegionDataset
from BMARegionClfManager import load_clf_model, predict_batch
from BMAassumptions import region_clf_ckpt_path

def run_one_slide(slide_prototype_path):
    """"""

    # step 1 run the low magnification region classifier on every region images
    # step 2 take the top 1000, then run the high magnification region classifier
    # step 3 save the top 1000 and then save the selected ones based on high magnification

    # load the low mag region dataset

    pass