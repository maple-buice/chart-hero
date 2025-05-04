import os
from model_training.data_preparation import data_preparation

from utils.file_utils import get_audio_set_dir, get_dataset_dir, get_labeled_audio_set_dir

def create_audio_set():
    if any(os.scandir(get_audio_set_dir())) or any(os.scandir(get_labeled_audio_set_dir())):
        return
    
    # Initiate the data container
    # Parameters:
    # directory_path: the path to the root directory of the dataset. This class assumes the use of GMD / E-GMD dataset 
    # dataset: the type of data we use,  either "gmd" or "egmd" is acceptable for now
    # sample_ratio: the fraction of dataset want to be used in creating the training/test/eval dataset
    # diff_threshold: a parameter to filter out the midi/audio pair that has the duration difference > specified value (in seconds) 
    # n_jobs: the number of CPU cores to use for parallel processing. -1 means using all available cores
    data_container = data_preparation(
        directory_path=get_dataset_dir(),
        dataset='egmd',
        sample_ratio=1,
        diff_threshold=1,
        n_jobs=-1) # Use all available CPU cores

    # Parameters
    # pad_before: padding added to the begining of each clip. Default setting is 0.02 seconds
    # pad_after: padding added to the end of each clip. Default setting is 0.02 seconds
    # fix_length: the total length of each extracted clip. accept value in seconds. If this is not None, the function will ignore pad_after parameter because fix_length is already adding padding to the eacd of each clip
    # batching: control the batching implementation. Only set this to True if you are processing >10% of egmd dataset. The egmd data have a size of ~110Gb, there is no way you can store it all in your computer memory unless you have a very powerful machine.
    #   By default, it will divide the dataset into 50 batches and create 50 pkl files. If batching is true, the function will also do the train test split automatically, so you will see a set of 50 training pkl files, 50 val pkl files.... created etc.
    # dir_path: the directory path of the store location of those output pkl files
    # num_batches: the number of batches to divide the dataset into
    data_container.create_audio_set(
        pad_before=0.02,
        pad_after=0,
        fix_length=0.2,
        batching=True,
        dir_path=get_audio_set_dir(),
        num_batches=50) # Example: keep it at 50, but now it's an explicit parameter
