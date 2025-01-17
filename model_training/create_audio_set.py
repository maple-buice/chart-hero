from model_training.data_preparation import data_preparation

class create_audio_set:
    def __init__(self, data_set_path, audio_set_path):

        # Initiate the class
        # A few parameter need to be specified here
        # directory_path is the path to the root directory of the dataset. This class assume the use of GMD / E-GMD dataset
        # dataset is the type of data we use,  either "gmd" or "egmd" is acceptable for now
        # sample_ratio is the fraction of dataset want to be used in creating the training/test/eval dataset
        # diff_threshold is a parameter to ilter out the midi/audio pair that has the  duration difference > specified  value (in seconds) 

        data_container=data_preparation(directory_path=data_set_path, dataset='egmd', sample_ratio=1, diff_threshold=1)

        # To create the training data, just simply do this
        # A few parameters here need to aware of

        # pad_before control the padding added to the begining of each clip. Default setting is 0.02 seconds

        # pad_after control the padding added to the end of each clip. Default setting is 0.02 seconds

        # fix_length control the total length of each extracted clip. accpet value in seconds. If this is not None, the function will ignore pad_after parameter because fix_length is already adding padding to the eacd of each clip 

        # batching control the batching implementation. Only set thei to True if you are processing >10% of egmd dataset. The egmd data have a size of ~110Gb, there is no way you can store it all in your company memory unless you have a very powerful machine  
        # By default, it will divide the dataset into 50 batches and created 50 pkl files. If batching is true, the function will also do the train test split automatically, so you will see a set of 50 training pkl files, 50 val pkl files.... created etc.

        # dir_path control the directory path of the store location of those output pkl files

        data_container.create_audio_set(pad_before=0.02, pad_after=0, fix_length=0.2, batching=True, dir_path=audio_set_path)
