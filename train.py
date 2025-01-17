# %%
# Import packages and declare variables

from model_training.train_model import train_model
from model_training.label_data import label_data

# %%
# create the audio set from the EGMD dataset
# create_audio_set(e_gmd_path, audio_set_path)

# %%
# label the data
label_data()

# %
train_model()

# %%
# Finish.
