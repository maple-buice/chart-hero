# %%
# Import packages and declare variables

from model_training.create_audio_set import create_audio_set
from model_training.train_model import train_model
from model_training.label_data import label_data

# %%
# create the audio set from the EGMD dataset
create_audio_set()

# %%
# label the data
label_data()

# %
# create and train the model
train_model()

# %%
# Finish.
