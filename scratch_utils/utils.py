import os
from pathlib import Path

training_data_path = os.path.join(os.getcwd(), 'training_data')
e_gmd_path = os.path.join(training_data_path, 'e-gmd-v1.0.0')
audio_set_path = os.path.join(training_data_path, 'audio_set')


# %%
# re-order npy files name pieces
npy_data_path = os.path.join(training_data_path, 'npy_data')

for file in os.listdir(npy_data_path):
    if file.endswith('.npy'):
      split_file_name = Path(file).stem.split('_')
      new_file_name = split_file_name[1] + '_' + split_file_name[0] + '_' + split_file_name[2] + '.npy'
      os.rename(os.path.join(npy_data_path, file), os.path.join(npy_data_path, new_file_name))
