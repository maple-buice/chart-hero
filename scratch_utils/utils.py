# %%
import os
from pathlib import Path

def re_order_npy_file_name_pieces():
    for file in os.listdir(npy_data_path):
        if file.endswith('.npy'):
            split_file_name = Path(file).stem.split('_')
            new_file_name = split_file_name[1] + '_' + split_file_name[0] + '_' + split_file_name[2] + '.npy'
            os.rename(os.path.join(npy_data_path, file), os.path.join(npy_data_path, new_file_name))
    
# %%
# data set is too big, laptop is too weak, so let's do the evens
def thin_out_dataset(base_path, process_later_path):
    if not os.path.exists(base_path):
        return
    if not os.path.exists(process_later_path):
        os.makedirs(process_later_path)
    process_later_base_path = os.path.join(process_later_path, Path(base_path).stem)
    if not os.path.exists(process_later_base_path):
        os.makedirs(process_later_base_path)
    
    for file in os.listdir(base_path):
        batch_number = Path(file).stem.split('_')[0]
        if not str.isnumeric(batch_number):
            continue
        
        # batch_present_in_npy = False
        # for file in os.listdir(npy_data_path):
        #     if file.startswith(batch_number):
        #         batch_present_in_npy = True
        #         break
        
        if int(batch_number) % 2 == 0:
            continue
          
        print(os.path.join(base_path, file) + ' --> ' + os.path.join(process_later_base_path, file))  
        os.rename(os.path.join(base_path, file),
                  os.path.join(process_later_base_path, file))

training_data_path = os.path.join(Path(os.getcwd()).parent, 'training_data')
e_gmd_path = os.path.join(training_data_path, 'e-gmd-v1.0.0')
audio_set_path = os.path.join(training_data_path, 'audio_set')
npy_data_path = os.path.join(training_data_path, 'npy_data')
process_later_path = os.path.join(training_data_path, 'process_later')

thin_out_dataset(audio_set_path, process_later_path)
thin_out_dataset(npy_data_path, process_later_path)

# %%
