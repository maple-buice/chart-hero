# %%
import os
from pathlib import Path

from utils.file_utils import get_audio_set_dir, get_labeled_audio_set_dir, get_process_later_dir

def re_order_npy_file_name_pieces():
    labeled_audio_set_sir = get_labeled_audio_set_dir()
    
    for file in os.listdir(labeled_audio_set_sir):
        if file.endswith('.npy'):
            split_file_name = Path(file).stem.split('_')
            new_file_name = split_file_name[1] + '_' + split_file_name[0] + '_' + split_file_name[2] + '.npy'
            os.rename(os.path.join(labeled_audio_set_sir, file), os.path.join(labeled_audio_set_sir, new_file_name))
    
# %%
# data set is too big, laptop is too weak, so let's a third
def thin_out_dataset(base_path):
    process_later_path = get_process_later_dir()
    
    if not os.path.exists(base_path):
        return
        
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
        
        if int(batch_number) % 3 == 0:
            continue
          
        print(os.path.join(base_path, file) + ' --> ' + os.path.join(process_later_base_path, file))  
        os.rename(os.path.join(base_path, file),
                  os.path.join(process_later_base_path, file))

thin_out_dataset(get_audio_set_dir())
thin_out_dataset(get_labeled_audio_set_dir())

# %%
