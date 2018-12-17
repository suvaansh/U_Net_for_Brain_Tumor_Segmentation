# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 23:46:05 2018

@author: suvaansh, joydeep
"""

import tensorlayer as tl
import numpy as np
import os, csv, gc, pickle
import nibabel as nib
from skimage import io

"""
In seg file
--------------
Label 1: necrotic and non-enhancing tumor
Label 2: edema 
Label 4: enhancing tumor
Label 0: background

MRI
-------
whole/complete tumor: 1 2 4
core: 1 4
enhance: 4
"""
###============================= SETTINGS ===================================###
DATA_SIZE = 'all' # (small, half or all)

save_dir = 'data/train_dev_all/'
if not os.path.exists(save_dir):
    os.makedir(save_dir)

HGG_data_path = "../Brats17TrainingData/HGG"
LGG_data_path = "../Brats17TrainingData/LGG"
###==========================================================================###
if DATA_SIZE == 'all':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)[0:10]

HGG_name_list = [os.path.basename(p) for p in HGG_path_list]

data_types = ['flair','t1', 't1ce', 't2']
data_types_mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in data_types}

for i in data_types:
    print(i)
    data_temp_list = []
    for j in HGG_name_list:
        print(j)
        img_path = os.path.join(HGG_data_path, j, j + '_' + i + '.nii.gz')
        img = nib.load(img_path).get_data()
        data_temp_list.append(img)

    data_temp_list = np.asarray(data_temp_list)
    m = np.mean(data_temp_list)
    s = np.std(data_temp_list)
    del data_temp_list
    data_types_mean_std_dict[i]['mean'] = m
    data_types_mean_std_dict[i]['std'] = s
print(data_types_mean_std_dict)
with open(save_dir + 'mean_std_dict.pickle', 'w') as f:
    pickle.dump(data_types_mean_std_dict, f)

##==================== GET NORMALIZE IMAGES
X_train_input = []
X_train_target = []

print(" HGG Train")
patient_num = -1
for i in HGG_name_list:
    print(i)
    patient_num = patient_num + 1
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = img.astype(np.float32)
        all_3d_data.append(img)

    seg_path = os.path.join(HGG_data_path, i, i + '_seg.nii.gz')
    seg_img = nib.load(seg_path).get_data()
    seg_img = np.transpose(seg_img, (1, 0, 2))
    slice_ix = -1
    for j in range(all_3d_data[0].shape[2]):
        slice_ix = slice_ix + 1
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float32)
        seg_2d = seg_img[:, :, j]
        
        X_train_input.append(combined_array)
        
        strip_combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j],seg_2d), axis=2)
        strip_combined_array = np.transpose(strip_combined_array, (1, 0, 2))#.tolist()        
        strip_combined_array = np.transpose(strip_combined_array)
        strip_combined_array.astype(np.float32)
        strip = strip_combined_array.reshape(5*240, 240)
        if np.max(strip) > 0 : # set values < 1
            strip /= np.max(strip)
        if np.min(strip) <= -1: # set values > -1
            strip /= abs(np.min(strip))
                # save as patient_slice.png
#        print(np.max(strip))
#        print(np.min(strip))
        
        io.imsave('z_normalized/{}_{}.png'.format(patient_num, slice_ix), strip)
#        seg_2d = seg_img[:, :, j]
        # whole = np.zeros_like(seg_2d)
        # core = np.zeros_like(seg_2d)
        # enhance = np.zeros_like(seg_2d)
        # for index, x in np.ndenumerate(seg_2d):
        #     if x == 1:
        #         whole[index] = 1
        #         core[index] = 1
        #     if x == 2:
        #         whole[index] = 1
        #     if x == 4:
        #         whole[index] = 1
        #         core[index] = 1
        #         enhance[index] = 1
        # X_train_target_whole.append(whole)
        # X_train_target_core.append(core)
        # X_train_target_enhance.append(enhance)
        seg_2d.astype(int)
        X_train_target.append(seg_2d)
        
        
    del all_3d_data
    print("finished {}".format(i))
    # print(len(X_train_target))
print(len(X_train_target))
print(X_train_input[0].shape)
print(X_train_target[0].shape)
