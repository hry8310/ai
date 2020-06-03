import os
import numpy as np
import logging
import cv2

import torch
from torch.utils.data import Dataset

class ImgDataset(Dataset):
    def __init__(self, dir_path, img_size, is_training, is_debug=False):
        self.img_path_list_shuffled=[]
        self.label_list_shuffled=[]
        self.img_size=img_size
        self.is_debug=is_debug
        img_path_list = []
        label_list = []
        img_count = 0
        shuffle=True
        folder_names=['cat', 'dog']
        for folder_name in folder_names:
            filenames = os.listdir(os.path.join(dir_path, folder_name))
            for f in filenames:
                img_path_list.append(os.path.join(dir_path, folder_name, f))
                label_list.append([0,1] if 'cat' in f else [1,0])
                img_count += 1
        img_path_list = np.array(img_path_list)
        label_list = np.array(label_list)
        if shuffle == True:
            index = np.random.permutation(np.arange(0, img_count, 1))
            self.img_path_list_shuffled = img_path_list[index]
            self.label_list_shuffled = label_list[index]
        else:
            self.mg_path_list_shuffled = img_path_list
            self.label_list_shuffled = label_list

    def _compose(self,sample):
        image, label = sample['image'], sample['label']
        if self.is_debug == False : 
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
        label=label.astype(np.float32)
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy( label)}
 
 
    def _resize(self, sample ,intp=cv2.INTER_LINEAR):
        image, label = sample['image'], sample['label']
        image = cv2.resize(image, tuple(self.img_size), interpolation=intp)
        return {'image': image, 'label': label}


    

    def __getitem__(self, index):
        img_path = self.img_path_list_shuffled[index % len(self.img_path_list_shuffled)].rstrip()
        labels = self.label_list_shuffled[index % len(self.img_path_list_shuffled)]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h, ori_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sample = {'image': img, 'label': labels}
        sample=self._resize(sample) 
        #print(sample['image'].shape)
        sample=self._compose(sample) 
        #sample=self._resize(sample) 

        sample["image_path"] = img_path
        sample["origin_size"] = str([ori_w, ori_h])
        return sample

    def __len__(self):
        return len(self.label_list_shuffled)


