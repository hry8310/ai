import os
import numpy as np
import logging
import cv2

import torch
from torch.utils.data import Dataset



class VOCDataset(Dataset):
    def __init__(self, list_path, img_size, is_training, is_debug=False):
        self.img_files = []
        for path in open(list_path, 'r'):
            self.img_files.append(path)
        self.img_size = img_size  # (w, h)
        self.max_objects = 50
        self.is_debug = is_debug

    
    def _compose(self,sample):
        image, labels = sample['image'], sample['label']
        if self.is_debug == False : 
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)

        filled_labels = np.zeros((self.max_objects, 5), np.float32)
        filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(filled_labels)}
 
    def _resize(self, sample ,intp=cv2.INTER_LINEAR):
        image, label = sample['image'], sample['label']
        image = cv2.resize(image, tuple(self.img_size), interpolation=intp)
        return {'image': image, 'label': label}


    

    def __getitem__(self, index):
        img_line = self.img_files[index % len(self.img_files)].rstrip()
        anno = img_line.strip().split(' ')
        img_path =  anno[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h, ori_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels=np.array([list(map(float, box.split(','))) for box in anno[1:]])
        y = np.zeros_like(labels)
        y[:, 0]= labels[:, -1]
        y[:, 1]= ((labels[:, 0]+labels[:,2])/2)/ori_w
        y[:, 2]= ((labels[:, 1]+labels[:,3])/2)/ori_h
        y[:, 3]= (labels[:, 2]-labels[:,0])/ori_w
        y[:, 4]= (labels[:, 3]-labels[:,1])/ori_h
        labels=y
        sample = {'image': img, 'label': labels}
        sample=self._resize(sample) 
        sample=self._compose(sample) 
        #sample=self._resize(sample) 

        sample["image_path"] = img_path
        sample["origin_size"] = str([ori_w, ori_h])
        return sample

    def __len__(self):
        return len(self.img_files)


