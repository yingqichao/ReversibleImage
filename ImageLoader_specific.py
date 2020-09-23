# Created on 2020-06-23
# Author: fanghan

import os
import cv2
import numpy as np
import numpy.random as random
import math
from torch.utils.data import Dataset
import scipy.io as scio

class ImageLoader(Dataset):
	def __init__(self,data_dir,w_path,transform=None):
		super(ImageLoader,self).__init__()
		self.data_dir = data_dir
		self.transform = transform
		self.img_paths = os.listdir(data_dir)
		W = scio.loadmat(w_path)
		self.w = W['w']
		
	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self,index):
		curr_img_path = self.img_paths[index]
		nn = curr_img_path.index('.')
		num = int(curr_img_path[:nn])
		img = cv2.imread(self.data_dir + curr_img_path,1)
		m = self.w[num,:,:]
		Data_img = img[:,:,:]

		Data_img = Data_img.transpose((2,0,1))
		Data_img = np.float32(Data_img/255*2-1)
		return  Data_img, m

# train_dataset = ImageLoader(train_path)
# train_load = dataf.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=False, num_workers=0)

