import pandas as pd
from torch import np # Torch wrapper for Numpy

import os
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.preprocessing import MultiLabelBinarizer

'''

Disease_map = {'Atelectasis' : 0, 'Cardiomegaly': 1, 'Effusion':2, 'Infiltration':3, 
			'Mass': 4, 'Nodule':5, 'Pneumonia':6, 'Pneumothorax':7, 'Consolidation':8,
			'Edema':9, 'Emphysema': 10, 'Fibrosis':11, 'Pleural_Thickening':12,
			'Hernia':13}

'''



	
FOLDER_DATASET = "/datasets/ChestXray-NIHCC/"
image_path = "/datasets/ChestXray-NIHCC/images/"
#image_path = "/datasets/tmp/gray_scaled_images/"

class ChestImage(Dataset):
	xt = []
	yt = []
	def __init__(self,folder,transform=None):

		reader = pd.read_csv(folder+'Data_Entry_2017.csv')
		self.mlb = MultiLabelBinarizer()
		self.xt = reader['Image Index']

		self.transform = transform
		lb = reader['Finding Labels']
		# self.yt = [[0.00]*14] * len(self.xt)
		# self.yt = np.array(self.yt)
		# self.yt = np.reshape(self.yt,(len(self.yt),14))

		#one hot
		self.yt = self.mlb.fit_transform(lb.str.split('|')).astype(np.float32)
		# print self.yt

		# for j in range(len(lb)):
		# 	l = lb[j].split('|')
		# 	for i in l:
		# 		if i in Disease_map:
		# 			self.yt[j][Disease_map[i]]=1
		

	def __getitem__(self, index):

		#img = torch.load(image_path+self.xt[index])
		img = Image.open(image_path+self.xt[index])
		img = img.convert('RGBA')
		if self.transform is not None:
			img = self.transform(img)
		label = torch.from_numpy(self.yt[index])
		return img, label

	def __len__(self):
		return len(self.xt.index)










