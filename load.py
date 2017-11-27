import pandas as pd
from torch import np # Torch wrapper for Numpy
from sklearn.utils import resample
import os
import PIL
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
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
class ChestImage(Dataset):
	xt = []
	yt = []
	def __init__(self,folder,transform=None):

		reader = pd.read_csv(folder+'Data_Entry_2017.csv')
		self.mlb = MultiLabelBinarizer()

		imgs = reader['Image Index']
		lb = reader['Finding Labels']
		IDX = self.balanceData(imgs,lb)
		self.xt = pd.Series([imgs[i] for i in IDX])
		print type(imgs)
		new_lb = pd.Series(np.array([lb[i] for i in IDX]))
		self.transform = transform
		
		# self.yt = [[0.00]*14] * len(self.xt)
		# self.yt = np.array(self.yt)
		# self.yt = np.reshape(self.yt,(len(self.yt),14))

		#one hot
		self.yt = self.mlb.fit_transform(lb.str.split('|')).astype(np.float32)
		# self.xt = self.xt[0:10]
		# for j in range(len(lb)):
		# 	l = lb[j].split('|')
		# 	for i in l:
		# 		if i in Disease_map:
		# 			self.yt[j][Disease_map[i]]=1
		

	def __getitem__(self, index):
		# print image_path+self.xt[index]
		img = Image.open(image_path+self.xt[index])
		img = img.convert('RGBA')
		# print image_path+self.xt[index]
		if self.transform is not None:
			img = self.transform(img)
		label = torch.from_numpy(self.yt[index])
		return img, label

	def __len__(self):
		return len(self.xt.index)

	def balanceData(self,x,y):
		from count import scan
		from count import upsample
		from count import downsample
		from sklearn.utils import resample

		# dic,avg,maxdis,mindis = scan(y)
		# print dic

		most = ['No Finding']
		most2 = ['Infiltration']
		maj = ['Effusion','Atelectasis']
		med = ['Pneumothorax','Mass','Pleural_Thickening','Consolidation','Nodule']
		mino = ['Edema','Hernia','Pneumonia','Emphysema','Fibrosis','Cardiomegaly']

		rest = [s for s in most+maj+med+mino+most2 if s != 'Hernia']

		zt = [idx  for idx in range(len(y)) if 'No Finding' not in y[idx].split('|')]
		# and 'Infiltration' not in y[idx].split('|')]# and 'Effusion' not in y[idx].split('|')
		# and 'Atelectasis' not in y[idx].split('|')]

		# zt = zt+ downsample(9426.66666667, 10000, most2, maj+med+most, y)
		# zt = zt + upsample(9426.66666667,3000, ['Hernia'], maj+most2, y)
		# zt = zt + upsample(9426.66666667,10000, mino, maj+most2+med, y)
		zt = zt + downsample(9426.66666667, 9426, ['No Finding'],['-'],y)
		# zt = zt + downsample(9426.66666667, 7000, maj,most+most2, y)

		# zt =  zt + upsample(1,3000, 'Pneumonia', y)
		# zt =  zt + upsample(1,3000, 'Fibrosis', y)
		# zt =  zt + upsample(1,3000, 'Edema', y) + upsample(1,3000, 'Emphysema', y)


		#print scan([y[i] for i in zt])
		from random import shuffle
		#print max(zt)
		shuffle(zt)
		return np.array(zt)













