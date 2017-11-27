from sklearn.utils import resample

import numpy as np
import pandas as pd
from torch import np # Torch wrapper for Numpy
# {'No Finding': 60412, 'Pneumothorax': 5298, 'Effusion': 13307, 'Cardiomegaly': 2772, 
# ''Pleural_Thickening'': 3385, 'Atelectasis': 11535, 'Consolidation': 4667, 'Edema': 2303, 
# 'Emphysema': 2516, 'Pneumonia': 1353, 'Nodule': 6323, 'Mass': 5746, 'Infiltration': 19870, 
# 'Hernia': 227, 'a': 1686}, 9426.666666666666, 'No Finding', 'Hernia'

def downsample(avg,down_size,down_class, boundary, data):

	majority = [idx  for idx in range(len(data)) if  any(i in down_class for i in list(data[idx].split('|'))) 
	and not any(i in list(data[idx].split('|')) for i in boundary)]
	downsampled = resample(majority,replace=False,n_samples=down_size)
	res = list(downsampled)
	return res

def upsample(vg,up_size,upclass,boundary,data):

	minority = [idx for idx in range(len(data)) if any(i in upclass for i in list(data[idx].split('|')))
	and not any(i in list(data[idx].split('|')) for i in boundary)]
	upsampled = resample(minority,replace=True,n_samples=up_size)
	return list(upsampled)

def scan(lb):
	dic = {}
	for d in range(len(lb)):
		arr = lb[d].split('|')
		for a in arr:
			if a in dic:
				dic[a] += 1
			else:
				dic[a] = 1
	sums = 0.00
	maxDis = ''
	maxNum = 0
	minDis = ''
	minNum = len(lb)
	for k in dic.keys():
		sums+=dic[k]
		if dic[k] > maxNum:
			maxDis = k
			maxNum = dic[k]
		if dic[k] < minNum:
			minDis = k
			minNum = dic[k]

	# print dic, sums/len(dic), maxDis, minDis
	return dic,sums/len(dic), maxDis, minDis

# yt = [idx  for idx in range(len(lb)) if 'No Finding' not in lb[idx].split('|') and 'Infiltration' not in lb[idx].split('|')]

# dwn = downsample(9426.66666667, 10000, 'No Finding',lb)
# dwn2 = downsample(9426.66666667, 11000, 'Infiltration',lb)

# up = upsample(1,3000, 'Hernia', lb)
# zt = yt + dwn + dwn2 + up
# new_lb = [lb[i] for i in zt]

# print scan(new_lb)

