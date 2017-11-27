from load import *
import sys
import matplotlib
matplotlib.use('Agg')
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn.init as init
from copy import deepcopy
from collections import defaultdict


transformations = transforms.Compose([transforms.Scale(256),transforms.ToTensor()])

print "loading data..."
dset_train = ChestImage(FOLDER_DATASET,transformations)
print "done."


labelSize = len(dset_train.yt)
valid_size = 0.1
shuffle = True
random_seed = 2
batch_size = 10
num_workers = 1
pin_memory = True
show_sample = True
learning_rate = 0.0005
best_acc = 0.0
threshold_list = [x * 0.1 for x in range(11)]

pred_threshold = torch.cuda.FloatTensor([
	[
	0.08, 0.08, 0.05,
	0.05, 0.11, 0.03,
	0.03, 0.02, 0.10,
	0.05, 0.45, 0.08,
	0.03, 0.05, 0.05
	] for i in range(batch_size)]
	)

EPOCH = 3

num_data = len(dset_train)
#num_data = 1000
print num_data
num_train = num_data - 5000
indices = list(range(num_train))
print 'num_train: ' + str(num_train)

split = int(np.floor(valid_size * num_train))
print 'split: ' + str(split)
test_idx = list(range(num_train, num_data))
if shuffle == True:
	np.random.seed(random_seed)
	np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]

print 'length of train_idx: ' + str(len(train_idx))

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = DataLoader(dset_train,
	batch_size=batch_size, sampler=train_sampler,
	num_workers=num_workers, pin_memory=pin_memory)

valid_loader = DataLoader(dset_train,
	batch_size=batch_size, sampler=valid_sampler,
	num_workers=num_workers, pin_memory=pin_memory)

test_loader = DataLoader(dset_train,
	batch_size=batch_size, sampler=test_sampler,
	num_workers=num_workers, pin_memory=pin_memory)

print 'length:\ntrain_loader: ' + str(len(train_loader))
print 'valid_loader: ' + str(len(valid_loader))
print 'test_loader: ' + str(len(test_loader))

#test_loader = Data

def test(loader, model, criterion, mode='test'):
	print "size of test loader: " + str(len(loader))
	if mode == 'test':
		out_file = open('test_loss_recall.txt', 'w')
	else:
		out_file = open('train_test_loss_recall.txt', 'w')
	out_file.write('loss \t recall\n')


	model.eval()

	running_loss = 0.0
	running_corrects = 0
	num_of_positive = 0
	# end = time.time()
	total_number_of_labels = 0.0
	running_tps = [0.0 for x in range(len(threshold_list))]
	num_of_pred_pos = [0.0 for x in range(len(threshold_list))]
	recall_record = [0.0 for x in range(len(threshold_list))]
	precision_record = [0.0 for x in range(len(threshold_list))]

	for i, (input, target) in enumerate(loader):
		# for a batch of batch_size
		input = input.cuda(async=True)
		target = target.cuda(async=True)
		input_var = Variable(input, volatile=True)
		target_var = Variable(target, volatile=True)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		num_of_positive += torch.sum(target_var.data)
		
		for t in range(len(threshold_list)):
			pred = output.data >= threshold_list[t]
			pred = pred.type(torch.cuda.FloatTensor)
			#rs_out, rs_tar = count_category_pos(pred, target_var.data, rs_out, rs_tar)
			num_true_pos = count_true_positive(pred, target_var.data)
			num_true_neg = count_true_negative(pred, target_var.data)
			num_of_pred_pos[t] += torch.sum(pred)


			running_tps[t] += num_true_pos
			if num_of_pred_pos[t] != 0:
				precision_record[t] = running_tps[t] / num_of_pred_pos[t]
			else:
				precision_record[t] = 0.0
			recall_record[t] = (running_tps[t] / num_of_positive)
		
		# measure accuracy and record loss
		running_loss += loss.data[0]
		running_corrects += torch.sum(pred == target_var.data)
		for each in target_var.data:
			total_number_of_labels += len(each)


		# print 'true pos: ' + str(num_true_pos)
		# print 'true neg: ' + str(num_true_neg)

		#running_true_positive += num_true_pos
		#running_recall = running_true_positive / num_of_positive  # need to be 1s only

		if i % 100 == 0:
			#print 'Test: [{0}/{1}]\t Loss {loss:.4f}\t'.format(i, len(valid_loader), loss=loss)
			print 'loss:', i, len(loader), loss
			write_string = str(loss.data[0]) + '\n'
			out_file.write(write_string)

	epoch_loss = running_loss / ( num_train * valid_size )
	#epoch_acc = running_corrects / ( num_of_pred_pos )
	#recall = running_true_positive / num_of_positive
	out_file.write('recall\n')
	for i, each in enumerate(recall_record):
		print 'recall for threshold of value {}: {}'.format(threshold_list[i], each)
		out_file.write(str(i) + ', ' + str(each)+ '\n')
	out_file.write('precision\n')

	for i, each in enumerate(precision_record):
		print 'precision for threshold of value {}: {}'.format(threshold_list[i], each)
		out_file.write(str(i) + ', ' + str(each)+ '\n')



	epoch_recall = max(recall_record)
	epoch_recall_idx = recall_record.index(epoch_recall)

	#print "current acc: " + str(epoch_acc)
	print 'current recall: ' + str(epoch_recall) + ' with threshold ' + str(threshold_list[epoch_recall_idx])
	out_file.close()

def validate(val_loader, model, criterion, best_acc, best_model, epoch):
	out_file = open('valid_loss_recall_'+str(epoch)+'.txt', 'w')

	# switch to evaluate mode
	model.eval()

	running_loss = 0.0
	running_corrects = 0
	num_of_positive = 0
	# end = time.time()
	total_number_of_labels = 0.0
	running_tps = [0.0 for x in range(len(threshold_list))]
	num_of_pred_pos = [0.0 for x in range(len(threshold_list))]
	recall_record = [0.0 for x in range(len(threshold_list))]
	precision_record = [0.0 for x in range(len(threshold_list))]

	for i, (input, target) in enumerate(val_loader):
		# for a batch of batch_size
		input = input.cuda(async=True)
		target = target.cuda(async=True)
		input_var = Variable(input, volatile=True)
		target_var = Variable(target, volatile=True)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		num_of_positive += torch.sum(target_var.data)
		
		for t in range(len(threshold_list)):
			pred = output.data >= threshold_list[t]
			pred = pred.type(torch.cuda.FloatTensor)
			#rs_out, rs_tar = count_category_pos(pred, target_var.data, rs_out, rs_tar)
			num_true_pos = count_true_positive(pred, target_var.data)
			num_true_neg = count_true_negative(pred, target_var.data)
			num_of_pred_pos[t] += torch.sum(pred)


			running_tps[t] += num_true_pos
			if num_of_pred_pos[t] != 0:
				precision_record[t] = running_tps[t] / num_of_pred_pos[t]
			else:
				precision_record[t] = 0.0
			recall_record[t] = (running_tps[t] / num_of_positive)
		
		# measure accuracy and record loss
		running_loss += loss.data[0]
		running_corrects += torch.sum(pred == target_var.data)
		for each in target_var.data:
			total_number_of_labels += len(each)


		# print 'true pos: ' + str(num_true_pos)
		# print 'true neg: ' + str(num_true_neg)

		#running_true_positive += num_true_pos
		#running_recall = running_true_positive / num_of_positive  # need to be 1s only

		if i % 100 == 0:
			#print 'Test: [{0}/{1}]\t Loss {loss:.4f}\t'.format(i, len(valid_loader), loss=loss)
			print 'loss:', i, len(val_loader), loss
			write_string = str(loss.data[0]) + '\n'
			out_file.write(write_string)

	epoch_loss = running_loss / ( num_train * valid_size )
	#epoch_acc = running_corrects / ( num_of_pred_pos )
	#recall = running_true_positive / num_of_positive

	for i, each in enumerate(recall_record):
		print 'recall for threshold of value {}: {}'.format(threshold_list[i], each)
		out_file.write(str(i) + ', ' + str(each)+ '\n')

	for i, each in enumerate(precision_record):
		print 'precision for threshold of value {}: {}'.format(threshold_list[i], each)
		out_file.write(str(i) + ', ' + str(each)+ '\n')



	epoch_recall = max(recall_record)
	epoch_recall_idx = recall_record.index(epoch_recall)
	if epoch_recall > best_acc:
		best_acc = epoch_recall
		best_model = deepcopy(model)


	#print "current acc: " + str(epoch_acc)
	print 'current recall: ' + str(epoch_recall) + ' with threshold ' + str(threshold_list[epoch_recall_idx])
	out_file.close()

	return best_acc, best_model


def train(train_loader, cnn, lossfunc, epoch):
	# f = open('out_comparison '+str(epoch) +'.txt', 'w')
	out_file = open('train_loss_'+str(epoch)+'.txt', 'w')

	print 'training......'
	cnn.train()
	print 'len of train_loader: ' + str(len(train_loader))

	running_corrects = 0.0
	running_true_positive = 0.0
	running_all_positive = 0.0
	threshold_list = [0.1 * x for x in range(11)]
	for step, (x,y) in enumerate(train_loader):
		# print step
		x, y = x.cuda(async=True), y.cuda(async=True)
		bx = Variable(x,requires_grad=True)
		by = Variable(y)

		out = cnn(bx)

		#print pred_threshold
		
		# pred_list = []
		# for t in threshold_list:
		# 	pred = out.data >= t
		# 	pred = pred.type(torch.cuda.FloatTensor)
		# 	pred_list.append()

		

		#running_true_positive += count_true_positive(pred, by.data)
		#running_all_positive += torch.sum(by.data)

		#running_recall = running_true_positive / running_all_positive
		#print 'running recall: ' + str(running_recall)

		loss = lossfunc(out,by)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epo, step * len(x), num_train - (num_train * valid_size),
				100. * step*10 / (num_train - (num_train * valid_size) ), loss.data[0]))

			out_file.write(str(loss.data[0]) + '\n')

			#print 'output from net: ' + str(out)
			
		# if step % 100 == 0:
			
			# f.write(str(out.data))
			# f.write(str(by.data))
			# f.write('-----------------------------------\n')

	# f.close()
	out_file.close()


def count_true_positive(output, target):
	rs = 0
	# print 'dim1: ' + str(len(output))
	# print 'dim2: ' + str(len(output[0]))
	for i in range(len(output)):
		for j in range(len(output[i])):
			if output[i][j] == target[i][j] == 1:
				rs += 1
	return rs

def count_true_negative(output, target):
	rs = 0
	for i in range(len(output)):
		for j in range(len(output[i])):
			if output[i][j] == target[i][j] == 0:
				rs += 1
	return rs

def count_category_pos(output, target, rs_out, rs_tar):
	for i in range(len(output)):
		for j in range(len(output[i])):
			print 'the {}th element in the {}th sample: output[{}], target[{}]'.format(j, i, output[i][j], target[i][j])
			if output[i][j] == target[i][j] == 1.0:
				rs_out[j] += 1.0
			if target[i][j] == 1.0:
				rs_tar[j] += 1.0
	return rs_out, rs_tar

def weights_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		init.xavier_uniform(m.weight.data)
		init.constant(m.bias.data, -0.1)

def adjust_learning_rate(optimizer, epoch, lr):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = lr * (0.1 ** (epoch / 1.5))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		# 256 x 256 x 4
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=4,
				out_channels=16,
				kernel_size=5
			),
			nn.BatchNorm2d(16, eps=0.001),
			# 256 x 256 x 32
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
			# 128 x 128 x 32
		)#out: # 128 x 128 x 32
		self.conv2 = nn.Sequential(

			nn.Conv2d(
			in_channels=16,
			out_channels=32,
			kernel_size=5
			),
			nn.BatchNorm2d(32, eps=0.001),
			# 128 x 128 x 64
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
			# 61 x 61 x 64
		)#out: 61 * 61 * 64
		self.conv3 = nn.Sequential(

			nn.Conv2d(
				in_channels=32,
				out_channels=64,
				kernel_size=5
			),
			nn.BatchNorm2d(64, eps=0.001),
			# 57 x 57 x 64
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
			# 28 x 28 x 64
		)#out: 28 x 28 x 64

		self.conv4 = nn.Sequential(
			# 28 x 28 x 64
			nn.Conv2d(
				in_channels=64,
				out_channels=128,
				kernel_size=5
			),
			nn.BatchNorm2d(128, eps=0.001),
			# 24 x 24 x 128
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
			# 12 x 12 x 128
		)#out: 12 x 12 x 128
		self.out = nn.Linear(12*12*128,15)



	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = F.dropout(x, p=0.2)

		x = x.view(x.size(0),-1)
		output = self.out(x)
		output = F.sigmoid(output)
		return output




cnn = CNN().cuda()
cnn.apply(weights_init)
print cnn
optimizer = torch.optim.Adam(cnn.parameters(), learning_rate)

lossfunc = nn.BCELoss()
best_model = cnn


#traing process
for epo in range(EPOCH):

	adjust_learning_rate(optimizer, epo, learning_rate)
	
	train(train_loader, cnn, lossfunc, epo)

	torch.save(best_model, 'net1.pkl')
	torch.save(best_model.state_dict(), 'net1_params.pkl')

	best_acc, best_model = validate(valid_loader, cnn, lossfunc, best_acc, cnn, epo)

test(train_loader, best_model, lossfunc, 'train')

test(test_loader, best_model, lossfunc)






















