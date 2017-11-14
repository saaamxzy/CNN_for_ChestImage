from load import *
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch.nn.init as init
from copy import deepcopy

transformations = transforms.Compose([transforms.Scale(256),transforms.ToTensor()])
dset_train = ChestImage(FOLDER_DATASET,transformations)

labelSize = len(dset_train.yt)
valid_size = 0.1
shuffle = False
random_seed = 2
batch_size = 20
num_workers = 1
pin_memory = True
show_sample = True
learning_rate = 0.0005

EPOCH = 5

num_train = len(dset_train)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
if shuffle == True:
	np.random.seed(random_seed)
	np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = DataLoader(dset_train,
	batch_size=batch_size, sampler=train_sampler,
	num_workers=num_workers, pin_memory=pin_memory)
valid_loader = DataLoader(dset_train,
	batch_size=batch_size, sampler=valid_sampler,
	num_workers=num_workers, pin_memory=pin_memory)




# train_loader = DataLoader(dset_train,
# 	                      batch_size=batchSize,
# 	                      shuffle=True,
# 	                      num_workers=1,
# 	                      pin_memory=True)
def validate(val_loader, model, criterion):
    # batch_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    # end = time.time()
    for i, (input, target) in enumerate(val_loader):
    	input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        pred = output.data >= 0.5

        loss = criterion(output, target_var)
        pred = pred.type(torch.cuda.FloatTensor)

        # measure accuracy and record loss
        running_loss += loss.data[0]
        running_corrects += torch.sum(pred == target_var.data)
        #print "accuracy table: "
        
        #print pred, target_var.data

        # prec = accuracy(output.data, target)

        if i % 100 == 0:
            #print 'Test: [{0}/{1}]\t Loss {loss:.4f}\t'.format(i, len(valid_loader), loss=loss)
            print i, len(val_loader), loss

    epoch_loss = running_loss / ( num_train * valid_size )
    epoch_acc = running_corrects / ( num_train * valid_size * 15 )

    if epoch_acc > best_acc:
    	best_acc = epoch_acc
    	best_model = deepcopy(model)

    print "current acc: " + str(epoch_acc)



def train(train_loader, cnn, lossfunc, epoch):
	f = open('out_comparison '+str(epoch) +'.txt', 'w')
	print 'training......'
	cnn.train()
	print sys.getsizeof(enumerate(train_loader))
	for step, (x,y) in enumerate(train_loader):
		# print step
		x, y = x.cuda(async=True), y.cuda(async=True)
		bx = Variable(x,requires_grad=True)
		by = Variable(y)

		out = cnn(bx)
		loss = lossfunc(out,by)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epo, step * len(x), len(train_loader.dataset),
				100. * step / len(train_loader), loss.data[0]))
			
		if step % 100 == 0:

			f.write(str(out.data))
			f.write(str(by.data))
			f.write('-------------------------------------------------------\n')

	f.close()

def accuracy(output, target):
    # TODO: calculate accuracy
    pass

def weights_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		init.xavier_uniform(m.weight.data)
		init.constant(m.bias.data, -0.1)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate"""
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
				out_channels=32,
				kernel_size=5
			),
			nn.BatchNorm2d(32, eps=0.001),
			# 252 x 252 x 32
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
			# 126 x 126 x 32
		)#out: # 126 x 126 x 32
		self.conv2 = nn.Sequential(

			nn.Conv2d(
			in_channels=32,
			out_channels=64,
			kernel_size=5
			),
			nn.BatchNorm2d(64, eps=0.001),
			# 122 x 122 x 64
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
			# 61 x 61 x 64
		)#out: 61 * 61 * 64
		self.conv3 = nn.Sequential(

			nn.Conv2d(
				in_channels=64,
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
		x = F.dropout(x, p=0.4)

		x = x.view(x.size(0),-1)
		output = self.out(x)
		output = F.sigmoid(output)
		return output




cnn = CNN().cuda()
cnn.apply(weights_init)
print cnn
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

lossfunc = nn.BCELoss()

best_model = cnn
best_acc = 0.0




#traing process
for epo in range(EPOCH):

	adjust_learning_rate(optimizer, epo, learning_rate)
	
	train(train_loader, cnn, lossfunc, epo)

	prec1 = validate(valid_loader, cnn, lossfunc)

torch.save(best_model, 'net1.pkl')
torch.save(best_model.state_dict(), 'net1_params.pkl')

print "accuracy: " + str(best_acc)




















