import torch.nn.functional as F 
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import Adam, SGD
import numpy as np
from torch.autograd import Variable

from dataset import getData
from model import NET
from model import focal_loss

import os
import random
import pandas as pd
import pickle

pd.options.display.max_columns=100

'''
作者：中国科学院大学 索玉玺
内容：模型训练
'''

def load_NET(lr):

	net = NET()#.cuda()
	opt = Adam(net.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

	if not os.path.exists('NET'):
		os.mkdir('NET')
		load_state_dict = {'net':net.state_dict(), 'opt':opt.state_dict()}
		torch.save(load_state_dict,'NET/net.pt7')

	else:
		checkpoint = torch.load(os.path.join('NET/net.pt7'))
		net.load_state_dict(checkpoint['net'])

	return net, opt

def save_conv(net, opt):

	load_state_dict = {'net':net.state_dict(), 'opt':opt.state_dict()}
	torch.save(load_state_dict,'NET/net.pt7')

def cntstate(pred, label, state):
	pre_label = pred.argmax(dim = 1)
	
	tp = 0
	fp = 0
	fn = 0
	tn = 0
	for i in range(pred.size()[0]):

		if (pre_label == state) and (label == state):
			tp = tp + 1
		elif (pre_label == state) and (label != state):
			fp = fp + 1
		elif (pre_label != state) and (label != state):
			tn = tn + 1
		elif (pre_label != state) and (label == state):
			fn = fn + 1
	
	return [tp,fp,fn]

def f1_score_l(l):

	P = l[0] / (l[0] + l[1])
	R = l[0] / (l[0] + l[2])

	scores = 2*P*R / (P+R)

	return scores,P,R

def valid(net):

	#读取数据
	with open('valid.txt','r') as f:
		lines = f.readlines()

	with open('attr.txt','r') as f:
		attrs = f.readlines()

	#初始化计数列表
	L0 = [0,0,0]
	L1 = [0,0,0]
	L2 = [0,0,0]
	for i in range(123546):
		data = getData(attrs,lines,i)

		# #处理嵌入向量，仅在使用embedding时使用
		# link_idx_raw = float(data[1])
		# link_idx = link_list.index(link_idx_raw)
		# link_attr1 = [linkattr_embedding_dict.loc[link_idx].values[i] for i in range(2,22)]
		# link_attr2 = [linkattr_embedding_dict.loc[link_idx].values[i] for i in range(47,52)]
		# link_attr = link_attr1 + link_attr2

		current_slice_id = data[2]
		future_slice_id = data[3]        
		index = data[1]
		label = data[0]

		current_seq = data[4]
		history_seq_1 = data[5][0]
		history_seq_2 = data[5][1]
		history_seq_3 = data[5][2]
		history_seq_4 = data[5][3]
		history_seq_5 = data[6][0]
		history_seq_6 = data[6][1]
		history_seq_7 = data[6][2]
		history_seq_8 = data[6][3]
		history_seq_9 = data[6][4]

		current_seq = Variable(torch.tensor(current_seq)).unsqueeze(0)
		history_seq_1 = Variable(torch.tensor(history_seq_1)).unsqueeze(0)
		history_seq_2 = Variable(torch.tensor(history_seq_2)).unsqueeze(0)
		history_seq_3 = Variable(torch.tensor(history_seq_3)).unsqueeze(0)
		history_seq_4 = Variable(torch.tensor(history_seq_4)).unsqueeze(0)
		history_seq_5 = Variable(torch.tensor(history_seq_5)).unsqueeze(0)
		history_seq_6 = Variable(torch.tensor(history_seq_6)).unsqueeze(0)
		history_seq_7 = Variable(torch.tensor(history_seq_7)).unsqueeze(0)
		history_seq_8 = Variable(torch.tensor(history_seq_8)).unsqueeze(0)
		history_seq_9 = Variable(torch.tensor(history_seq_9)).unsqueeze(0)
		# link_attr = Variable(torch.tensor(link_attr)).type(torch.FloatTensor).unsqueeze(0)
       
        #prediction
        ##使用embedding
		#x = net(history_seq_1,history_seq_2,history_seq_3,history_seq_4,history_seq_5,history_seq_6,history_seq_7,history_seq_8,history_seq_9,current_seq,link_attr,batch_size = 1)
		#不使用embedding
		x = net(history_seq_1,history_seq_2,history_seq_3,history_seq_4,history_seq_5,history_seq_6,history_seq_7,history_seq_8,history_seq_9,current_seq,batch_size = 1)
		
		l0 = cntstate(x, label, 0)
		l1 = cntstate(x, label, 1)
		l2 = cntstate(x, label, 2)
		for j in range(3):
			L0[j] = L0[j] + l0[j]
			L1[j] = L1[j] + l1[j]
			L2[j] = L2[j] + l2[j]

	f1_score0,P0,R0 = f1_score_l(L0)
	f1_score1,P1,R1 = f1_score_l(L1)
	f1_score2,P2,R2 = f1_score_l(L2)

	print('======================================')
	print('**************************************')
	print('畅行：')
	print('f1_score: ',f1_score0)
	print('召回率:',R0,'准确率:',P0)
	print('缓行：')
	print('f1_score: ',f1_score1)
	print('召回率:',R1,'准确率:',P1)
	print('拥堵：')
	print('f1_score: ',f1_score2)
	print('召回率:',R2,'准确率:',P2)
	scoreofall = 0.2*f1_score0+0.2*f1_score1+0.6*f1_score2
	print('总分:',scoreofall)
	print('**************************************')
	print('======================================')

def train_NET(epochs,lr):

	net, opt = load_NET(lr)

	for epoch in range(epochs):

		#读取数据
		with open('20190730.txt','r') as f:
			lines = f.readlines()

		with open('attr.txt','r') as f:
			attrs = f.readlines()

		# #打乱数据
		# random.shuffle(lines)

		#初始化所有信息：batch_size input_lists loss_sum loss_epoch
		batch_size = 64
		labels = []
		current_seqs = []
		history_seqs_1 = []	
		history_seqs_2 = []	
		history_seqs_3 = []
		history_seqs_4 = []
		history_seqs_5 = []	
		history_seqs_6 = []	
		history_seqs_7 = []
		history_seqs_8 = []
		history_seqs_9 = []
		# link_attrs = []

		loss_sum = 0.0
		loss_epoch = 0.0
		
		#遍历数据集进行训练
		for i in range(510336):

			data = getData(attrs,lines,i)

			# #处理嵌入向量
			# link_idx_raw = float(data[1])
			# link_idx = link_list.index(link_idx_raw)
			# link_attr1 = [linkattr_embedding_dict.loc[link_idx].values[i] for i in range(2,22)]
			# link_attr2 = [linkattr_embedding_dict.loc[link_idx].values[i] for i in range(47,52)]
			# link_attr = link_attr1 + link_attr2

			current_seq = data[4]
			history_seq_1 = data[5][0]
			history_seq_2 = data[5][1]
			history_seq_3 = data[5][2]
			history_seq_4 = data[5][3]
			history_seq_5 = data[6][0]
			history_seq_6 = data[6][1]
			history_seq_7 = data[6][2]
			history_seq_8 = data[6][3]
			history_seq_9 = data[6][4]
			label = data[0]#已经减去1，即取值范围0，1，2，3，注意标签状态是没有缺省的
			if label == 3:
				label = 2
			
			labels.append(int(label))
			current_seqs.append(current_seq)
			history_seqs_1.append(history_seq_1)
			history_seqs_2.append(history_seq_2)
			history_seqs_3.append(history_seq_3)
			history_seqs_4.append(history_seq_4)
			history_seqs_5.append(history_seq_5)
			history_seqs_6.append(history_seq_6)
			history_seqs_7.append(history_seq_7)
			history_seqs_8.append(history_seq_8)
			history_seqs_9.append(history_seq_9)
			# link_attrs.append(link_attr)
			
			#数据蓄满一个batch，开始前馈计算、反向传播、优化
			if (i+1) % (batch_size) == 0:
				
				current_seqs = Variable(torch.tensor(current_seqs))
				history_seqs_1 = Variable(torch.tensor(history_seqs_1))
				history_seqs_2 = Variable(torch.tensor(history_seqs_2))
				history_seqs_3 = Variable(torch.tensor(history_seqs_3))
				history_seqs_4 = Variable(torch.tensor(history_seqs_4))
				history_seqs_5 = Variable(torch.tensor(history_seqs_5))
				history_seqs_6 = Variable(torch.tensor(history_seqs_6))
				history_seqs_7 = Variable(torch.tensor(history_seqs_7))
				history_seqs_8 = Variable(torch.tensor(history_seqs_8))
				history_seqs_9 = Variable(torch.tensor(history_seqs_9))
				# link_attrs = Variable(torch.tensor(link_attrs)).type(torch.FloatTensor)

				labels = Variable(torch.tensor(labels))
				
				opt.zero_grad()
				# #使用embedding
				# x = net(history_seqs_1,history_seqs_2,history_seqs_3,history_seqs_4,history_seqs_5,history_seqs_6,history_seqs_7,history_seqs_8,history_seqs_9,current_seqs,link_attrs,batch_size = batch_size)
				#使用embedding
				x = net(history_seqs_1,history_seqs_2,history_seqs_3,history_seqs_4,history_seqs_5,history_seqs_6,history_seqs_7,history_seqs_8,history_seqs_9,current_seqs,batch_size = batch_size)
				
				loss = criterion(x,labels) 
				loss.backward()
				opt.step()

				loss_sum = loss_sum + loss.data.item()
				loss_epoch = loss_epoch + loss.data.item()

				#重置input_list
				labels = []
				history_seqs_1 = []
				history_seqs_2 = []	
				history_seqs_3 = []
				history_seqs_4 = []
				history_seqs_5 = []	
				history_seqs_6 = []	
				history_seqs_7 = []
				history_seqs_8 = []
				history_seqs_9 = []
				current_seqs = []
				# link_attrs = []			

				#输出日志
				if (i+1) % (96000) == 0:
					save_conv(net, opt)
					print('训练批次epoch=',epoch+1,'训练条数：',i+1)
					print('loss_sum=',loss_sum)
					loss_sum = 0.0

						
		print('loss of epoch_', epoch+1, ':', loss_epoch)
		print('=======================================================')
		loss_epoch = 0.0
		save_conv(net, opt)

		# #valid
		# print('请等待验证')
		# valid(net)

	save_conv(net, opt)

if __name__ == '__main__':

	# #处理嵌入向量
	# with open('embedding_pretrained/linkattr_embedding_dict_N2.pkl','rb') as f:
	# 	linkattr_embedding_dict = pickle.load(f)

	# linkattr_embedding_dict = linkattr_embedding_dict.reset_index(drop=True)

	# link_list = []
	# for i in range(15370):
	# 	link_list.append(linkattr_embedding_dict.loc[i].values[0])

	#带权交叉熵
	criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1,3,6])).float())
	# #focal loss
	# criterion = focal_loss(num_classes=3, alpha=[0.15,0.25,0.6])

	# train_NET(3,0.01)
	# train_NET(1,0.001)
	print('请等待验证')
	net, opt = load_NET(lr=0.1)
	valid(net)