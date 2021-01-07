import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import torch.nn.functional as F 

'''
作者：中国科学院大学 索玉玺
内容：网络构建和focal loss定义
'''

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax
    
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))    #self
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())       #self
        if self.size_average: 
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

#单向lstm网络
class NET(nn.Module):
	
	def __init__(self,batch_size=64):
		super().__init__()

		#处理history信息中以周次为序的序列，输入特征：7，隐藏特征：7，lstm层数：2
		self.lstm_day_1 = nn.LSTM(7, 7, 2)
		self.lstm_day_2 = nn.LSTM(7, 7, 2)
		self.lstm_day_3 = nn.LSTM(7, 7, 2)
		self.lstm_day_4 = nn.LSTM(7, 7, 2)
		self.lstm_day_5 = nn.LSTM(7, 7, 2)

		#处理history信息中以时间片为序的序列，输入特征：7，隐藏特征：7，lstm层数：2
		self.lstm_1 = nn.LSTM(7, 7, 2)
		self.lstm_2 = nn.LSTM(7, 7, 2)
		self.lstm_3 = nn.LSTM(7, 7, 2)
		self.lstm_4 = nn.LSTM(7, 7, 2)

		#处理current信息序列，输入特征：6，隐藏特征：6，lstm层数：2
		self.lstm = nn.LSTM(7, 7, 2)

		# #添加了embedding
		# self.fc = nn.Linear(95,3)
		#未添加embedding
		self.fc = nn.Linear(70,3)

	def forward(self,history_seq_1,history_seq_2,history_seq_3,history_seq_4,history_seq_day_1,history_seq_day_2,history_seq_day_3,history_seq_day_4,history_seq_day_5,current_seq,link_attrs=None,batch_size = 64):

		'''
			history_seq是以时间片为序的序列，每个周次都有一个序列，序列长度是5，一共四个
			history_seq_day是以周次为序的序列，每个时间片都有一个序列，序列长度是4，一共五个
			current_seq是当前5时间片序列
		'''

		h0 = Variable(torch.zeros(2, batch_size, 7))
		c0 = Variable(torch.zeros(2, batch_size, 7))

		#current lstm特征提取，隐藏层数：2，隐藏状态特征数：4
		current_seq = current_seq.permute(1, 0, 2)
		y, (h, c) = self.lstm(current_seq, (h0, c0))
		feature0 = h[1]

		#history lstm提取特征，隐藏层数：2，隐藏状态特征数：7
		history_seq_1 = history_seq_1.permute(1, 0, 2)
		y1, (h1, c1) = self.lstm_1(history_seq_1, (h0, c0))
		feature1 = h1[1]

		history_seq_2 = history_seq_2.permute(1, 0, 2)
		y2, (h2, c2) = self.lstm_2(history_seq_2, (h0, c0))
		feature2 = h2[1]

		history_seq_3 = history_seq_3.permute(1, 0, 2)
		y3, (h3, c3) = self.lstm_3(history_seq_3, (h0, c0))
		feature3 = h3[1]

		history_seq_4 = history_seq_4.permute(1, 0, 2)
		y4, (h4, c4) = self.lstm_4(history_seq_4, (h0, c0))
		feature4 = h4[1]

		# 以周次为序的历史序列的lstm提取
		history_seq_day_1 = history_seq_day_1.permute(1,0,2)
		x1, (hh1, cc1) = self.lstm_day_1(history_seq_day_1, (h0, c0))
		feature5 = hh1[1]

		history_seq_day_2 = history_seq_day_2.permute(1,0,2)
		x2, (hh2, cc2) = self.lstm_day_2(history_seq_day_2, (h0, c0))
		feature6 = hh2[1]

		history_seq_day_3 = history_seq_day_3.permute(1,0,2)
		x3, (hh3, cc3) = self.lstm_day_3(history_seq_day_3, (h0, c0))
		feature7 = hh3[1]

		history_seq_day_4 = history_seq_day_4.permute(1,0,2)
		x4, (hh4, cc4) = self.lstm_day_4(history_seq_day_4, (h0, c0))
		feature8 = hh4[1]

		history_seq_day_5 = history_seq_day_5.permute(1,0,2)
		x5, (hh5, cc5) = self.lstm_day_5(history_seq_day_5, (h0, c0))
		feature9 = hh5[1]

		# #添加了embedding
		# feature = torch.cat((feature0,feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,link_attrs),1)
		#未添加embedding
		feature = torch.cat((feature0,feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9),1)
		return self.fc(F.relu(feature))

if __name__ == '__main__':


	net = NET()
	history_seq_1 = torch.randn(1, 5, 7)
	history_seq_2 = torch.randn(1, 5, 7)
	history_seq_3 = torch.randn(1, 5, 7)
	history_seq_4 = torch.randn(1, 5, 7)
	history_seq_5 = torch.randn(1, 5, 7)
	history_seq_6 = torch.randn(1, 5, 7)
	history_seq_7 = torch.randn(1, 5, 7)
	history_seq_8 = torch.randn(1, 5, 7)
	history_seq_9 = torch.randn(1, 5, 7)


	current_seq = torch.randn(1, 5, 7)

	link_attrs = torch.randn(1, 25)

	x = net(history_seq_1,history_seq_2,history_seq_3,history_seq_4,history_seq_5,history_seq_6,history_seq_7,history_seq_8,history_seq_9,current_seq,batch_size=1)

	
	print(x.size())

