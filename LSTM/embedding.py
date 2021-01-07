import pandas as pd
import pickle
# from tqdm import tqdm
pd.options.display.max_columns=100


'''
作者：中国科学院大学 索玉玺
内容：这个文件就是测试一下pkl的使用，没有别的作用
'''


with open('embedding_pretrained/linkattr_embedding_dict_N2.pkl','rb') as f:
	linkattr_embedding_dict = pickle.load(f)

linkattr_embedding_dict = linkattr_embedding_dict.reset_index(drop=True)

link_list = []
for i in range(15370):
	link_list.append(linkattr_embedding_dict.loc[i].values[0])

print(link_list[15])

link_idx = link_list.index(3869)

link_attr = [linkattr_embedding_dict.loc[link_idx].values[i] for i in range(2,22)] + [linkattr_embedding_dict.loc[link_idx].values[i] for i in range(47,52)]
print(link_attr)
print(len(link_attr))

# index = []

# # 前20是link_id，后面40是8个道路属性，每个属性用5位表示, 按需求取，可以只要link_id这个embedding
# linkattr_embedding_dict.head()


# # 将linkattr_embedding_dict按link拼接到数据中，然后留下的是link_lbe，用来做embedding字典的索引
# data = pd.merge(data, linkattr_embedding_dict[['link','link_lbe']], on='link')

# # 获取embedding字典，现在data里面有link_lbe作为索引，而这个embedding_dict又是按索引排列的

# embedding_dict = linkattr_embedding_dict.loc[:,linkattr_embedding_dict.columns!=['link','link_lbe']].values
# embedding_dict = torch.tensor(embedding_dict, dtype=torch.float32)

# class model(nn.Module):

# 	def init(self, embedding_pretrained_dict=embedding_dict):
#         # 实例化
# 		self.embed = nn.Embedding.from_pretrained(embedding_dict, freeze=False)
    
# 	def forward(data):
#         # 以link_lbe来获取embedding_dict里对应id的embedding
# 		linkattr_embedding = self.embed(data[:,'link_lbe'].long())
        
        
#         # 拼接 进入全连接
# 		torch.cat([linkattr_embedding, time_correlation], dim=-1)

# if __name__ == '__main__':
# 	model = model()
# 	print(model(0))