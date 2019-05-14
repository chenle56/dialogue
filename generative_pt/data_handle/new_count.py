import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding

pn=pd.read_table('word_ww.txt',header=None,encoding='gbk')  #读取文件

# print (pd.Series(pn))
# cw = lambda x: list(jieba.cut(x)) #定义分词函数
# pn['words'] = pn[0].apply(cw)

cw = lambda x: list(x.split(" "))
pn['words'] = pn[0].apply(cw)


w = [] #将所有词语整合在一起
for i in pn['words']:
    w.extend(i)


dict = pd.DataFrame(pd.Series(w).value_counts())

count_index=pd.Series(w).value_counts().to_dict()  # {'index': values, 'wdae': 1, ',': 1, '!!': 1, '人土': 1}
a_dict ={}
a_dict=count_index
result = {}
for k, v in a_dict.items():
    result[v]=k
print(result)

dict.to_csv('data11.csv')


dict['id']=list(range(1,len(dict)+1)) #排序词频大小
dict['id'].to_csv('data22.csv')
print(dict['id'])
count_index2=dict['id'].to_dict()
print(count_index2)

get_sent = lambda x: list(dict['id'][x]) #按照词频标号 来标记句子
pn['sent'] = pn['words'].apply(get_sent) #速度太慢
pn['sent'].to_csv('data33.csv')


# maxlen = 50

# print("Pad sequences (samples x time)")
# pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
# print(pn['sent'])
