#load data
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import pickle
import gzip
import pandas as pd

# load and uncompress.
with gzip.open('label.pickle','rb') as f:
    label = pickle.load(f)


#https://wjddyd66.github.io/tensorflow/Tensorflow-AutoEncoder/#stacked-autoencoder%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EB%B9%84%EC%A7%80%EB%8F%84-%EC%82%AC%EC%A0%84%ED%95%99%EC%8A%B5%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5
#https://deepinsight.tistory.com/126

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("I love chatbots".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)


# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('103241')
print(similar_doc)

model.corpus_count
model.corpus_total_words
model.wv.vectors.shape
model.wv.vocab
len(model.docvecs)


data = []
for i in range(0, len(model.docvecs)):
    print('iteration {0}'.format(i))
    data.append(model.docvecs[i])


import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
# split the data into a training set and a validation set
random.seed(9999)
VALIDATION_SPLIT = 0.2

data = np.array(data)
min_max_scaler = MinMaxScaler()
data = min_max_scaler.fit_transform(data)
# min_max_scaler.transform(newData)
# min_max_scaler.inverse_transform(data)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

label = np.array(label)

x_train = data[:-num_validation_samples]
x_val = data[-num_validation_samples:]

lab_train = label[:-num_validation_samples]
lab_val = label[-num_validation_samples:]

len(x_train)
len(lab_train)
len(x_val)
len(lab_val)


import tensorflow as tf
from keras.models import model_from_json
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# aeReLU
# 모델 불러오기
json_file = open("autoencoderReLU.json", "r")
loaded_model_json = json_file.read()
json_file.close()
autoencoder = model_from_json(loaded_model_json)

# 모델 가중치 불러오기
autoencoder.load_weights("autoencoderReLU.h5")
print("Loaded model from disk")

# 모델을 사용할 때는 반드시 컴파일을 다시 해줘야 한다.
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
score = autoencoder.evaluate(x_val, x_val,verbose=0)


# extract encoder network from autoencoder model
from keras.models import Model
from keras.layers import Input

encoder = Model(autoencoder.layers[0].input, autoencoder.layers[9].output)
encoder.summary()


# extract decoder network from autoencoder model
encoding_dim = 10
encoded_input = Input(shape=(encoding_dim,))

deco = autoencoder.layers[-9](encoded_input)
deco = autoencoder.layers[-8](deco)
deco = autoencoder.layers[-7](deco)
deco = autoencoder.layers[-6](deco)
deco = autoencoder.layers[-5](deco)
deco = autoencoder.layers[-4](deco)
deco = autoencoder.layers[-3](deco)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)

# create the decoder model
decoder = Model(encoded_input, deco)
decoder.summary()


# dimension reduction
x_valEncoded = encoder.predict(x_val)
x_val.shape
x_valEncoded.shape


# Threshold 구하기
import numpy as np


data = x_valEncoded

def sqrt(inp):
    result = inp/2
    for i in range(30):
        try:
            result = (result + (inp / result)) / 2
        except:
            result = 0
    return result



def euclidean(inp):
    result = []
    len_inp = len(inp)
    for i in inp:
        for j in inp:
            tmp = sqrt((i[0] - j[0])**2 + (i[1] - j[1])**2)
            result.append(tmp)
    result = np.array(result)
    result = np.reshape(result, (len_inp, len_inp))
    result = pd.DataFrame(result)
    return result

result = euclidean(data[:100])


resultTrl = np.tril(result, k=-1)
resultTrl = resultTrl.flatten()
resultTrlPlus = resultTrl[resultTrl > 0]

Threshold = list(reversed(resultTrlPlus))[int(len(resultTrlPlus) * 0.999)]


import umap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#tmp = np.random.rand(4096, 10)
data = x_valEncoded

fit = umap.UMAP(n_neighbors=15) # neighbors number = 라벨 개수
u = fit.fit_transform(data)
plt.scatter(u[:, 0], u[:, 1]) #, c=data
plt.title('UMAP dimension reduction result')
# https://umap-learn.readthedocs.io/en/latest/parameters.html


import pandas as pd
from sklearn.cluster import DBSCAN

dbs = DBSCAN(eps=Threshold, min_samples=Threshold) # Threshold of PPDM
predict = pd.DataFrame(dbs.fit_predict(u)) # chain with umap
predict.columns = ['predict']

r = pd.concat([pd.DataFrame(data), predict], axis=1)
r['predict'].describe()
pd.Series.unique(r['predict'])
r['predict'].value_counts()

plt.scatter(u[:, 0], u[:, 1], c=r['predict'])


def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)


r['predict'].value_counts()
len(which(r['predict']==0))
pd.DataFrame(lab_val[which(r['predict']==0)]).value_counts()
pd.DataFrame(lab_val[which(r['predict']==1)]).value_counts()
pd.DataFrame(lab_val[which(r['predict']==2)]).value_counts()
pd.DataFrame(lab_val[which(r['predict']==3)]).value_counts()
pd.DataFrame(lab_val[which(r['predict']==4)]).value_counts()
pd.DataFrame(lab_val[which(r['predict']==5)]).value_counts()

# 이상검출 알고리즘 -> 성향점수매칭
# https://medium.com/@bmiroglio/introducing-the-pymatch-package-6a8c020e2009

# 앙상블 알고리즘 -> voting
# https://nonmeyet.tistory.com/entry/Python-Voting-Classifiers%EB%8B%A4%EC%88%98%EA%B2%B0-%EB%B6%84%EB%A5%98%EC%9D%98-%EC%A0%95%EC%9D%98%EC%99%80-%EA%B5%AC%ED%98%84
