import pickle
import gzip
import pandas as pd

# load and uncompress.
with gzip.open('corpusUs.pickle','rb') as f:
    corpus = pickle.load(f)


with gzip.open('label.pickle','rb') as f:
    label = pickle.load(f)




from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# example data
#data = ["I love machine learning. Its awesome.",
#        "I love coding in python",
#        "I love building chatbots",
#        "they chat amagingly well"]

len(corpus)

data = corpus


list(enumerate(data))


tagged_data = []

for i, _d in enumerate(data):
 tagged_data.append(TaggedDocument(words=_d, tags=[str(i)]))

#tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)] 축약형 표현


max_epochs = 100
vec_size = 300
alpha = 0.025

model = Doc2Vec(window=10,
                vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=2, #출현빈도
                dm=1,
                negative=5,
                seed=9999
                )

model.build_vocab(tagged_data)

#파라메터 설명
#window: 모델 학습할때 앞뒤로 보는 단어의 수
#size: 벡터 차원의 크기
#alpha: learning rate
#min_count: 학습에 사용할 최소 단어 빈도 수
#dm: 학습방법 1 = PV-DM, 0 = PV-DBOW
#negative: Complexity Reduction 방법, negative sampling
#max_epochs: 최대 학습 횟수
#결과 확인
#특정 문서와 유사한 문서를 찾기 위해서는 2단계를 거친다.
#1. 문서의 vector화
#2. 변환된 vector와 가장 가까운 vector 추출
#* infer_vector 사용시 seed값을 주지 않으면 random한값이 seed로 사용되어 값이 계속 변경된다.
#* 학습되지 않은 단어를 사용한 문서도 결과가 나온다.

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")


#ipc에 따른 카테고리 필요 (labeling) - done!
#해당 카테고리에 대해서 Doc2vec 수행 - done!
#Doc2vec 데이터에 대해서 AE 모델 훈련


