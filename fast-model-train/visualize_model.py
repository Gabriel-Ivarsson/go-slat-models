# load model
import sys
from gensim.models.fasttext import load_facebook_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


model_file = sys.argv[1]
model = load_facebook_model(model_file)

# print vocab
words = model.wv.index_to_key

model_word_data=model.wv[model.wv.key_to_index]
df=pd.DataFrame(model_word_data)
df.shape
df.head()
print(df)

# Computing the correlation matrix
X_corr=df.corr()

#Computing eigen values and eigen vectors
values,vectors=np.linalg.eig(X_corr)

#Sorting the eigen vectors coresponding to eigen values in descending order
args = (-values).argsort()
values = vectors[args]
vectors = vectors[:, args]

#Taking first 2 components which explain maximum variance for projecting
new_vectors=vectors[:,:2]

#Projecting it onto new dimesion with 2 axis
neww_X=np.dot(model_word_data,new_vectors)

plt.figure(figsize=(150,150))
plt.scatter(neww_X[:,0],neww_X[:,1],linewidths=10,color='blue')
plt.xlabel("PC1",size=200)
plt.ylabel("PC2",size=200)
plt.title("Word Embedding Space",size=200)

vocab=list(model.wv.key_to_index)
for i, word in enumerate(vocab):
  plt.annotate(word,xy=(neww_X[i,0],neww_X[i,1]))

plt.savefig("plot.png")
