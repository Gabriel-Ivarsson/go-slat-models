import gensim
import sys
w1 = sys.argv[1]
model = gensim.models.Word2Vec.load("w2v-model.bin")

print("Computing similarity between given words")
print(model.wv.most_similar(w1))
