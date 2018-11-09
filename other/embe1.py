from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

inp = "../data/novel/data.src"
outp = "../resource/novel_embedding_64.txt"
model = Word2Vec(LineSentence(inp), size=64, window=5, min_count=1)
model.wv.save_word2vec_format(outp,binary=False)