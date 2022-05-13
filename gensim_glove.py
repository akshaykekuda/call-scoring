from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# glove_file = datapath('test_glove.txt')
tmp_file = get_tmpfile("/mnt/transcriber/word_embeddings/vectors_128d.txt")
# _ = glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)
model.save("/mnt/transcriber/word_embeddings/sa_glove_vectors")