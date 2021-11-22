from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# glove_file = datapath('test_glove.txt')
tmp_file = get_tmpfile("/Users/akshaykekuda/Desktop/CSR-SA/word_embeddings/glove.w2v.txt")
# _ = glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)
model.save("/Users/akshaykekuda/Desktop/CSR-SA/word_embeddings/glove_word_vectors")