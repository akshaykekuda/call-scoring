import pandas as pd
import os
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from Preprocessing import preprocess_transcript

def TrainWord2Vec(path):
    w2v_list = []
    for file in os.listdir(path):
        if file.endswith('.txt'):
            file_loc = path + '//' + file
            txt = preprocess_transcript(file_loc)
            w2v_list.append(txt)
    train_sentences = [word_tokenize(item) for sublist in w2v_list for item in sublist]
    model = Word2Vec(sentences=train_sentences, min_count = 2,
                 vector_size=100, sg=1,
                 workers=4)
    print(len(model.wv.index_to_key))
    model.save('custom_w2v_100d')

def FastText(path):
    w2v_list = []
    for file in os.listdir(path):
        if file.endswith('.txt'):
            file_loc = path + '//' + file
            txt = preprocess_transcript(file_loc)
            w2v_list.append(txt)
    train_sentences = [word_tokenize(item) for sublist in w2v_list for item in sublist]
    #print(train_sentences)
    vec_size = 100
    window_size = 40
    min_word = 5
    down_sampling = 1e-2
    #vocab_size = len(call_vocab)
    ft_model = FastText(sentences = train_sentences,
                      vector_size=vec_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      sg=1,
                      epochs=100)    
                 
    print(len(ft_model.wv.index_to_key))
    ft_model.save('custom_ft_100d')
