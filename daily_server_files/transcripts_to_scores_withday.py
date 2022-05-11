import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')

stemmer = nltk.stem.porter.PorterStemmer()
import os
from numpy.linalg import norm
import numpy as np
import random
import wave
import contextlib

import csv
from gensim.test.utils import get_tmpfile
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

from gensim.models.doc2vec import Doc2Vec


def make_scoring_corpus(corpus, m):
    """ Takes the input corpus and returns the Doc2Vec representation of the
    inputs."""
    a = []
    for file in corpus:
        file = file.split()
        a = np.append(a,m.infer_vector(file))       
    a = a.reshape(len(corpus),200)
    return a

def file_path_to_corpus(file_path, subset_files = []):
    """ Given a file path, returns a numpy tensor of all the text files
    as strings. Optional parameters is a list of files to be included, while
    ignoring others in the directory that are not in the list. Also returns
    an array of all the files included in the corpus."""
    if len(subset_files) == 0:
        corpus = np.array([], dtype=str)
        files = [f for f in os.listdir(file_path) if ".txt" in f]
        for f in files:
            with open (file_path + f, "r") as myfile:
                corpus = np.append(corpus, myfile.read())
    else:
        corpus = np.array([], dtype=str)
        files = [f for f in os.listdir(file_path) if ".txt" in f and f in subset_files]
        for f in files:
            with open (file_path + f, "r") as myfile:
                corpus = np.append(corpus, myfile.read())
        
    return files, corpus

def remove_strings(corpus):
    """ Cleans the corpus by removing silence tokens and newlines."""
    for cnt, doc in enumerate(corpus):
        doc = doc.replace("<#s>", "")
        doc = doc.replace("\n", "")
        corpus[cnt] = doc
    return corpus

def get_similarity(above_85, below_60, file, files,m):
    """ Returns whether a call is closer to a bad call (1) or good call (0).
        pos_sums and neg_sums are used if we want to determine the 5
        closest calls for that classification."""
    # Sums both start at 0, and initialize vectors
    pos_sum = 0
    neg_sum = 0
    pos_sums = []
    neg_sums = []
    print(len(above_85), len(below_60))
    # File -> Doc2Vec
    file = m.infer_vector(file.split())
    
    # Loop through positive calls
    for cnt, x in enumerate(above_85):
        pos = np.dot(file,x) / (norm(file) * norm(x))
        pos_sum += pos
        pos_sums.append((cnt,pos))
        
    # Loop through negative calls
    for cnt, x in enumerate(below_60):
        neg = np.dot(file,x) / (norm(file) * norm(x))
        neg_sum += neg
        neg_sums.append((cnt,neg))
        
    # Find top 5 in each category
    pos_sums.sort(key=lambda x: x[1],reverse=True)
    neg_sums.sort(key=lambda x: x[1],reverse=True)
    pos_sums = pos_sums[:5]
    neg_sums = neg_sums[:5]
    
    # return
    if pos_sum > neg_sum:
        return pos_sums, 1- pos_sum/len(above_85)
        #return pos_sums, 0
    else:
        return neg_sums, neg_sum/len(below_60)
        #return neg_sums, 1
    
def normalize(text):
    return nltk.word_tokenize(text.lower())

def quote_score(call, stopwords):
    """ Does bag of words cosine distance between quote questions and call."""
    quote_qs = "have all residents of your household age sixteen or older been disclosed on this application have all drivers such as children away from home or in college or anyone who may operate your vehicle on a regular or occasional basis been listed on this application are all vehicles in the household listed in the application are any vehicles used during the course of business employment are any vehicles used for delivery purposes or for any commercial purpose has any driver ever suffered from blackouts seizures epilepsy diabetes or any other physical impairments does any driver take regularly prescribed medicine has any driver been involved in an accident or reported a claim to an insurance company in the past five years is there any existing damage or broken glass to the vehicle listed on this application does any driver drive out of state on regular basis are any vehicles used during the course of an insured persons employment to transport people"

    call = ' '.join([word for word in call.split() if word not in stopwords])
    corpus = [call]
    sim = 0
    quote_qs = ' '.join([word for word in quote_qs.split() if word not in stopwords])
    cv = CountVectorizer()
    corpus.append(quote_qs)
    counts = cv.fit_transform(corpus)
    a = counts[0].toarray()
    b = counts[1].toarray()
    a = np.array(a[0])
    b = np.array(b[0])
    if norm(a)*norm(b) < .0001:
        return 0
    sim += np.dot(a,b) / (norm(a)* norm(b))
    corpus = [call]
    return sim

def get_length(file_name, path):
    #message = "c*" + file[12:-4] + ".wav"
    #path = "/dds_share/Genesys/manual/"
    #wav_path = !find {path} -name {message}
    #wav_path = wav_path[0]
    wav_path = path+file_name[:-4] + ".wav"
    with contextlib.closing(wave.open(wav_path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration
    
def get_first_word(file, path):
    #message = "a*" + file[12:-4] + ".ctm"
    #path = "/dds_share/GenesysTranscriptions"
    #path = "/media/data/mcmurw/tools/eesen-offline-transcriber"
    #ctm_path = !find {path} -name {message}
    #ctm_path = ctm_path[0]
    ctm_path = path+file
    with open(ctm_path, "r") as f:
        return f.readline().split(" ")[2]
    
    
def pick_fifty(corpus, files):
    test = np.array([])
    for x in range(50):
        r = random.randint(0,len(corpus)-1)
        test = np.append(test,corpus[r])
        corpus = np.delete(corpus,r)
        files = np.delete(files,r)
    return corpus, files, test
    
if __name__ == "__main__":

    path_to_wavs = sys.argv[1]
    path_to_ctms = sys.argv[2]
    path_to_docvec_model = sys.argv[3]
    path_to_negative_transcripts = sys.argv[4]
    print(path_to_negative_transcripts)
    path_to_positive_transcripts = sys.argv[5]
    path_to_candidate_transcripts = sys.argv[2]
    path_to_output = sys.argv[6] 
    stopwords = stopwords.words('english')
    
    m = Doc2Vec.load(path_to_docvec_model)
    
    below_60_files, below_60_corpus = file_path_to_corpus(path_to_negative_transcripts)
    above_85_files, above_85_corpus = file_path_to_corpus(path_to_positive_transcripts)
    below_60_corpus = remove_strings(below_60_corpus)
    above_85_corpus = remove_strings(above_85_corpus)
    above_85_corpus, above_85_files, above_85_test = pick_fifty(above_85_corpus, above_85_files)
    above_85_test = make_scoring_corpus(above_85_test, m)
    below_60_corpus, below_60_files, below_60_test = pick_fifty(below_60_corpus, below_60_files)
    below_60_test = make_scoring_corpus(below_60_test, m)
    
    files_val, corpus_val = file_path_to_corpus(path_to_candidate_transcripts)
    corpus_val = remove_strings(corpus_val)
    
    data = []
    for cnt, x in enumerate(files_val):
        _, l = get_similarity(above_85_test, below_60_test, corpus_val[cnt], files_val,m)
        data.append([x[0:-4],l,\
                     get_length(x, path_to_wavs), quote_score(corpus_val[cnt], stopwords)])
    with open(path_to_output, "w+") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(data)

