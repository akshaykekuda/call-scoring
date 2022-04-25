import os.path

import spacy

nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.utils.extmath import randomized_svd
import re
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
pt_noun_phrases = "train_data_fbk_vector.p"
import os
import pickle as pickle


def spacy_extract_np(series):
    docs = nlp.pipe(series, batch_size=64)
    m = map(lambda x: [nounphrase.text for nounphrase in x.noun_chunks], docs)
    return [comment_np for comment_np in m]


def load_tscpt_nps(df):
    if os.path.isfile(pt_noun_phrases):
        with open(pt_noun_phrases, 'rb') as file:
            tcpt_noun_phrases = pickle.load(file)
        if len(tcpt_noun_phrases) != len(df):
            "Len of cached nps different from len of df, updating nps"
            tcpt_noun_phrases = update_np_for_tscpt(df, tcpt_noun_phrases)
    else:
        "Did not find cached nps, creating nps"
        tcpt_noun_phrases = get_np_for_tscpt(df)
    return tcpt_noun_phrases


def update_np_for_tscpt(df, tcpt_noun_phrases):
    with nlp.select_pipes(disable=[]):
        for id, text in df.text.iteritems():
            if id not in tcpt_noun_phrases.index:
                sents = sent_tokenize(text)
                docs = nlp.pipe(sents, batch_size=64)
                tcpt_noun_phrases[id] = [np.text.lower() for doc in docs for np in doc.noun_chunks]
    save_pickle(tcpt_noun_phrases)
    return tcpt_noun_phrases


def get_np_for_tscpt(df):
    tcpt_noun_phrases = pd.Series(index=df.index, dtype='object')
    with nlp.select_pipes(disable=[]):
        for id, text in df.text.iteritems():
            sents = sent_tokenize(text)
            docs = nlp.pipe(sents, batch_size=64)
            tcpt_noun_phrases[id] = [np.text.lower() for doc in docs for np in doc.noun_chunks]
    save_pickle(tcpt_noun_phrases)
    return tcpt_noun_phrases


def save_pickle(tcpt_noun_phrases):
    with open(pt_noun_phrases, "wb") as f:
        pickle.dump(tcpt_noun_phrases, f)


class FeedbackComments:
    def __init__(self, score_df, k):
        self.df = score_df.dropna()
        self.vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, stop_words='english')
        self.tcpt_noun_phrases = load_tscpt_nps(self.df)
        self.k = k

    def preprocess_comments(self, scoring_criteria):
        self.df[scoring_criteria] = self.df[scoring_criteria].applymap(
            lambda x: re.sub('[^A-Za-z0-9\.\' ]', '', x.lower()))

    def extract_top_k_comments(self, scoring_criteria):
        feedback_str = [criterion + " Feedback" for criterion in scoring_criteria]
        self.preprocess_comments(feedback_str)
        noun_phrase_str = [criterion + ' nounphrases' for criterion in scoring_criteria]
        self.df[noun_phrase_str] = self.df[feedback_str].apply(lambda x: spacy_extract_np(x))
        comment_data = {}
        for criterion in scoring_criteria:
            comment_data[criterion + " top phrases"] = self.get_tscpt_comment_vectors(criterion)
        return comment_data

    def get_tscpt_comment_vectors(self, criterion):
        top_features = set(self.get_comment_vectors(criterion).index)
        training_data = np.zeros((len(self.df), len(top_features)))
        for idx, nps in enumerate(self.tcpt_noun_phrases):
            feature_mp = Counter(nps)
            feature_vec = [1 if phrase in feature_mp else 0 for phrase in top_features]
            training_data[idx] = feature_vec
        self.df[criterion + " fbk_vector"] = training_data.tolist()
        return top_features

    def get_comment_vectors(self, criterion):
        good_indices = np.where(self.df[criterion] == 1)
        vect_representation = self.vectorizer.fit_transform(self.df[criterion + ' nounphrases'])
        X = vect_representation.todense()
        tfidf = TfidfTransformer()
        Y = tfidf.fit_transform(X[good_indices[0], :]).todense()
        tf_idf_df = pd.DataFrame(Y.T, index=self.vectorizer.get_feature_names_out())
        top_comments = tf_idf_df.mean(axis=1).sort_values(ascending=False)[:self.k]
        print("top noun phrases for {} are {}:".format(criterion, top_comments))
        # feature_indices = [i for i, name in enumerate(self.vectorizer.get_feature_names_out()) if name in top_comments]
        # count_vecs = X[:, feature_indices]
        # count_vecs = 1 * (count_vecs > 0)
        # self.df[criterion + " fbk_vector"] = count_vecs.tolist()
        return top_comments





