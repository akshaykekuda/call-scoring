import spacy
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.utils.extmath import randomized_svd
import re
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize


def spacy_extract_np(series):
    docs = nlp.pipe(series, batch_size=2000, n_process=4)
    m = map(lambda x: [nounphrase.text for nounphrase in x.noun_chunks], docs)
    return [comment_np for comment_np in m]


class FeedbackComments:
    def __init__(self, score_df):
        self.df = score_df.dropna()
        self.vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, stop_words='english')

    def preprocess_comments(self, scoring_criteria):
        self.df[scoring_criteria] = self.df[scoring_criteria].applymap(lambda x: re.sub('[^A-Za-z0-9\.\' ]', '', x.lower()))

    def extract_top_k_comments(self, k, scoring_criteria):
        feedback_str = [criterion+" Feedback" for criterion in scoring_criteria]
        self.preprocess_comments(feedback_str)
        noun_phrase_str = [criterion + ' nounphrases' for criterion in scoring_criteria]
        self.df[noun_phrase_str] = self.df[feedback_str].apply(lambda x: spacy_extract_np(x))
        comment_data = {}
        for criterion in scoring_criteria:
            comment_data[criterion + " top phrases"] = self.get_comment_vectors(k, criterion)
        return comment_data

    def get_comment_vectors(self, k, criterion):
        good_indices = np.where(self.df[criterion] == 0)
        vect_representation = self.vectorizer.fit_transform(self.df[criterion + ' nounphrases'])
        X = vect_representation.todense()
        tfidf = TfidfTransformer()
        Y = tfidf.fit_transform(X[good_indices[0], :]).todense()
        tf_idf_df = pd.DataFrame(Y.T, index=self.vectorizer.get_feature_names_out())
        top_feature_count = k
        top_comments = tf_idf_df.mean(axis=1).sort_values(ascending=False)[:top_feature_count]
        print("top noun phrases for {} are {}:".format(criterion, top_comments))
        feature_indices = [i for i, name in enumerate(self.vectorizer.get_feature_names_out()) if name in top_comments]
        count_vecs = X[:, feature_indices]
        count_vecs = 1 * (count_vecs > 0)
        self.df[criterion + " fbk_vector"] = count_vecs.tolist()
        return top_comments








