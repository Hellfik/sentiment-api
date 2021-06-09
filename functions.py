from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def tfidf_vectorizer(data, MAX_NB_WORDS = 1000):
    tfv = TfidfVectorizer(max_features=MAX_NB_WORDS, 
            strip_accents='unicode', analyzer='word',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1)
    emb = tfv.fit_transform(data).toarray()

    print("Tfidf vectorize with", str(np.array(emb).shape[1]), "features")
    return emb, tfv

def cv(data, ngram = 1, MAX_NB_WORDS = 1000):
    count_vectorizer = CountVectorizer(ngram_range = (ngram, ngram), max_features = MAX_NB_WORDS)
    emb = count_vectorizer.fit_transform(data).toarray()

    return emb, count_vectorizer
