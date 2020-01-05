from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import os
import numpy as np
from nltk.corpus import stopwords


class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim=0, word_emb_folder=""):
        self.dim = dim
        self.word_emb_folder = word_emb_folder
        self.word2weight = None
        # we need to remove all the
        self.tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w+', stop_words=stopwords.words('english'))

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        # if during feature inference, word which was not in the training vocabulary - it must be at least as infrequent as any of the known words - so the default idf is the max of known idf's
        self.max_idf = max(self.tfidf.idf_)
        self.word2weight = {word:self.tfidf.idf_[ind] for word, ind in self.tfidf.vocabulary_.items()}

        if self.dim != 0:
            self.word2vec = {}
            with open(os.path.join(self.word_emb_folder, "glove.6B.%dd.txt" % self.dim), "rb") as word_embedding:
                for line in word_embedding:
                    self.word2vec[line.strip().split()[0]] = np.array(map(float, line.strip().split()[1:]))

        return self

    def transform(self, X):
        vec = []
        if self.dim != 0:
            X = np.array(X)
            for row in X:
                print(self.word2vec.get(row.split()[0]).shape)
                vec.append(np.mean(np.array(self.word2vec.get(word, np.ones(self.dim)) * self.word2weight.get(word, self.max_idf) for word in row.split()), axis=0))
            return np.array(vec)
        else:
            return self.tfidf.transform(X).toarray()



class CharacterFeatureGen(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_feat_gen_ = 3

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X["count_question_marks"] = X.apply(lambda x: str(x).count("?"))
        X["count_lines"] = X.apply(lambda x: str(x).count("\n"))
        X["count_exclamation"] = X.apply(lambda x: str(x).count("!"))
        return X
