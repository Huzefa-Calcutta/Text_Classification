from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import re
import numpy as np
import pandas as pd


class Stem(BaseEstimator, TransformerMixin):
    def __init__(self, do_stem=False):
        self.do_stem = do_stem

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        if self.do_stem:
            X= X.apply(
                lambda post: " ".join([SnowballStemmer("english").stem(word) for word in post.strip().split()]))
            return pd.DataFrame(X)
        else:
            return pd.DataFrame(X)


class TextTokeniser(BaseEstimator, TransformerMixin):
    def __init__(self, str_col, vocab=set()):
        self.str_col = str_col
        self.vocab = vocab
        
    def fit(self, X, Y=None):
        self.word_2_index = {}
        self.index_2_word = {}
        if len(self.vocab) == 0:
            for sentence in X[self.str_col]:
                self.vocab = self.vocab.union(sentence.lower().split())
            self.vocab = sorted(self.vocab)
        i = 0
        for token in (self.vocab):
            i += 1
            self.word_2_index[token] = i
            self.index_2_word[i] = token
            
    def transform(self, X):
        token_doc = []
        try:
            for sentence in X[self.str_col]:
                token_doc.append([self.word_2_index.get(word, np.NaN) for word in sentence.split()])
            return token_doc
        except KeyError:
            raise KeyError("The DataFrame does not include the column: %s" % self.str_col)
    
    def inverse_transform(self, inv_X):
        decoded_doc = []
        for inv_row in inv_X:
            decoded_doc.append(" ".join([self.index_2_word[index] for index in inv_row]))
        return pd.Series(decoded_doc)

            
class TextCleaning(BaseEstimator, TransformerMixin):
    
    def __init__(self, str_col):
        self.str_col = str_col

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        try:
            X[self.str_col] = X[self.str_col].astype(str)
            X[self.str_col] = X[self.str_col].str.lower().str.strip().replace(re.compile('[^\w\s]'), ' ')
            X[self.str_col] = X[self.str_col].str.replace(re.compile('\s+'), ' ')
            X[self.str_col] = X[self.str_col].str.strip()
            X[self.str_col] = X[self.str_col].apply(lambda x: " ".join([word for word in x.strip().split() if word not in stopwords.words('english')]))
            return X
        except KeyError:
            raise KeyError("The DataFrame does not include the column: %s" % self.str_col)


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
