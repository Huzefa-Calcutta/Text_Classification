#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python script for defining custom transformer classes for pre-processing of data
"""

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import re
import numpy as np
import pandas as pd


class Stem(BaseEstimator, TransformerMixin):
    """
    custom class for doing stemming the input text
    """
    def __init__(self, do_stem=False):
        """

        :param do_stem: bool variable indicating whether to do stemming or not
        """
        self.do_stem = do_stem

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        """

        :param X: input pandas series of text data
        :return: pandas dataframe with or without stemming
        """
        if self.do_stem:
            # stemming the
            X= X.apply(
                lambda post: " ".join([SnowballStemmer("english").stem(word) for word in post.strip().split()]))
            return pd.DataFrame(X) # Output has to have two dimension hence converting series to dataframe
        else:
           # returning as it is
           return pd.DataFrame(X)


class TextTokeniser(BaseEstimator, TransformerMixin):
    """
    transformer to tokenise text in to list of integers. Used for neural network based models. Not required for sklearn based models
    """
    def __init__(self, vocab=set()):
        """
        :param vocab: set of words in dictionary
        """
        self.vocab = vocab
        # dictionary to map word to index. Empty string is mapped to 0
        self.word_2_index = {"": 0}
        # dictionary to map index to the word. Index 0 is mapped to empty string
        self.index_2_word = {0:""}

    def fit(self, X, Y=None):
        """

        :param X: series of text data
        :param Y:
        :return:
        """

        # if empty vocab we generate new vocab from the text
        if len(self.vocab) == 0:

            for sentence in X:
                self.vocab = self.vocab.union(sentence.lower().split())
            self.vocab = sorted(self.vocab)
        i = 0
        for token in (self.vocab):
            i += 1
            self.word_2_index[token] = i
            self.index_2_word[i] = token
        return self

    def transform(self, X):
        """

        :param X: series of text data
        :return: list of token indexes for each word in each row
        """
        token_doc = []
        for sentence in X:
            # assigning NaN values to out of vocabulary words
            token_doc.append([self.word_2_index.get(word, np.NaN) for word in sentence.split()])
        return token_doc
    
    def inverse_transform(self, inv_X):
        decoded_doc = []
        for inv_row in inv_X:
            # assigning space to Nan values
            decoded_doc.append(" ".join([self.index_2_word.get(index," ") for index in inv_row]))
        return pd.Series(decoded_doc)

            
class TextCleaning(BaseEstimator, TransformerMixin):
    """
    transformer class for removing special characters and stopwords. Used for neural network models. Not required for sklearn models
    """
    def __init__(self):
        return None

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        X = X.astype(str)
        X = X.str.lower().str.strip().replace(re.compile('[^\w\s]'), ' ')
        X = X.str.replace(re.compile('\s+'), ' ')
        X = X.str.strip()
        X = X.apply(lambda x: " ".join([word for word in x.strip().split() if word not in stopwords.words('english')]))
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    transformer class for selecting set of columns from dataframe
    """
    def __init__(self, columns):
        """

        :param columns: list of column names to be selected
        """
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
