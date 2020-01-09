#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python script for defining custom transformer classes for feature extraction
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import os
import numpy as np
from nltk.corpus import stopwords
import re
import warnings


class WordVectorWarning(UserWarning):
    pass


class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    class for tfidf weighted word2vec transformation of input vector
    """
    def __init__(self, dim=0, word_emb_folder=""):
        """
        :param dim: int value for dimension of word vectors
        :param word_emb_folder: str directory where word vectors are stored
        """
        self.dim = dim
        if dim not in [0, 50, 100, 200, 300]:
            self.dim = 0
            warnings.warn("Invalid word vector dimension. Using tfidf transformation. Word vector dimension can only be one of 50,100,200,300", WordVectorWarning)
        self.word_emb_folder = word_emb_folder
        self.word2weight = None
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
                    self.word2vec[line.strip().split()[0]] = np.array(line.strip().split()[1:], dtype=np.float)
        return self

    def transform(self, X):
        doc_array = []
        if self.dim != 0:
            X = np.array(X)
            for row in X:
                # removing punctuation marks and new line characters and stopwords
                row = re.sub(re.compile(r'\s+'), " ", re.sub(re.compile(r'[^\w\s]'), " ", row))
                row = " ".join([word for word in row if word not in stopwords.words('english')])

                if row == " " or row == "":
                    # assigning n-dimension 0 word vectors to empty strings
                    word_vec_array = np.zeros(self.dim)
                else:
                    # assigning mean of all word vectors in the post weighted by corresponding tfidf values.
                    # words which ar out of dicitionary are assigned max tfidf values
                    # words for which pretrained word vectors are not available we assign unit n-dimensional vector
                    word_vec_array = np.mean(np.array([self.word2vec.get(word.lower(), np.ones(self.dim)) * self.word2weight.get(word.lower(), self.max_idf) for word in row.split()]), axis=0)
                doc_array.append(word_vec_array)
            return np.array(doc_array)
        else:
            return self.tfidf.transform(X)


class CharacterFeatureGen(BaseEstimator, TransformerMixin):
    """
    Transformer class to generate character based count features
    """
    def __init__(self):
        self.num_feat_gen_ = 3

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        X = pd.Series(X)
        # counting number of question marks
        count_question_marks = pd.Series(X.apply(lambda x: str(x).count("?")))
        # counting number of lines
        count_lines = pd.Series(X.apply(lambda x: str(x).count("\n")))
        # counting number of exclamation marks
        count_exclamation = pd.Series(X.apply(lambda x: str(x).count("!")))
        return pd.DataFrame({'count_question_marks': count_question_marks.values, 'count_lines': count_lines.values, 'count_exclamation': count_exclamation.values})
