#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python script for defining model classes for traditional machine learning algorithm
"""

from utils.feature_extraction import *
from utils.preprocessing_util import *
import warnings
import inspect
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def get_unique_label(data, label_col, annotator_col, count_dict):
    """
    function to get unique label from list of multiple labels
    :param data: dataframe
    :param label_col: column in the dataframe containing multiple label annotations
    :param annotator_col: label denoting annotators for the post
    :param count_dict: dictionary with annotators as key and number of posts annotated by respective annotator as values
    :return: Series of unique labels assigned to post
    """
    unique_label_list = []
    for row in data.itertuples(index=False):
        if len(set(row[data.columns.get_loc(label_col)].split(";"))) == 1:
            # if all labels assigned are same we simply take the unique value
            unique_label = row[data.columns.get_loc(label_col)].split(";")[0]
        elif len(set(row[data.columns.get_loc(label_col)].split(";"))) == len(row[data.columns.get_loc(label_col)].split(";")):

            # if there's tie we select label corresponding to annotator which has annotated the most number of post among all
            unique_label = row[data.columns.get_loc(label_col)].split(";")[0]
            annotator = row[data.columns.get_loc(annotator_col)].split(";")[0]
            for i in range(1, len(row[data.columns.get_loc(annotator_col)].split(";"))):
                if count_dict[row[data.columns.get_loc(annotator_col)].split(";")[i]] > count_dict[annotator]:
                    unique_label = row[data.columns.get_loc(label_col)].split(";")[i]
                    annotator = row[data.columns.get_loc(annotator_col)].split(";")[i]
        else:
            # finally we select label which is given by majority of annotators
            count_label = {label: 0 for label in set(row[data.columns.get_loc(label_col)].split(";"))}
            for label in row[data.columns.get_loc(label_col)].split(";"):
                count_label[label] += 1
            unique_label = sorted(count_label.items(), key=lambda x: x[1], reverse=True)[0][0]
        unique_label_list.append(unique_label)
    return unique_label_list


class ModelTrainingWarning(UserWarning):
    pass


class Basemodel(object):
    """
    base model object for either classification or regression. Specific classes for specific use can be derived from this class
    """

    def __init__(self, data_loc, output_col, text_col, word_2_vec_dim, categorical_vars_lab_enc, categorical_vars_one_hot_enc, categorical_vars_rank_enc, numerical_vars, do_kernel_transform, test_size, model):
        """
        :param data_loc: location of training data set
        :param output_col: name of the output variable
        :param text_col: name of column containing text
        :param categorical_vars_lab_enc: list of categorical variable which have to be label encoded
        :param categorical_vars_one_hot_enc: list of categorical variables which have to be one-hot encoded
        :param categorical_vars_rank_enc: list of categorical variables which have to be ordinal encoded
        """
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        assert os.path.exists(self.data_loc) == 1, "Given path for dataset is incorrect"
        self.data = pd.read_csv(self.data_loc)
        self.feature_vars = list(set([self.text_col] + self.categorical_vars_lab_enc + self.categorical_vars_one_hot_enc + self.categorical_vars_rank_enc + self.numerical_vars))
        # checking if the variables specified exist in the data set
        for var in self.feature_vars:
            assert var in self.data.columns, "%s column not in the given dataset. Kindly ensure the spelling of each feature variable is correct"

        # text feature extraction pipeline
        self.text_feat_ext = Pipeline([('stem', Stem()),
                                       ("text_num_feat", ColumnTransformer([('char_feat_extraction', CharacterFeatureGen(), self.text_col),
                                                           ('word_2_vec', TfidfEmbeddingVectorizer(self.word_2_vec_dim), self.text_col)],
                                                          remainder='drop', verbose=True))])
        # categorical feature extraction
        self.categorical_feat_ext = ColumnTransformer([('one_hot', Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder())]), self.categorical_vars_one_hot_enc),
                                                       ('label', Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", LabelEncoder())]), self.categorical_vars_lab_enc),
                                                       ('ordinal', Pipeline([("imputer", SimpleImputer(strategy="median")), ("encoder", OrdinalEncoder())]), self.categorical_vars_rank_enc)],
                                                      remainder='drop', verbose=True)

        # building the overall feature extraction pipeline
        self.feat_ext = self.build_feature_extraction_pipeline()

        # building model pipeline
        self.model_pipeline = self.build_model_pipeline()
        self.all_hyperparameters_list = [param for param in self.model_pipeline.get_params(True) if "__" in param]

    def build_feature_extraction_pipeline(self):
        """
        method to build the feature extraction pipeline
        :return: feat extraction pipeline of type sklearn.pipeline.Pipeline
        """
        if getattr(self, 'do_kernel_transform', False):
            # if kernel transformation is specified to be done we do kernel transformation of the generated features to account for non-linearity
            feat_ext = Pipeline([('feat_ext', ColumnTransformer([('text_feat_ext', self.text_feat_ext, self.text_col),
                                                                 ('cat_feat_ext', self.categorical_feat_ext,
            self.categorical_vars_lab_enc + self.categorical_vars_one_hot_enc + self.categorical_vars_rank_enc)],
                                              remainder='passthrough')), ("kernel_transform", Nystroem(n_components=1000))], verbose=True)
        else:
            feat_ext = ColumnTransformer([('text_feat_ext', self.text_feat_ext, self.text_col),
                                          ('cat_feat_ext', self.categorical_feat_ext, self.categorical_vars_lab_enc + self.categorical_vars_one_hot_enc + self.categorical_vars_rank_enc)], remainder='passthrough', verbose=True)
        return feat_ext

    def build_model_pipeline(self):
        """
        function build the overall model pipeline
        :return: model pipeline of type sklearn.pipeline.Pipeline
        """
        model_pipeline = Pipeline(
            [('feature_extraction_pipeline', self.feat_ext),
             ('clf', self.model)])
        return model_pipeline

    def fit(self, **kwargs):
        """
        function to fit the model to training data.
        We can provide hyperparameter dictionary to select optimum hyperparameter values or else model is fitted with default values
        :param kwargs:
        :return: None
        """
        tuning_hyperparameters_dict = {}
        args_dict = kwargs
        hyperparameter_dict = args_dict.get('hyperparameter_dict', {})
        complete_search = args_dict.get('complete_search', False)
        n_cpus = args_dict.get('n_cpus', -1)
        n_folds = args_dict.get('n_folds', 5)
        num_random_search_iter = args_dict.get('num_random_search_iter', 50)

        # getting hyperparameters dicitionary with actual name as per model pipeline.
        for param in hyperparameter_dict:
            for model_param in self.all_hyperparameters_list:
                if param == model_param.rsplit("__", 1)[1]:
                    tuning_hyperparameters_dict[model_param] = hyperparameter_dict[param]
        # check if hyperparameter dictionary is empty or not
        if not tuning_hyperparameters_dict:
            warnings.warn('No hyperparameters to train. fitting the model with default/pre-defined optimum values',
                          ModelTrainingWarning)
            self.best_model = self.model_pipeline.fit(self.X_train, self.Y_train)
            return None

        if complete_search == True:
            print("Grid search for optimal hyper parameters")
            model_cross_val = GridSearchCV(self.model_pipeline, param_grid=tuning_hyperparameters_dict,
                                           scoring=args_dict['evaluation_metric'], n_jobs=n_cpus,
                                           cv=n_folds, verbose=True)
        else:
            print("Randomized search for optimal hyper parameters")
            model_cross_val = RandomizedSearchCV(self.model_pipeline, tuning_hyperparameters_dict,
                                                 scoring=args_dict['evaluation_metric'],
                                                 n_jobs=n_cpus, cv=n_folds,
                                                 n_iter=num_random_search_iter,
                                                 random_state=100, verbose=True)
        model_cross_val.fit(self.X, self.Y)
        self.cross_val_result_summary = model_cross_val.cv_results_
        self.best_score = model_cross_val.best_score_
        self.best_model = model_cross_val.best_estimator_
        return None

    def save_model(self, model_loc):
        """
        method to save fitted model
        :param model_loc: directory to save the model
        :return:
        """
        if not os.path.isdir(os.path.split(model_loc)[0]) and os.path.split(model_loc)[0] != "":
            os.makedirs(os.path.split(model_loc)[0])
        joblib.dump(self.best_model, model_loc)

    def predict(self, trained_model_loc, test_data_loc, pred_loc):
        try:
            clf = joblib.load(trained_model_loc)
        except Exception as e:
            print("Got following error while loading model/n%s" % e)

        if not os.path.isdir(os.path.split(pred_loc)[0]) and os.path.split(pred_loc)[0] != "":
            os.makedirs(os.path.split(pred_loc)[0])

        try:
            test_data = pd.read_csv(test_data_loc)
        except Exception as e:
            print("Got following error while reading test data/n%s" % e)
        test_data = self.preprocess_data(test_data)
        try:
            test_data['predicted_label'] = clf.predict(test_data)
        except Exception as e:
            print("Got following error while predicting/n%s" % e)

        test_data.write_csv(pred_loc)

    def preprocess_data(self, data):
        raise NotImplementedError


class ClfReddit(Basemodel):
    """
    class for generic classifier for reddit post classifier. By default fits Multi naive bayes
    """

    def __init__(self, data_loc='reddit_post.csv', output_col='final_label', text_col='text', word_2_vec_dim=0, categorical_vars_lab_enc=[], categorical_vars_one_hot_enc= ['depth'], categorical_vars_rank_enc=[], numerical_vars=[], test_size=0.2, label_col='labels', deleted_post_str_indicator='[deleted]', do_kernel_transform=False, annotator_col='annotators', model = MultinomialNB(), **kwargs):
        super().__init__(data_loc, output_col, text_col, word_2_vec_dim, categorical_vars_lab_enc, categorical_vars_one_hot_enc, categorical_vars_rank_enc, numerical_vars, do_kernel_transform, test_size)
        for arg, val in kwargs.items():
            setattr(self, arg, val)
        # basic preprocessing of the data which includes removal of rows with missing values for the label column or text column and generating the output label column
        self.label_col = label_col
        self.deleted_post_str_indicator = deleted_post_str_indicator
        self.annotator_col = annotator_col
        self.preprocess_data = self.preprocess_data(self.data)
        # creating X and Y data frame
        self.X = self.preprocess_data[self.feature_vars]
        self.Y = self.preprocess_data[self.output_col]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=self.test_size, random_state=100)

    def preprocess_data(self, data):
        """
        method to preprocess given data
        :param data: input pandas DataFrame
        :return: preprocess output DataFrame
        """
        # removing rows which have missing values for text or label column
        data = data.dropna(subset=[self.text_col, self.label_col]).reset_index(drop=True)
        # removing rows which have text deleted or empty or spaces or special characters only
        data = data[~(data[self.text_col].str.replace(re.compile(r'[^\w\s]'), " ").str.replace(re.compile(r'\s+'), " ").isin([self.deleted_post_str_indicator, ' ', '']))].reset_index(drop=True)
        annotators_count = {}
        for row in data.iter(index=False):
            for annotator in row[data.columns.get_loc(self.annotator_col)].split(";"):
                if annotator in annotators_count:
                    annotators_count[annotator] += 1
                else:
                    annotators_count[annotator] = 1
        data[self.output_col] = get_unique_label(data, self.label_col, self.annotator_col, annotators_count)
        return data
