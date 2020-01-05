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


def get_unique_label(data, label_col, annotator_col, count_dict):
    unique_label_list = []
    for row in data.itertuples(index=False):
        if len(set(row[data.columns.get_loc(label_col)].split(";"))) == 1:
            unique_label = row[data.columns.get_loc(label_col)].split(";")[0]
        elif len(set(row[data.columns.get_loc(label_col)].split(";"))) == len(
                row[data.columns.get_loc(label_col)].split(";")):
            unique_label = row[data.columns.get_loc(label_col)].split(";")[0]
            annotator = row[data.columns.get_loc(annotator_col)].split(";")[0]
            for i in range(1, len(row[data.columns.get_loc(annotator_col)].split(";"))):
                if count_dict[row[data.columns.get_loc(annotator_col)].split(";")[i]] > count_dict[annotator]:
                    unique_label = row[data.columns.get_loc(label_col)].split(";")[i]
                    annotator = row[data.columns.get_loc(annotator_col)].split(";")[i]
        else:
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
    base model object
    """

    def __init__(self, data_loc, output_col, text_col, word_2_vec_dim, categorical_vars_lab_enc, categorical_vars_one_hot_enc, categorical_vars_rank_enc, numerical_vars, do_kernel_transform=False, test_size=0.2):
        """

        :param data_loc: location of training data set
        :param output_col: name of the output variable
        :param text_col: name of column containing text
        :param categorical_vars_lab_enc: list of categorical variable which have to be label encoded
        :param categorical_vars_one_hot_enc: list of categorical variables which have to onehot encoded
        """
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        assert os.path.exists(self.data_loc) == 1, "Given path for dataset is incorrect"
        self.data = pd.read_csv(self.data_loc)
        self.feature_vars = list(set([self.text_col] + self.categorical_vars_lab_enc + self.categorical_vars_one_hot_enc + self.categorical_vars_rank_enc + self.numerical_vars))
        for var in self.feature_vars:
            assert var in self.data.columns, "%s column not in the given dataset. Mention the name of the feature variable correctly"
        self.text_feat_ext = Pipeline([('char_feat_extraction', CharacterFeatureGen()), ('word_2_vec', ColumnTransformer([('word_vec_tfidf_weighted', TfidfEmbeddingVectorizer(self.word_2_vec_dim), self.text_col)], remainder='drop', verbose=True))])
        self.categorical_feat_ext = ColumnTransformer([('one_hot', OneHotEncoder(), self.categorical_vars_one_hot_enc), ('label', LabelEncoder(), self.categorical_vars_lab_enc), ('ordinal', OrdinalEncoder(), self.categorical_vars_rank_enc)], remainder='drop')
        self.feat_ext = self.build_feature_extraction_pipeline()

    def build_feature_extraction_pipeline(self):
        if getattr(self, 'do_kernel_transform', False):
            feat_ext = Pipeline([('feat_ext', ColumnTransformer([('text_feat_ext', self.text_feat_ext, self.text_col), (
            'cat_feat_ext', self.categorical_feat_ext,
            self.categorical_vars_lab_enc + self.categorical_vars_one_hot_enc + self.categorical_vars_rank_enc)],
                                              remainder='passthrough')), ("kernel_transform", Nystroem(n_components=1000))])
        else:
            feat_ext = ColumnTransformer([('text_feat_ext', self.text_feat_ext, self.text_col),
                                          ('cat_feat_ext', self.categorical_feat_ext, self.categorical_vars_lab_enc + self.categorical_vars_one_hot_enc + self.categorical_vars_rank_enc)], remainder='passthrough')
        return feat_ext

    def build_model_pipeline(self):
        model_pipeline = Pipeline(
            [('feature_extraction_pipeline', self.feat_ext),
             ('clf', self.model)])
        return model_pipeline

    def fit(self, **kwargs):
        tuning_hyperparameters_dict = {}
        args_dict = kwargs
        hyperparameter_dict = args_dict.get('hyperparameter_dict', {})
        complete_search = args_dict.get('complete_search', False)
        n_cpus = args_dict.get('n_cpus', -1)
        n_folds = args_dict.get('n_folds', 5)
        num_random_search_iter = args_dict.get('num_random_search_iter', 50)
        self.model_pipeline = self.build_model_pipeline()
        self.all_hyperparameters_list = [param for param in self.model_pipeline.get_params(True) if "__" in param]
        for param in hyperparameter_dict:
            for model_param in self.all_hyperparameters_list:
                if param == model_param.rsplit("__", 1)[1]:
                    tuning_hyperparameters_dict[model_param] = hyperparameter_dict[param]
        print(tuning_hyperparameters_dict)

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

    def save_model(self, model_loc):
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

        try:
            test_data['predicted_label'] = clf.predict(test_data)
        except Exception as e:
            print("Got following error while predicting/n%s" % e)

        test_data.write_csv(pred_loc, index=False)


class ClfReddit(Basemodel):

    def __init__(self, data_loc, output_col, text_col, word_2_vec_dim, categorical_vars_lab_enc, categorical_vars_one_hot_enc, categorical_vars_rank_enc, numerical_vars, do_kernel_transform, test_size, label_col, deleted_post_str_indicator, annotator_col):
        super().__init__(data_loc, output_col, text_col, word_2_vec_dim, categorical_vars_lab_enc, categorical_vars_one_hot_enc, categorical_vars_rank_enc, numerical_vars, do_kernel_transform, test_size)
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
        # removing rows which have missing values for text or label column
        data = data.dropna(subset=[self.text_col, self.label_col]).reset_index(drop=True)
        # removing rows which have text deleted
        data = data[data[self.text_col] != self.deleted_post_str_indicator].reset_index(drop=True)
        annotators_count = {}
        for row in data.itertuples(index=False):
            for annotator in row[data.columns.get_loc(self.annotator_col)].split(";"):
                if annotator in annotators_count:
                    annotators_count[annotator] += 1
                else:
                    annotators_count[annotator] = 1
        data[self.output_col] = get_unique_label(data, self.label_col, self.annotator_col, annotators_count)
        return data


class RandomForestClfReddit(ClfReddit):

    def __init__(self, data_loc='reddit_post.csv', output_col='final_label', text_col='text', word_2_vec_dim=0, categorical_vars_lab_enc=['depth'], categorical_vars_one_hot_enc= [], categorical_vars_rank_enc=[], numerical_vars=[], test_size=0.2, label_col='labels', deleted_post_str_indicator='[deleted]', do_kernel_transform=False, annotator_col='annotators', **kwargs):
        super().__init__(data_loc, output_col, text_col, word_2_vec_dim, categorical_vars_lab_enc, categorical_vars_one_hot_enc, categorical_vars_rank_enc, numerical_vars, do_kernel_transform, test_size, label_col, deleted_post_str_indicator, annotator_col)
        for arg, val in kwargs.items():
            setattr(self, arg, val)
        self.model = RandomForestClassifier(class_weight="balanced", verbose=True, random_state=100)


class SVMClfReddit(ClfReddit):

    def __init__(self, data_loc='reddit_post.csv', output_col='final_label', text_col='text', word_2_vec_dim=0, categorical_vars_lab_enc =[], categorical_vars_one_hot_enc =['depth'], categorical_vars_rank_enc=[], numerical_vars=[], do_kernel_transform=True, test_size=0.2, label_col='labels', deleted_post_str_indicator='[deleted]', annotator_col='annotators', **kwargs):
        super().__init__(data_loc, output_col, text_col, word_2_vec_dim, categorical_vars_lab_enc, categorical_vars_one_hot_enc, categorical_vars_rank_enc, numerical_vars, do_kernel_transform, test_size, label_col, deleted_post_str_indicator, annotator_col)
        self.model = SGDClassifier(loss='hinge', penalty='elasticnet', verbose=True)
        for arg, val in kwargs.items():
            setattr(self, arg, val)


class NBClfReddit(ClfReddit):

    def __init__(self, data_loc='reddit_post.csv', output_col='final_label', text_col='text', word_2_vec_dim=0, categorical_vars_lab_enc=['depth'], categorical_vars_one_hot_enc=[], categorical_vars_rank_enc=[], numerical_vars=[], do_kernel_transform=False, test_size=0.2, label_col='labels', deleted_post_str_indicator='[deleted]', annotator_col='annotators', **kwargs):
        super().__init__(data_loc, output_col, text_col, word_2_vec_dim, categorical_vars_lab_enc, categorical_vars_one_hot_enc, categorical_vars_rank_enc, numerical_vars, do_kernel_transform, test_size, label_col, deleted_post_str_indicator, annotator_col)
        for arg, val in kwargs.items():
            setattr(self, arg, val)
        self.model = MultinomialNB()
