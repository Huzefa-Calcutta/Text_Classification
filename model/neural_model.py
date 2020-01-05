import tensorflow as tf
from utils.preprocessing_util import *
from utils.feature_extraction import *
import inspect
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, BatchNormalization, Dropout, MaxPooling1D, Conv1D, Activation, Flatten, Input, Dense, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math


def focal_loss(alpha, gamma):
    """
    defining softmax focal loss
    :param alpha: balances focal loss
    :param gamma: modulating factor which adjusts the rate at which easy examples are down-weighted
    :param y_true: ground truth labels
    :param y_pred: predicted softmax probabilities for each class
    :return: focal loss function for given alpha and gamma
    """
    def focal_loss_fixed(y_true, y_pred):
        cross_entropy = tf.multiply(y_true, -tf.log(y_pred))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., y_pred), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, cross_entropy))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed


def get_word_embedding_matrix(word_2_vec, word_vec_dim, word_2_ind):
    vocab_size = len(list(word_2_ind.keys())) + 1
    weight_matrix = np.zeros(vocab_size, word_vec_dim)
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in word_2_ind.items():
        vector = word_2_vec.get(word)
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix


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


class ConvTextRedditClf(Model):

    def __init__(self, data_loc, output_col, text_col, annotator_col, other_var, test_split_ratio, **kwargs):
        super().__init__()

        # getting all arguments except for kwargs
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        # setting arguments of the class
        for arg, val in values.items():
            setattr(self, arg, val)

        for arg, val in kwargs.items():
            setattr(self, arg, val)

        if isinstance(self.conv_filter_size, int):
            self.conv_filter_size_list = [self.conv_filter_size] * self.num_conv_layers
        if isinstance(self.conv_filter_size, (list, tuple, np.ndarray, set)):
            if len(self.conv_filter_size_list) < self.num_conv_layers:
                raise ValueError("The filter size for all convolution layers have not been specified. Please ensure the conv_filter_size has same numebr of elements as num_conv_layers")
            self.conv_filter_size_list = self.conv_filter_size

        if isinstance(self.num_conv_filter, int):
            self.conv_filter_num_list = [self.num_conv_filter] * self.num_conv_layers
        if isinstance(self.num_conv_filter, (list, tuple, np.ndarray, set)):
            if len(self.conv_filter_num_list) < self.num_conv_layers:
                raise ValueError(
                    "The number of filters for all convolution layers have not been specified. Please ensure the conv_filter_size has same numeber of elements as num_conv_layers")
            self.conv_filter_num_list = self.num_conv_filter

        if isinstance(self.dense_neuron_list, int):
            self.dense_neuron_list = [self.dense_neuron] * self.num_dense_layers
        if isinstance(self.conv_filter_size, (list, tuple, np.ndarray, set)):
            if len(self.dense_neuron_list) < self.num_dense_layers:
                raise ValueError( "The number of neurons for all dense layers have not been specified. Please ensure the dense_neuron has same number of elements as num_dense_layers")
            self.dense_neuron_list = self.dense_neuron
        self.predict_var = [self.text_col, self.depth_col] + self.other_var

        self.data = pd.read_csv(self.data_loc)

        # basic preprocessing of the data which includes removal of rows with missing values for the label column or text column and generating the output label column
        self.data = self.preprocess_data(self.data)

        # train and test data. Note test data is not used for hyper parameter tuning. for hyperparameter training we generate validation data from train_data itself
        self.train_data, self.test_data = train_test_split(self.data, self.test_split_ratio)
        self.text_preprocessing = Pipeline([("column_selection", ColumnSelector([self.text_col])),
                                ("text_cleaning", TextCleaning(self.text_col)), ("stem", Stem(self.text_col, True))])
        self.text_feat_generation = Pipeline([('preprocessing', self.text_preprocessing), ('tokeniser', TextTokeniser(self.text_col))])
        self.other_feat_gen = ColumnTransformer([("other_column_selector", "column_selection", ColumnSelector(self.other_var), self.other_var), ("Character_feature_gen", CharacterFeatureGen(self.text_col), self.text_col),
                              ('Encoding_post_depth_var', OneHotEncoder(handle_unknown='ignore'), [self.depth_post_col])], remainder='drop')

        self.is_other_feat_gen_train = False
        self.is_text_feat_gen_train = False
        self.no_other_feature = len(pd.unique(self.data[self.depth_col])) + CharacterFeatureGen(self.text_col).num_feat_gen_
        self.input_sequence_size = max(self.text_preprocessing.fit_transform(self.data)[self.txt_col].apply(lambda x: len(x.strip().lower().split(" "))))
        self.vocab = sorted(set(self.text_preprocessing.fit_transform(self.data)[self.txt_col].str.strip().sum()))

        # creating dictionaries to map word to integers and vice-versa and dictionary to map words to their embedded vectors. We reserve the index 0 for empty string ''
        self.word_2_index, self.index_2_word, self.word_2_vec_dict = self.get_word_dict()

    def get_word_dict(self):
        word_2_index = {word: index + 1 for index, word in enumerate(self.vocab)}
        index_2_word = {index: word for word, index in self.word_2_index.items()}

        word_2_vec_dict = {}
        with open(os.path.join(self.word_emb_folder, "glove.6B.%dd.txt" % self.word_vec_dim), "rb") as word_embedding:
            for line in word_embedding:
                word_2_vec_dict[line.strip().split()[0]] = np.array(map(float, line.strip().split()[1:]))
        return word_2_index, index_2_word, word_2_vec_dict

    def preprocess_data(self, data):
        # removing rows which have missing values for text or label column
        data = data.dropna(subset=[self.text_col, self.label_col]).reset_index(drop=True)
        # removing rows which have text deleted
        data = data[data[self.text_col] != self.empty_post_str_indicator].reset_index(drop=True)
        annotators_count = {}
        for row in data.itertuples(index=False):
            for annotator in row[data.columns.get_loc(self.annotator_col)].split(";"):
                if annotator in annotators_count:
                    annotators_count[annotator] += 1
                else:
                    annotators_count[annotator] = 1
        data[self.output_col] = get_unique_label(data, self.label_col, self.annotator_col, annotators_count)
        return data

    def build_model(self, save_model_arch_file=False):
        # build the first (CONV => RELU) * 1 => POOL layer set
        embedding_vectors = get_word_embedding_matrix(self.word_2_vec_dict, self.word_vec_dim, self.word_2_index)

        text_input_layer = Input(shape=(self.input_sequence_size,))
        embedding_layer = Embedding(len(self.vocab), 100, weights=[embedding_vectors], input_length=self.input_sequence_size,
                                    trainable=True)
        x = embedding_layer(text_input_layer)
        for i in range(self.num_conv_layers):
            conv_layer = Conv1D(filters=self.num_conv_filter[i], kernel_size=5, activation=None,
                                kernel_initializer='glorot_uniform', name="conv1_%d" % i)
            x = conv_layer(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        other_feat_input = Input(shape=(self.no_other_feature,))
        dense_inp = concatenate([x, other_feat_input])
        for i in range(self.num_dense_layers):
            dense_inp = Dropout(0.4)(dense_inp)
            dense_inp = Dense(self.dense_neuron_list[i], name="dense_layer_%d" % i)(dense_inp)
            dense_inp = BatchNormalization()(dense_inp)
            dense_inp = Activation('sigmoid')(dense_inp)

        output = Dense(1, activation='softmax', name="final_output")(dense_inp)
        self.model = Model(inputs=[text_input_layer, other_feat_input],
                      outputs=output)
        if save_model_arch_file:
            tf.keras.utils.plot_model(self.model, show_shapes=True)
        return None

    def batch_data_generator(self, inp_data, batch_size):
        if not self.is_preprocessing_train:
            self.text_preprocessing.fit(inp_data)
            self.is_preprocessing_train = True

        if not self.is_other_feat_gen_train:
            self.other_feat_gen.fit(inp_data)
            self.is_other_feat_gen_train = True

        size = 0
        text_inp = []
        other_feature_inp = []
        while True:
            for row in inp_data.itertuples(index=False):
                while size < batch_size:
                    text_inp.append(pad_sequences(self.text_feat_generation.transform(row), maxlen=self.input_sequence_size, padding='post'))
                    other_feature_inp.append(self.other_feat_gen.transform(row))
                    size += 1

            yield [np.array(text_inp), np.array(other_feature_inp)], row[inp_data.columns.get_loc(self.output_col)]

    def fit(self, train_data=None, val_data=None, batch_size=128, val_split=0.25, num_epochs=1000, optimizer='rmsprop', loss=focal_loss(0.4, 2.0), gpu_no=0):

        if self.model is None:
            self.build_model(True)
        if train_data is None:
            train_data = self.data
        else:
            train_data = self.preprocess_data(train_data)

        if val_data is None:
            train_data, val_data = train_test_split(train_data, val_split, random_state=100)
        self.model.compile(optimizer, loss)
        earlystop = EarlyStopping(monitor='val_final_output_loss', patience=20, mode='min',
                                  restore_best_weights=True)  # ensures that we have model weights corresponding to the best value of the metric at the end of

        # make tensorbaord log_dir
        if not os.path.exists("logs"):
            os.mkdir("logs")
        tensorboard = TensorBoard(log_dir='./logs', write_graph=True, update_freq='epoch')

        # Saving model_checkpoint
        filepath = "forecaster_model-{epoch:02d}-{val_final_output_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_final_output_loss', verbose=1,
                                     save_best_only=True,
                                     mode='min', save_weights_only=False)
        with tf.device('/gpu:%s' % gpu_no):
            self.history = self.model.fit_generator(self.batch_generator(train_data, batch_size), epochs=num_epochs,
                                                   steps_per_epoch=math.ceil(train_data.shape[0] / num_epochs) + 1,
                                                   validation_data = self.batch_generator(val_data, batch_size),
                                                   validation_steps=int(val_data.shape[0] / batch_size) + 1,
                                                   callbacks=[checkpoint, earlystop, tensorboard], verbose=2,
                                                   shuffle=False,
                                                   use_multiprocessing=True, workers=8)

        return None

    def predict(self, test_data=None, batch_size=8, gpu_no=None):
        if test_data is None:
            test_data = self.test_data
        else:
            test_data = self.preprocess_data(test_data)

        if gpu_no is None:
            self.model.predict_generator(self.batch_generator(test_data, batch_size)[0], steps=math.ceil(test_data.shape[0] / batch_size), shuffle=False, use_multiprocessing=True, workers=8)
        else:
            with tf.device('/gpu:%s' % gpu_no):
                self.model.predict_generator(self.batch_generator(test_data, batch_size)[0], steps=math.ceil(test_data.shape[0]/batch_size), shuffle=False, use_multiprocessing=True, workers=1)
