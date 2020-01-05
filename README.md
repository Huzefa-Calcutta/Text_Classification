# Introduction
This is a Python based project which aims to build classifier for data containing text such as tweets, social media posts. 
We illustrate this with the help of reddit post data. The dataset was prepared by researchers from Google and is described in this blog post. The
original dataset that they collected is available in JSON format, but we have processed it into a
CSV, retaining only part of the information. Each post has been labeled by three or fewer annotators as belonging to one of the ten
categories. Note that the annotators are not always in agreement about the category, which
needs to be handled in your solution. We leave it up to you to define the classification target and
evaluation metrics based on these annotations. The depth column indicates how many levels
into a thread a post is found. A depth of 0 means that the post is the one that started the thread.
Replies directly to the start post are depth 1. Replies to depth 1 posts are in turn depth 2, and so
on.
Certain preprocessing functions have been written specifically for reddit data. The project could still be used other datasets. It's an going going to make this project generalizable to all datasets


# Project structure
There are 2 sub folders:
2. model - contains python script for model classes
   1. model - classes for classical machine learning algorithms. currently Random Forest, SVM, Naive Bayes implementation are available
   2. neural_model - classes for building neural network based algorithms for text classification. Currently we have Convolutional network implementati3. utils - contains python script for preprocessing and feature engineering

3. utils - contains python files for utility functions
    1. feature_extraction - python script for feature extraction from text data. Currently we have Tfidf weighted word embeddings feature extractor and transformer for generating character based count feature
    2. preprocessing_util - python script for text cleaning such as removing punctuation marks and tokenising the text
        
 # How to use
 1. Installing python packages       
    `pip install requirements.txt`
    
 2. create `data` folder and with in data create subfolders `train_data`, `model_data` and `test_data` and then place the data you intend to use for training in `data/train_data`.
    ```
    mkdir data
    cd data && mkdir train_data && mkdir test_data && mkdir model_data && cd ..
    ```
 3. Alternatively you can specify the location of training_data and model file in the config file. Specify the training configuration including dimensions of word vector and where your model needs to be stored etc. 
 4. Download glove word vectors 
    1. glove word vectors trained Wikipedia 2014
    `wget http://nlp.stanford.edu/data/glove.6B.zip`
    2. glove word vectors trained Common crawl (42 Billion tokens)
    `wget http://nlp.stanford.edu/data/glove.42B.300d.zip`
    3. glove word vectors trained Common crawl (840 Billion tokens)
    `wget http://nlp.stanford.edu/data/glove.840B.300d.zip`
    4.  glove word vectors trained on Tweets
    `wget http://nlp.stanford.edu/data/glove.twitter.27B.zip` 
 5. start jupyter notebook and open model_train.ipynb for training the model and data_visualization for exploratory analysis of data 
 
  