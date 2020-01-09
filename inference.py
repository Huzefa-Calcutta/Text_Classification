#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python script for doing inference
"""

import os
import datetime
import configparser
from model.model import *
from model.neural_model import *


if __name__ == '__main__':
    # loading the config file with info about location of test data and model
    cfgParse = configparser.ConfigParser()
    cfgParse.read("input.cfg")

    # location of model storage
    model_loc = cfgParse.get("model", "folder")
    # location of test data
    test_data = cfgParse.get("data", "test_data")
    # directory where predictions have to be stored
    prediction_loc = cfgParse.get('data', 'predicted_loc')
    if not os.path.isdir(prediction_loc):
        os.makedirs(prediction_loc)

    prediction_time_st = datetime.datetime.now()
    # creating classifier instance
    clf = ClfReddit()
    clf.predict(model_loc, test_data, prediction_loc)

    prediction_time_end = datetime.datetime.now()
    prediction_time = (prediction_time_end - prediction_time_st).total_seconds() / 60.0
    print("Time required for prediction for random forest model is %.3f minutes" % prediction_time)
