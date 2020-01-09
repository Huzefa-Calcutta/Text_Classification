import os
import datetime
import configparser
from model.model import *
from model.neural_model import *


if __name__ == '__main__':
    # loading the config file
    cfgParse = configparser.ConfigParser()
    cfgParse.read("input.cfg")

    model_loc = cfgParse.get("model", "folder")
    test_data = cfgParse.get("data", "test_data")
    prediction_dir = cfgParse.get('data', 'predicted_dir')
    if not os.path.isdir(prediction_dir):
        os.makedirs(prediction_dir)

    prediction_time_st = datetime.datetime.now()

    clf = ClfReddit()
    clf.predict(model_loc, test_data, prediction_dir)

    prediction_time_end = datetime.datetime.now()
    prediction_time = (prediction_time_end - prediction_time_st).total_seconds() / 60.0
    print("Time required for prediction for random forest model is %.3f minutes" % prediction_time)
