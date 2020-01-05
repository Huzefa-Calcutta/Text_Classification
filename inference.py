import pandas as pd
import joblib
import datetime
import sys
import configparser
import os
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')


all_stopwords = stopwords.words('english') + stopwords.words('german') + stopwords.words('italian') + stopwords.words('french') + stopwords.words('portuguese') + stopwords.words('spanish')
all_stopwords.extend(['http', 'https'])


def pre_process(input_data_loc, source_encoder):
    tweet_data = pd.read_csv(input_data, engine='python')
    tweet_data.dropna(subset=['tweet'], inplace=True)
    tweet_data.reset_index(drop=True, inplace=True)

    imputer = ColumnTransformer([('cat_imputer', SimpleImputer(missing_values=None, strategy='most_frequent'), ['has_hashtag', 'has_mentions', 'has_url', 'has_media', 'lang']),
                                 ('numerical_imputer', SimpleImputer(strategy='most_frequent'), ['source'])], remainder='drop')
    tweet_data[['has_hashtag', 'has_mentions', 'has_url', 'has_media', 'lang', 'source']] = imputer.fit_transform(tweet_data)

    tweet_data[['has_hashtag', 'has_mentions', 'has_url', 'has_media']] = tweet_data[['has_hashtag', 'has_mentions', 'has_url', 'has_media']].astype(bool)
    tweet_data['tweet'] = tweet_data['tweet'].str.lower().str.replace('[^\w\s]', '').apply(lambda x: " ".join(x.split()))
    tweet_data['tweet'] = tweet_data['tweet'].apply(lambda x: " ".join([word for word in x.split() if word != ""]))
    tweet_data.loc[~tweet_data['lang'].isin(['en', 'und', 'es', 'pt']), 'lang'] = 'others'

    source_map = {}
    with open("source_mapping.txt", 'r') as inp:
        for line in inp:
            source_map[int(line.strip().split(":")[1].strip())] = line.strip().split(":")[0]

    tweet_data['source_info'] = tweet_data['source'].apply(lambda x: source_map[x])

    source_enc_df = pd.DataFrame(source_encoder.transform(tweet_data[['source_info']]).todense())
    source_enc_df.columns = source_encoder.get_feature_names()
    tweet_data = pd.concat([tweet_data[['lang', 'has_hashtag', 'has_mentions', 'has_url', 'has_media', 'tweet']], source_enc_df], axis=1)

    return tweet_data


def prediction(test_data, model, author_enc):
    pred = model.predict(test_data)
    return author_enc.inverse_transform(pred)


if __name__ == '__main__':
    # loading the config file
    cfgParse = configparser.ConfigParser()
    cfgParse.read(sys.argv[1])

    inp_data = cfgParse.get("input", "data")
    source_label_enc = joblib.load(cfgParse.get("input", "source_label_enc_map"))
    author_label_enc = joblib.load(cfgParse.get("input", "author_label_enc_map"))
    model = joblib.load(cfgParse.get("input", "model"))

    prediction_dir = cfgParse.get('output', 'predicted_dir')

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    prediction_time_st = datetime.datetime.now()

    processed_data = pre_process(inp_data, source_label_enc)
    processed_data['author_pred'] = prediction(processed_data, model, author_label_enc)
    processed_data[['tweet', 'author_pred']].to_csv(os.path.join(prediction_dir, "prediction.csv"), index=False)

    prediction_time_end = datetime.datetime.now()
    prediction_time = (prediction_time_end - prediction_time_st).total_seconds() / 60.0
    print("Time required for prediction for random forest model is %.3f minutes" % prediction_time)
