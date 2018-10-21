"""
Purpose data Prepration for classification
Sardar 10/20/2018

"""
import json

# %%
import pandas as pd
import numpy as np
import codecs
import nltk
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support, classification_report
from sklearn.metrics.pairwise import linear_kernel
from sklearn import svm


# %%
# READ DATA FILES
def ds_gen(data_path):
    # Read data from file 'filename.csv'
    data_ = []
    for item in open(data_path).readlines():
        data_.append(json.loads(item))
    df = pd.DataFrame.from_dict(data_)
    return df



fb_ads_data = ds_gen("fbpac2.jsonl")

print(fb_ads_data.isnull().sum())

# %%
# SET LABEL FOR SEEDS
# fb_ads_data['label'] =fb_ads_data['label'].astype('category').cat.codes


# %%
# SET LABEL AND TEXT FOR FB_ADS (ALL ADS IN THIS DATA WERE PRE-CLASSIFIED AS POLITICAL)

data = fb_ads_data[['label', 'text']]
train, test = train_test_split(data, test_size=0.2, random_state=42)

# CONVERTING THE SERIES TO LIST

y_train=train['label'].values.tolist()
y_test=test['label'].values.tolist()


# %%
# SET TOKENIZER WITH STEMMING
# from nltk.stem.porter import *
# stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")

# pattern = r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'
pattern = r'\w+|\?|\!|\"|\'|\;|\:'



class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer(pattern)
        self.stemmer = stemmer

    def __call__(self, doc):
        return [self.stemmer.stem(token)
                for token in self.tok.tokenize(doc)]


# %%

clf = Pipeline(
    [('vect', CountVectorizer(tokenizer=Tokenizer(), ngram_range=(1,1), max_features=5000, stop_words='english')),
     ('clf',svm.SVC(kernel='rbf', gamma=10))])


clf.fit(train['text'], y_train),  # sample_weight=train['weight'])

# %%
# SAVE CLASSIFIER
filename = 'prp_classifier.pk'
with open('' + filename, 'wb') as file:
    pickle.dump(clf, file)

# %%
# LOAD CLASSIFIER
with open("prp_classifier.pk", 'rb') as f:
    clf = pickle.load(f)

y_predict_train = clf.predict(train['text'])
y_predict_test = clf.predict(test['text'])
print(classification_report(y_train, y_predict_train))
print("accuracy: %f" %(accuracy_score(y_train, y_predict_train)))



