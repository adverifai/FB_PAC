import pandas as pd
import numpy as np
import codecs
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.metrics.pairwise import linear_kernel

#%%
# READ DATA FILES
train_file=codecs.open('fbpac-ads-en-US-train.csv', 'r',encoding="utf-8")
train_df=pd.read_csv(train_file)
train_file.close()
test_file=codecs.open('fbpac-ads-en-US-test.csv', 'r',encoding="utf-8")
test_df=pd.read_csv(test_file)
test_file.close()

#%%
# SET TOKENIZER WITH STEMMING
#from nltk.stem.porter import *    
#stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")

#pattern = r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'
pattern = r'\w+|\?|\!|\"|\'|\;|\:'

class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer(pattern)
        self.stemmer = stemmer
    def __call__(self, doc):
        return [self.stemmer.stem(token) 
                for token in self.tok.tokenize(doc)]


#%%
# SET PIPELINE WITH TOKENIZER, N-GRAMS AND LOGISTIC REGRESSION MODEL
clf = Pipeline([('vect', CountVectorizer(tokenizer=Tokenizer(),ngram_range=(1, 2),max_features=5000,stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', LogisticRegression(class_weight='balanced'))])
        
# ANOTHER OPTION FOR PIPELINE WITH GRID SEARCH CV
#parameters = {
#    'max_features': (None, 5000, 10000, 50000),
#    'vect__ngram_range': [(1, 1), (1, 2)],
#    'tfidf__max_df': (0.25, 0.5, 0.75),
#    'tfidf__ngram_range': [(1, 2)],
#    'clf__estimator__alpha': (1,1e-2, 1e-3)
#}
#clf = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=3)

clf.fit(train['text'],train['target']),#sample_weight=train['weight'])

#%%
# SAVE CLASSIFIER
filename = 'text_classifier.pk'
with open(''+filename, 'wb') as file:
	pickle.dump(clf, file)


#%%
# LOAD CLASSIFIER
with open("text_classifier.pk" ,'rb') as f:
    clf= pickle.load(f)
    
    
#%%
# EVALUATE F1 AND ROC_AUC FOR TRAIN AND TEST SETS
y_predict_train=clf.predict(train['text'])
y_predict_test=clf.predict(test['text'])

print(roc_auc_score(train['target'],y_predict_train))
print(roc_auc_score(test['target'],y_predict_test))

print(f1_score(train['target'],y_predict_train))
print(f1_score(test['target'],y_predict_test))
