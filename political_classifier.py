#%%
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
fb_ads_file=codecs.open('fbpac-ads-en-US.csv', 'r', encoding="utf-8")
fb_ads_data=pd.read_csv(fb_ads_file)
fb_ads_file.close()

seeds_file=codecs.open('seeds.csv', 'r', encoding="utf-8")
seeds_data=pd.read_csv(seeds_file)
seeds_file.close()


#%%
# SANITY CHECKS..
print(fb_ads_data.isnull().sum())
print(seeds_data.isnull().sum())


#%%
# SET LABEL FOR SEEDS
seeds_data['target'] = np.where(seeds_data['target']=='political', 1, -1)


#%%
# SET LABEL AND TEXT FOR FB_ADS (ALL ADS IN THIS DATA WERE PRE-CLASSIFIED AS POLITICAL)
fb_ads_data['target']=1
fb_ads_data["text"]=fb_ads_data["title"].fillna("")+" "+fb_ads_data["message"]


#%%
# SOME CODE WE COULD USE IN THE FUTURE TO WEIGH FB_ADS TARGET BASED ON ACTUAL VOTES
#data['weight1']=data['political']/(data['political']+data['not_political'])
#data.dropna(subset=['weight1'],inplace=True)

#all_votes=data['political'].sum()+data['not_political'].sum()
#data['weight2']=(data['political']+data['not_political'])/all_votes
#data['weight2']=(data['weight2']-data['weight2'].min())/(data['weight2'].max()-data['weight2'].min())

#data['weight']=data['weight1']*data['weight2']

#data['target']=np.sign(data['political']-data['not_political'])
#data=data[data['target']!=0]

#data=data.sample(frac=1)


#%%
# SET THE TRAINING DATA, EITHER: 1) SEEDS DATA ONLY 2) SEEDS DATA + FB_ADS
#data=seeds_data
fb_ads_data2=fb_ads_data[['target','text']]
data=pd.concat([seeds_data,fb_ads_data2])
#X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
train,test=train_test_split(data,test_size=0.2,random_state=42)


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
