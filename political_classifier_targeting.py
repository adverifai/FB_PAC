#%%
import json
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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion


#%%
# READ DATA FILES
fb_ads_file=codecs.open('fbpac-ads-en-US.csv', 'r',encoding="utf-8")
fb_ads_data=pd.read_csv(fb_ads_file)
fb_ads_file.close()

#seeds_file=codecs.open('seeds.csv', 'r',encoding="utf-8")
#seeds_data=pd.read_csv(seeds_file)
#seeds_file.close()


#%%
# SANITY CHECKS..
#print(fb_ads_data.isnull().sum())
#print(seeds_data.isnull().sum())
#%%
# SET LABEL FOR SEEDS
seeds_data['target'] = np.where(seeds_data['target']=='political', 1, -1)

#%%
# SET LABEL AND TEXT FOR FB_ADS (ALL ADS IN THIS DATA WERE PRE-CLASSIFIED AS POLITICAL)
#fb_ads_data['target']=1
fb_ads_data["text"]=fb_ads_data["title"].fillna("")+" "+fb_ads_data["message"]
data=fb_ads_data.copy()

#%%
# SOME CODE WE COULD USE IN THE FUTURE TO WEIGH FB_ADS TARGET BASED ON ACTUAL VOTES
#data['weight1']=data['political']/(data['political']+data['not_political'])
#data.dropna(subset=['weight1'],inplace=True)

#all_votes=data['political'].sum()+data['not_political'].sum()
#data['weight2']=(data['political']+data['not_political'])/all_votes
#data['weight2']=(data['weight2']-data['weight2'].min())/(data['weight2'].max()-data['weight2'].min())

#data['weight']=data['weight1']*data['weight2']

data['total_votes']=data['political']+data['not_political']
data.dropna(subset=['total_votes'],inplace=True)
data['target']=np.sign(data['political']-data['not_political'])
data=data[data['target']!=0]
#%%
print(data.shape)
print(data.columns)
#data=data.sample(frac=1)

#%%
advertisers=data.copy()
advertisers=advertisers['advertiser'].unique()
advertisers.shape
#%%
advertisers=pd.DataFrame(advertisers,columns=["advertiser"])
advertisers=advertisers.sample(frac=0.2,random_state=42)
advertisers['test']=1
advertisers.shape

#%%
# SET THE TRAINING DATA, EITHER: 1) SEEDS DATA ONLY 2) SEEDS DATA + FB_ADS
#data=seeds_data
#fb_ads_data2=fb_ads_data[['target','text']]
#data=pd.concat([seeds_data,fb_ads_data2])
#X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
#train,test=train_test_split(data,test_size=0.2,random_state=42)
data=pd.merge(data,advertisers,on=['advertiser'],how='left')


#%%
train=data[data['test']!=1]
test=data[data['test']==1]
train.to_csv('fbpac-ads-en-US-train.csv',index=False)
test.to_csv('fbpac-ads-en-US-test.csv',index=False)

#%%
data['targets']=data['targets'].astype(str)
unique_lables=set()
for index, row in data.iterrows():
    try:
        targets=json.loads(row['targets'])
    except Exception as e: 
    #    print(targets)
    #    print(e)
    #    break
        x=1
    for t in targets:
        k=t['target']
        unique_lables.add(k)
print(unique_lables)
#%%
target_labels=unique_lables
target_cols={}
for lbl in target_labels:
        target_cols[lbl]=[]
for index, row in data.iterrows():
    #for x in row['targets']:
    try:
        targets=json.loads(row['targets'])
    except Exception as e:
         x=1
    targets_dict={}
    for t in targets:
        k=t['target']
        if k!='List' and k!='Like' and k!='Activity on the Facebook Family' and k!='Engaged with Content':
            #print(t)
            v=t['segment']
            #print(targets)
        else:
            v=1
        targets_dict[k]=v
    for lbl in target_labels:
        to_add=None
        if lbl in targets_dict:
            to_add=targets_dict[lbl]
        target_cols[lbl].append(to_add)

#%%
targets_df=pd.DataFrame.from_dict(target_cols)
for col in targets_df.columns:
    print(col+"|"+str(targets_df[col].count())+"|"+str(len(targets_df[col].unique())))   
#%%
print(pd.Series(target_cols['Activity on the Facebook Family']).value_counts())

#%%
data=data[['target','advertiser','text','test']]
#%%
for col in targets_df.columns.drop(['MaxAge','MinAge','State']):
    print(col)
    # integer encode
    label_encoder = LabelEncoder()
    targets_df[col] = label_encoder.fit_transform(targets_df[col].astype(str))
    integer_encoded = label_encoder.fit_transform(targets_df[col])
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    col_names=[col+"_"+str(x) for x in range(1,onehot_encoded.shape[1])]
    data=pd.concat([data, pd.DataFrame(onehot_encoded[:,1:],columns = col_names)],axis=1)
#%%
print(data.shape)
#%%
train=data[data['test']!=1]
test=data[data['test']==1]
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
categorical_feautres=list(data.columns.drop(['advertiser','text','target','test']))
features=['text']+categorical_feautres
#%%
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)
get_categorical_data = FunctionTransformer(lambda x: x[categorical_feautres], validate=False)        
        
clf = Pipeline([
    ('features', FeatureUnion([
            ('categorical_features', Pipeline([
                ('selector', get_categorical_data)
            ])),
             ('text_features', Pipeline([
                ('selector', get_text_data),
                ('vec', CountVectorizer(tokenizer=Tokenizer(),ngram_range=(1, 2),max_features=5000,stop_words='english')),
                ('tfidf', TfidfTransformer())
            ]))
         ])),
    ('clf', LogisticRegression(class_weight='balanced'))
])
    
clf.fit(train[features],train['target'])
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

