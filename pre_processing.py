import string
import gensim
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
import numpy as np
import lxml.html.clean as clean
import re
np.random.seed(2018)

# uncomment the following two lines if your wordnet is not up-to-date
# import nltk
# nltk.download('wordnet')


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*>|<.*\"')
    result = re.sub(clean, '', text)
    return result


def remove_nonprintable(text):
    """Remove non printable characters from a string"""
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, text))


def lemmatize_stemming(text):
    """Lemmatizing and stemming a string"""
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    return stemmer.stem(WordNetLemmatizer().lemmatize(text))


def pre_process(text):
    result = []
    # removing the URLs
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    for token in gensim.utils.simple_preprocess(remove_nonprintable(remove_html_tags(text))):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
