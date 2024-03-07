import pandas as pd
import numpy as np
from nltk.tokenize import (word_tokenize)
from nltk import WordNetLemmatizer
#from gensim.models import Word2Vec
#import gensim.downloader
#import spacy
from sklearn.mixture import GaussianMixture

df = pd.read_csv("/Users/rakesh/Desktop/cnn_dailymail/train.csv")
df.drop(["id"], axis=1, inplace=True)
print(df.describe())