import numpy as np
import pandas as pd
import seaborn as sns

df=pd.read_csv('/IMDB Dataset.csv.zip')

df.head()

#data preprocessing

df=df.sample(10000)

#using one hot encoding convert 'positive' to 1 and 'negative' to 0

df = pd.get_dummies(df, columns=['sentiment'], drop_first=True)

df.head(5)

#cleaning the data

#remove the html tags

import re

def remove_tag(org):
  find = re.compile('<.*?>')
  cleantext = re.sub(find, '',org)
  return cleantext

df['review']=df['review'].apply(remove_tag)

df.head()

#tags have been removed

#next function to remove special characters

def remove_special_characters(org):
  cleantext=re.sub(r'\W+,. ', ' ', org)
  return cleantext

df['review']=df['review'].apply(remove_special_characters)

df.head()

#convert every character to lower case

def convert_to_lower(org):
  cleantext=org.lower()
  return cleantext

df['review']=df['review'].apply(convert_to_lower)



#every character has been converted to lowercase

def remove_punctuation(org):
  cleantext=re.sub(r'[^\w\s]', ' ',org)
  return cleantext

df['review']=df['review'].apply(remove_punctuation)

df.head()

pip install nltk

import nltk

from nltk.corpus import stopwords

#list of stopwords in english language

nltk.download('stopwords')

stopwords.words('english')

def remove_stopwords(org):
  cleantext=[]
  for word in org.split():
    if word not in stopwords.words('english'):
      cleantext.append(word)
  new=cleantext[:]
  cleantext.clear()
  return new

df['review']=df['review'].apply(remove_stopwords)

df.head()

from nltk.stem.porter import PorterStemmer

stemming_model=PorterStemmer()

def stem_words(org):
  cleantext=[]
  for word in org:
    cleantext.append(stemming_model.stem(word))
  new=cleantext[:]
  cleantext.clear()
  return new

df['review']=df['review'].apply(stem_words)

df.head()

def join_back(org):
  return " ".join(org)

df['review']=df['review'].apply(join_back)

df.head()

#converting updated dataframe to csv

df.to_csv('updated_sentiment_analysis.csv')

df2=pd.read_csv('updated_sentiment_analysis.csv')

df.head()

import warnings

warnings.filterwarnings('ignore', '.*do not.*', )

#end of preprocessing
 # converting to vectors from string 
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=500)
x= cv.fit_transform(df2['review']).toarray()

print(x.shape)

y=df2.iloc[:,-1].values


#end of preprocessing 