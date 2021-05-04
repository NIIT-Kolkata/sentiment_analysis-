# -*- coding: utf-8 -*-
"""Untitled15.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yAnc6o7AXzZeWtD4xUraYMgwzchFtwmU
"""

import numpy as np
import pandas as pd
import seaborn as sns

df=pd.read_csv('/content/updated_sentiment_analysis(2).csv')

df.head()

df=df.sample(10000)
data=df

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=500)
x= cv.fit_transform(df['review']).toarray()

y=df['sentiment_positive']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=2021,test_size=0.2)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

ann_model=Sequential()

ann_model.add( Dense(8, activation="relu", input_shape = (x.shape[1],) ) )
ann_model.add( Dense(8, activation="relu") )
ann_model.add( Dense(1, activation="sigmoid"))

ann_model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

ann_model.fit(x_train,y_train, batch_size=100,epochs=20,verbose=1,validation_data=(x_test,y_test))

