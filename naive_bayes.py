from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2021)

print(x_train.shape)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()

gnb.fit(x_train,y_train)

mnb.fit(x_train,y_train)

bnb.fit(x_train,y_train)

y_gnb_pred=gnb.predict(x_test)

y_mnb_pred=mnb.predict(x_test)

y_bnb_pred=bnb.predict(x_test)

from sklearn.metrics import accuracy_score

print("The accuracy score of Gaussian Naive Bayes is: ", accuracy_score(y_test,y_gnb_pred))

print("The accuracy score of Multinomial Naive Bayes is: ", accuracy_score(y_test,y_mnb_pred))

print("The accuracy score of Bernoulli Naive Bayes is: ", accuracy_score(y_test,y_bnb_pred))

#graphical comparison of accuracy 

import seaborn as sns

sns.barplot(x=['Gaussian NB','Multinomial NB','Bernoulli NB'],
            y=[accuracy_score(y_test,y_gnb_pred),accuracy_score(y_test,y_mnb_pred),accuracy_score(y_test,y_bnb_pred)],palette='viridis')