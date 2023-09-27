import pandas
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pandas.read_csv("G:/all ml/phishing_site_urls.csv")
df = data.where((pandas.notnull(data)), '')

df['Type'] = df['Label'].apply(lambda x: 1 if x == 'bad' else 0)

X_train, X_test, y_train, y_test = train_test_split(df.Url, df.Type)

v = CountVectorizer()

X_train_count = v.fit_transform(X_train.values)
X_test_count = v.transform(X_test.values)

from sklearn.svm import SVC
clf = SVC(kernel='rbf', random_state = 1)
clf=clf.fit(X_train_count, y_train)
svc_pred = clf.predict(X_test_count)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="gini", max_depth=20)
clf=clf.fit(X_train_count, y_train)
dc_pred = clf.predict(X_test_count)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf = clf.fit(X_train_count,y_train)
nb_pred = clf.predict(X_test_count)

from sklearn.linear_model import SGDClassifier 
clf = SGDClassifier()
clf = clf.fit(X_train_count,y_train)
sgdc_pred = clf.predict(X_test_count)

from sklearn.linear_model import LogisticRegression 
clf = LogisticRegression()
clf = clf.fit(X_train_count,y_train)
lr_pred = clf.predict(X_test_count)

from sklearn.linear_model import RidgeClassifier 
clf = RidgeClassifier()
clf = clf.fit(X_train_count,y_train)
rc_pred = clf.predict(X_test_count)
#from sklearn.neighbors import KNeighborsClassifier
#       ACCURACY SCORE 
print("Accuracy")
print("SVC:\t",metrics.accuracy_score(y_test, svc_pred))
print("DC:\t",metrics.accuracy_score(y_test, dc_pred))
print("NB:\t",metrics.accuracy_score(y_test, nb_pred))
print("SGDC:\t",metrics.accuracy_score(y_test, sgdc_pred))
print("LR:\t",metrics.accuracy_score(y_test, lr_pred))
print("RC:\t",metrics.accuracy_score(y_test, rc_pred))

from sklearn.metrics import precision_score
print('Precision')
print('SVC:\t', precision_score(y_test, svc_pred))
print('DC:\t' , precision_score(y_test, dc_pred))
print('NB:\t' , precision_score(y_test, nb_pred))
print('SGDC:\t', precision_score(y_test, sgdc_pred))
print('LR:\t', precision_score(y_test, lr_pred))
print('RC:\t', precision_score(y_test, rc_pred))

from sklearn.metrics import recall_score
print('Recall')
print('SVM:\t', recall_score(y_test, svc_pred))
print('DC:\t', recall_score(y_test, dc_pred))
print('NB:\t', recall_score(y_test, nb_pred))
print('SGDC:\t', recall_score(y_test, sgdc_pred))
print('LR:\t', recall_score(y_test, lr_pred))
print('RC:\t', recall_score(y_test, rc_pred))

from sklearn.metrics import f1_score
print('F1 Score')
print('SVM:\t', f1_score(y_test, svc_pred))
print('DC:\t', f1_score(y_test, dc_pred))
print('NB:\t', f1_score(y_test, nb_pred))
print('SGDC:\t', f1_score(y_test, sgdc_pred))
print('LR:\t', f1_score(y_test, lr_pred))
print('RC:\t', f1_score(y_test, rc_pred))

from sklearn.metrics import mean_squared_error
from math import sqrt
print('Root Mean Squared Error')
print('SVM:\t', sqrt(mean_squared_error(y_test, svc_pred)))
print('DC:\t', sqrt(mean_squared_error(y_test, dc_pred)))
print('NB:\t', sqrt(mean_squared_error(y_test, nb_pred)))
print('SGDC:\t', sqrt(mean_squared_error(y_test, sgdc_pred)))
print('LR:\t', sqrt(mean_squared_error(y_test, lr_pred)))
print('RC:\t', sqrt(mean_squared_error(y_test, rc_pred)))

#       CONFUSION MATRIX
print("Confusion Matrix")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, svc_pred)
print("SVC:")
print(cf_matrix)
cf_matrix = confusion_matrix(y_test, dc_pred)
print("DC:")
print(cf_matrix)
cf_matrix = confusion_matrix(y_test, nb_pred)
print("NB:")
print(cf_matrix)
cf_matrix = confusion_matrix(y_test, sgdc_pred)
print("SGDC:")
print(cf_matrix)
cf_matrix = confusion_matrix(y_test, lr_pred)
print("LR:")
print(cf_matrix)
cf_matrix = confusion_matrix(y_test, rc_pred)
print("RC:")
print(cf_matrix)
