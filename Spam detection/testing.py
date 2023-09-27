import pandas
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

data = pandas.read_csv("phishing_site_urls.csv")
df = data.where((pandas.notnull(data)), '')
data1 = pandas.read_csv("data_12000.csv")
df1 = data1.where((pandas.notnull(data1)), '')
df1=df1.head()
df=df.head()
df['bad'] = df['Label'].apply(lambda x: 1 if x == 'bad' else 0)
df1['spam'] = df1['Category'].apply(lambda x:1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(df.URL, df.bad)
X1_train, X1_test, y1_train, y1_test = train_test_split(df1.Message, df1.spam)
v = TfidfVectorizer()
X_train_count = v.fit_transform(X_train.values)
X1_train_count = v.fit_transform(X1_train.values)
X_test_count = v.transform(X_test.values)
X1_test_count = v.transform(X1_test.values)

model = SVC(kernel='rbf', probability=True)
model = model.fit(X1_train_count, y1_train)

y1_pred = model.predict(X1_test_count)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(1), random_state=1) #5,2  2,5,2 7,4 1,1
clf = clf.fit(X_train_count, y_train)
mlp_pred = clf.predict(X_test_count)


print("Accuracy for MLP:\t",metrics.accuracy_score(y_test, mlp_pred))
print("Accuracy For SVM:",metrics.accuracy_score(y1_test, y1_pred))


from sklearn.metrics import precision_score

print('Precision For MLP:\t', precision_score(y_test, mlp_pred))
print('Precision For SVM: %.3f' % precision_score(y1_test, y1_pred))


from sklearn.metrics import recall_score

print('Recall Score For MLP:\t', recall_score(y_test, mlp_pred))
print('Recall Score For SVM: %.3f' % recall_score(y1_test, y1_pred))

from sklearn.metrics import f1_score

print('F1 Score For MLP:\t', f1_score(y_test, mlp_pred))
print('F1 Score For SVM: %.3f' % f1_score(y1_test, y1_pred))

from math import sqrt

from sklearn.metrics import mean_squared_error

print('Root Mean Squared Error')
print('MLP:\t', sqrt(mean_squared_error(y_test, mlp_pred)))

#       CONFUSION MATRIX
print("Confusion Matrix")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, mlp_pred)
print("Confusion Matrix for MLP:")
print(cf_matrix)
#Generate the confusion matrix For SVM
cf_matrix1 = confusion_matrix(y1_test, y1_pred)
print("Confusion Matrix For SVM:")
print(cf_matrix1)
