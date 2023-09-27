import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pandas.read_csv("phishing_site_urls.csv")
df = data.where((pandas.notnull(data)), '')
df.head()

df['bad'] = df['Label'].apply(lambda x: 1 if x == 'bad' else 0)

X_train, X_test, y_train, y_test = train_test_split(df.URL, df.bad)

v = TfidfVectorizer()

X_train_count = v.fit_transform(X_train.values)
X_test_count = v.transform(X_test.values)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(1), random_state=1) #5,2  2,5,2 7,4 1,1
clf=clf.fit(X_train_count, y_train)
mlp_pred = clf.predict(X_test_count)

print("Accuracy")
print("MLP:\t",metrics.accuracy_score(y_test, mlp_pred))

from sklearn.metrics import precision_score
print('Precision')
print('MLP:\t', precision_score(y_test, mlp_pred))

from sklearn.metrics import recall_score
print('Recall')
print('MLP:\t', recall_score(y_test, mlp_pred))

from sklearn.metrics import f1_score
print('F1 Score')
print('MLP:\t', f1_score(y_test, mlp_pred))

from sklearn.metrics import mean_squared_error
from math import sqrt
print('Root Mean Squared Error')
print('MLP:\t', sqrt(mean_squared_error(y_test, mlp_pred)))

#       CONFUSION MATRIX
print("Confusion Matrix")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, mlp_pred)
print("MLP:")
print(cf_matrix)