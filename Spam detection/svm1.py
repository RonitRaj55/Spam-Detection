from sklearn.pipeline import Pipeline
import pandas
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pandas.read_csv("data_12000.csv")
df = data.where((pandas.notnull(data)), '')
#df.head()

df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
#df['blacklisted'] = df['Status'].apply(lambda x: 2 if x == 'blacklisted' else 3)
#print(df.head())
#print(df.Ip)

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam)
#Z_train, Z_test, C_train, C_test = train_test_split(df.Ip, df.blacklisted)

v = TfidfVectorizer()

X_train_count = v.fit_transform(X_train.values)
#print(X_train_count)
#X_train_count.toarray()[:2]
#print(X_train_count)
#Z_train_count = v.fit_transform(Z_train.values)
#Z_train_count.toarray()[:2]
#print(Z_train_count)
'''
print(X_train.values)
print(type(X_train.values))
print(X_train)
print(type(X_train))
'''

model = SVC(kernel='rbf', random_state = 1)
model=model.fit(X_train_count, y_train)

X_test_count = v.transform(X_test.values)
y_pred = model.predict(X_test_count)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test,y_pred)
#accuracy = float(cm.diagonal().sum())/len(y_test)
#print("\nAccuracy Of SVM For The Given Dataset using Confusion Matrix: ", accuracy)

import matplotlib.pyplot as plt
#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cf_matrix)

'''      Printing CONFUSION MATRIX USING SEABORN
import numpy as np
import seaborn as sns
ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, cmap='Blues', fmt='.2%')
ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Ham','Spam'])
ax.yaxis.set_ticklabels(['Ham','Spam'])
## Display the visualization of the Confusion Matrix.
plt.show()
'''
from sklearn.metrics import precision_score
print('Precision: %.3f' % precision_score(y_test, y_pred))

from sklearn.metrics import recall_score
print('Recall: %.3f' % recall_score(y_test, y_pred))

from sklearn.metrics import f1_score
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

email = ['''Subject: Invest in our business The Computron
We are a team of developers that make great custom websites and other software as per the industry needs.
Invest in our compani if you want to be a part of the phenomenon.
Start investing now and be rich in less than an year.
The Computron''']
'''

email_count = v.transform(email)
#model.predict(email_count)
X_test_count = v.transform(X_test)
print(model.score(X_test_count, y_test))

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', DecisionTreeClassifier())
])
model.fit(X_train, y_train)

#print("pipeline: ", clf.score(X_test, y_test))

prediction = model.predict(email_count)
print(prediction)


if (prediction[0] == 1):
  print('Spam')

else:
  print('Ham')
'''