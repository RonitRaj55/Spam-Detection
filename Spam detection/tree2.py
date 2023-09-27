from sklearn.pipeline import Pipeline
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from csv import writer


data = pandas.read_csv("phishing_site_urls.csv")
df = data.where((pandas.notnull(data)), '')
df.head()

df['bad'] = df['Label'].apply(lambda x: 1 if x == 'bad' else 0)

X_train, X_test, y_train, y_test = train_test_split(df.URL, df.bad)


v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)



model = DecisionTreeClassifier(criterion='gini', max_depth=20)
model = model.fit(X_train_count, y_train)

email = ['''http : / / www . wwwbargins . biz / off .
''']
email_count = v.transform(email)
model.predict(email_count)
X_test_count = v.transform(X_test.values)
model.score(X_test_count, y_test)
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', DecisionTreeClassifier())
])
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
prediction = clf.predict(email)
print(prediction)


if (prediction[0] == 1):
  print('Spam')

else:
  print('Ham')







