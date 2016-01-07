__author__ = 'shubhamkumar'
from sklearn.tree import DecisionTreeClassifier as RFC
import pandas as pd
reader = pd.read_csv('train.csv')
out = reader['hand'].values
del reader['hand']
values = reader.values
test = pd.read_csv('test.csv')
ids = test['id']
del test['id']
te_val = test.values
k = RFC()
X = k.fit(values, out)
y = k.predict(te_val)
print(max(y))
final = pd.DataFrame()
final['id'] = ids
final['hand'] = y
final.to_csv('Submission1.csv', index=False)