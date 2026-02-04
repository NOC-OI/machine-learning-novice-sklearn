---
~
---

### Stacking: classification

import seaborn as sns
penguins = sns.load\_dataset('penguins')

feature\_names = ['bill\_length\_mm', 'bill\_depth\_mm', 'flipper\_length\_mm', 'body\_mass\_g']
penguins.dropna(subset=feature\_names, inplace=True)

species\_names = penguins['species'].unique()

## Define data and targets

X = penguins[feature\_names]

y = penguins.species

## Split data in training and test set

from sklearn.model\_selection import train\_test\_split

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.2, random\_state=5)

print(f'train size: {X\_train.shape}')
print(f'test size: {X\_test.shape}')

from sklearn.ensemble import (
GradientBoostingClassifier,
RandomForestClassifier,
VotingClassifier,
)
from sklearn.gaussian\_process import GaussianProcessClassifier
from sklearn.gaussian\_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier

## training estimators

rf\_clf = RandomForestClassifier(n\_estimators=100, max\_depth=7, min\_samples\_leaf=1, random\_state=5)
gb\_clf = GradientBoostingClassifier(random\_state=5)
gp\_clf = GaussianProcessClassifier(1.0 \* RBF(1.0), random\_state=5)
dt\_clf = DecisionTreeClassifier(max\_depth=5, random\_state=5)

voting\_reg = VotingClassifier([("rf", rf\_clf), ("gb", gb\_clf), ("gp", gp\_clf), ("dt", dt\_clf)])

## fit voting estimator

voting\_reg.fit(X\_train, y\_train)

## lets also train the individual models for comparison

rf\_clf.fit(X\_train, y\_train)
gb\_clf.fit(X\_train, y\_train)
gp\_clf.fit(X\_train, y\_train)
dt\_clf.fit(X\_train, y\_train)

import matplotlib.pyplot as plt

## make predictions

X\_test\_20 = X\_test[:20] # first 20 for visualisation

rf\_pred = rf\_clf.predict(X\_test\_20)
gb\_pred = gb\_clf.predict(X\_test\_20)
gp\_pred = gp\_clf.predict(X\_test\_20)
dt\_pred = dt\_clf.predict(X\_test\_20)
voting\_pred = voting\_reg.predict(X\_test\_20)

print(rf\_pred)
print(gb\_pred)
print(gp\_pred)
print(dt\_pred)
print(voting\_pred)

plt.figure()
plt.plot(gb\_pred,  "o", color="green", label="GradientBoostingClassifier")
plt.plot(rf\_pred,  "o", color="blue", label="RandomForestClassifier")
plt.plot(gp\_pred,  "o", color="darkblue", label="GuassianProcessClassifier")
plt.plot(dt\_pred,  "o", color="lightblue", label="DecisionTreeClassifier")
plt.plot(voting\_pred,  "x", color="red", ms=10, label="VotingRegressor")

plt.tick\_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.ylabel("predicted")
plt.xlabel("training samples")
plt.legend(loc="best")
plt.title("Regressor predictions and their average")

plt.show()

print(f'random forest: {rf\_clf.score(X\_test, y\_test)}')

print(f'gradient boost: {gb\_clf.score(X\_test, y\_test)}')

print(f'guassian process: {gp\_clf.score(X\_test, y\_test)}')

print(f'decision tree: {dt\_clf.score(X\_test, y\_test)}')

print(f'voting regressor: {voting\_reg.score(X\_test, y\_test)}')


