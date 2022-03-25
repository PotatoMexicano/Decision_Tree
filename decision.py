import os
os.environ["PATH"] += os.pathsep + 'E:/Program Files/Graphviz/bin'


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz

clf = DecisionTreeClassifier(random_state=0, max_depth=10, criterion='entropy')

iris = load_iris()

x = iris.data
y = iris.target

model = clf.fit(x, y)

viz = dtreeviz(clf, x, y, target_name="target", feature_names=iris.feature_names, class_names=list(iris.target_names))
viz.save("decision_tree.svg")