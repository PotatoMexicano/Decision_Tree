import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz


import os
os.environ["PATH"] += os.pathsep + 'E:/Program Files/Graphviz/bin'


# Sol       1 
# Nuvens    2
# Chuva     3

# Ameno     1
# Fresco    2
# Quente    3

# Normal    1
# Elevada   2

# Fraco     1
# Forte     2

data = pd.read_excel('./base.xlsx', sheet_name="Planilha2")

X = data.iloc[:,:-1]
Y = data.iloc[:, -1]

clf = DecisionTreeClassifier(random_state=0, max_depth=50)
model = clf.fit(X,Y)

viz = dtreeviz(tree_model=clf, x_data=X, y_data=Y, target_name="target", feature_names=data.columns, class_names=dict({0:"Não", 1:"Sim"}))
viz.save("decision_tree.svg")