# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
from sklearn.tree import export_graphviz
from IPython.display import SVG
# You may need to install the Python graphviz library. At the command line:
#   pip install graphviz
# You will also need to install the graphviz executables. You can use apt,
# macports, or other installer for your system.
from graphviz import Source
from sklearn import tree

import matplotlib.pyplot as plt
# %%

df = pd.read_csv('data.csv')
df.head()

# %%
feature_cols = [' ROA(C) before interest and depreciation before interest',
' ROA(A) before interest and after tax', ' ROA(B) before interest and depreciation after tax',
' Cash flow rate', ' Net worth/Assets']

X = df[feature_cols]
y = df.Bankrupt
# %%
from sklearn.tree import DecisionTreeClassifier
treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)
treeclf.fit(X, y)
# %%
graph = Source(tree.export_graphviz(treeclf, out_file=None,
                                    feature_names=feature_cols,
                                    class_names=['Not Bankrupt', 'Bankrupt'], filled = True, rounded=True))
graph.format = 'png'
graph.render('dtree_render', view=True)
display(SVG(graph.pipe(format='svg')))

# %%
pd.DataFrame({'feature':feature_cols, 'importance':treeclf.feature_importances_})

# %%
feature_cols1 = [' ROA(C) before interest and depreciation before interest',
' ROA(A) before interest and after tax', ' ROA(B) before interest and depreciation after tax',
' Cash flow rate', ' Net worth/Assets', ' Persistent EPS in the Last Four Seasons', ' Inventory/Working Capital', ]

X = df[feature_cols1]
y = df.Bankrupt

# %%
treeclf1 = DecisionTreeClassifier(max_depth=5, random_state=1)
treeclf1.fit(X, y)

# %%
graph = Source(tree.export_graphviz(treeclf1, out_file=None,
                                    feature_names=feature_cols1,
                                    class_names=['Not Bankrupt', 'Bankrupt'], filled = True, rounded=True))
graph.format = 'png'
graph.render('dtree_render', view=True)
display(SVG(graph.pipe(format='svg')))
# %%
pd.DataFrame({'feature':feature_cols1, 'importance':treeclf1.feature_importances_})
# %%