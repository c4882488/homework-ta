from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import pandas as pd

data = pd.read_excel('1.xlsx')
data['price_category'] = pd.cut(data['Y house price of unit area'], 
                                 bins=[-float('inf'), 30.00, 55.00, float('inf')], 
                                 labels=['0', '1', '2'])
X = data[['X2 house age', 'X3 distance to the nearest MRT station']]
Y = data['price_category'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train, y_train)

plot_decision_regions(X_train.values, y_train.values, clf=lr, legend=2)

plt.scatter(X_test['X2 house age'], X_test['X3 distance to the nearest MRT station'], c='yellow', edgecolor='black', alpha=0.5, linewidth=1, marker='o', s=100, label='Test set')

plt.xlabel('House Age')
plt.ylabel('Distance to the nearest MRT station')
plt.title('Logistic Regression Decision Regions')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
