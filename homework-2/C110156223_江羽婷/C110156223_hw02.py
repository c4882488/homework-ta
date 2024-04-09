import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np

data = pd.read_csv('processed_dataset.csv')

X = data[['brand', 'aggregateRating/reviewCount']]
y = data['aggregateRating/ratingValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=419)

# Logistic Regression 模型
model = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
model.fit(X_train, y_train)

# 繪製決策邊界
plot_decision_regions(X_train.values, y_train.values, clf=model, legend=2)
plt.xlabel('Brand')
plt.ylabel('Review Count')
plt.title('Logistic Regression')
plt.show()