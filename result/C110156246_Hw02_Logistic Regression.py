# C110156246 王皓 
# Hw02_Logistic Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

'''
。Logistic Regression Model
    建立一個Logistic Regression Model 為了繪圖方便
    只需用兩個特徵加入訓練即可
'''
data = pd.read_csv('D:/ALL/NKUST/機器學習/Hw/data/Final_Autism_data_encoded.csv')

X = data[["age", "result"]]
y = data[['Class/ASD']]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)
y_train = y_train.values.ravel()

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

LogReg_model = LogisticRegression(C = 100, 
                           random_state=1, 
                           solver="lbfgs", 
                           multi_class="ovr")
LogReg_model.fit(X_train, y_train)

plot_decision_regions(X_train.values, y_train, 
                      clf = LogReg_model)
plt.xlabel("Age")
plt.ylabel("Result Score")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
