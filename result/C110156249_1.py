import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions

# 讀取資料
data = pd.read_csv("Credit card transactions - India.csv")

# 將非數值的特徵進行編碼
label_encoder = LabelEncoder()
data['Card Type'] = label_encoder.fit_transform(data['Card Type'])
data['Exp Type'] = label_encoder.fit_transform(data['Exp Type'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# 選擇兩個特徵作為模型的輸入
X = data[['Card Type', 'Exp Type']].values
y = data['Gender'].values

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化 Logistic Regression 模型
lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')

# 訓練模型
lr.fit(X_train, y_train)

# 繪製決策邊界
plot_decision_regions(X_train, y_train, clf=lr, legend=2)

# 加上標題與軸標籤
plt.xlabel('Card Type')
plt.ylabel('Exp Type')
plt.title('Logistic Regression Decision Boundary')

# 顯示圖形
plt.show()
