import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("D:\salary.csv")

# data.head(), data.info()

# data.replace('?', pd.NA, inplace=True)

# # Check for missing values
# missing_data = data.isnull().sum()

# # Handling missing data:
# # 1. Drop rows with missing values
# data_dropped = data.dropna()

# # 2. Fill missing categorical values with the mode
# data_filled = data.copy()
# for column in data_filled.select_dtypes(include='object').columns:
#     data_filled[column].fillna(data_filled[column].mode()[0], inplace=True)

# missing_data, data_dropped.shape, data_filled.shape

# # Descriptive statistics for numerical features
# numerical_stats = data.describe()

# # Descriptive statistics for categorical features
# categorical_stats = data.describe(include=['object'])

# numerical_stats, categorical_stats

# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# # Selecting numerical features for scaling
# data = pd.read_csv("D:\salary.csv")
# numerical_features = data.select_dtypes(include=['int64', 'float64']).columns

# # Initializing the StandardScaler
# scaler = StandardScaler()

# # Fitting and transforming the numerical features
# data_scaled = data.copy()
# data_scaled[numerical_features] = scaler.fit_transform(data_scaled[numerical_features])

# # Displaying the first few rows to check the scaling
# data_scaled.head()


# from sklearn.preprocessing import OneHotEncoder

# # Selecting categorical features for encoding
# categorical_features = data.select_dtypes(include=['object']).columns.drop('salary')  # Exclude the label

# # Initializing the OneHotEncoder
# encoder = OneHotEncoder(sparse=False)

# # Fitting and transforming the categorical features
# encoded_features = encoder.fit_transform(data[categorical_features])

# # Creating a DataFrame with the encoded features
# encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# # Concatenating the encoded features with the original DataFrame (excluding the original categorical features)
# data_preprocessed = pd.concat([data_scaled.drop(columns=categorical_features), encoded_features_df, data['salary']], axis=1)

# # Displaying the first few rows to check the encoding
# data_preprocessed.head()

# from sklearn.model_selection import train_test_split

# # Separating the features and the label
# X = data_preprocessed.drop('salary', axis=1)
# y = data_preprocessed['salary']

# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train.shape, X_test.shape, y_train.shape, y_test.shape



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# 假設 data 是您已經預處理好的 DataFrame
# 選擇兩個特徵作為模型的輸入，此處以 'feature1' 和 'feature2' 為例，請根據您的數據集實際情況替換
X = data[['age', 'capital-gain']]
y = data['salary']  # 'Label' 是目標變量列名，根據實際情況替換

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 Logistic Regression 模型
model = LogisticRegression()

# 使用訓練集數據訓練模型
model.fit(X_train, y_train)

# 使用測試集評估模型
score = model.score(X_test, y_test)
print(f'模型準確度: {score:.4f}')
