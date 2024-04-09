import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 讀取數據
data = pd.read_csv('best_buy_laptops_2024.csv')
print(data.isnull().sum())

# 處理缺失值
data_no_nan = data.fillna(round(data.mean(),2))

# 移除沒有用的類別特徵和其他不需要的列
data_processed = data_no_nan.drop(columns=['offers/priceCurrency', 'model', 'features/0/description', 'features/1/description'])

# 描述性統計
print(data_processed.describe())

# 類別特徵轉換
brand_map={'Acer':1,'Alienware':2,'ASUS':3,'Dell':4,'GIGABYTE':5,'HP':6,'HP OMEN':7,'Lenovo':8,'LG':9,'Microsoft':10,'MSI':11,'Razer':12,'Samsung':13,'Thomson':14}
data_processed['brand']=data_processed['brand'].map(brand_map)

def rating_map(rating):
    if rating >= 4.0:
        return '5'
    elif rating >= 3.0:
        return '4'
    elif rating >= 2.0:
        return '3'
    elif rating >= 1.0:
        return '2'
    else:
        return '1'
    
data_processed['aggregateRating/ratingValue'] = data_processed['aggregateRating/ratingValue'].map(rating_map)

# 特徵縮放 
scaler = StandardScaler()
numeric_features = ['aggregateRating/reviewCount', 'offers/price', 'depth', 'width']
data_processed[numeric_features] = scaler.fit_transform(data_processed[numeric_features])

# 資料分割
X = data_processed.drop(columns=['aggregateRating/ratingValue'])
y = data_processed['aggregateRating/ratingValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=419)
print(len(X_train), len(X_test))

# 保存處理後的數據
data_processed.to_csv('processed_dataset.csv', index=False)
