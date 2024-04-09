import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

#Reading file 
bestBuyLaptops = pd.read_csv('best_buy_laptops_2024.csv')

#The 'offers/priceCurrency' category is all in USD, so it can be deleted.
#Delete the data entry if either features/0/description or features/1/description and brand model is empty.
bestBuyLaptops = bestBuyLaptops.drop('offers/priceCurrency', axis=1)
bestBuyLaptops = bestBuyLaptops.dropna(
    subset=[
        'features/0/description', 
        'features/1/description',
        'brand',
        'model'
    ],
    how = 'any'
)

bestBuyLaptops = bestBuyLaptops.dropna(
   subset = [
       'aggregateRating/ratingValue', 
       'aggregateRating/reviewCount', 
       'offers/price'
    ], 
    how='all'
)

columnsFill = [
    'aggregateRating/ratingValue', 
    'aggregateRating/reviewCount', 
    'offers/price',
    'depth', 
    'width',
]

for column in columnsFill:
    meanValue = bestBuyLaptops[column].mean()
    bestBuyLaptops[column].fillna(meanValue, inplace = True)

print(bestBuyLaptops.head(20))


#Descriptive statistics of each feature and label
describeColumns = [
    'aggregateRating/ratingValue', 
    'aggregateRating/reviewCount', 
    'offers/price', 
    'depth', 
    'width',
]
featuresDescription = bestBuyLaptops[describeColumns].describe()
#print(featuresDescription)


'''下面有問題'''

#Split data
X = bestBuyLaptops[['aggregateRating/ratingValue','brand', 'model','aggregateRating/reviewCount','offers/price', 'depth', 'width']]
y = bestBuyLaptops['aggregateRating/ratingValue']


#Transforming features for brands
encoder = OneHotEncoder()
brand_encoded = encoder.fit_transform(X[['brand']]).toarray()
brand_encoded_df = pd.DataFrame(brand_encoded, columns=[
    f'brand_{int(i)}' for i in range(brand_encoded.shape[1])
    ]
)
X = pd.concat([X.drop('brand', axis=1), brand_encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

#Feature scaling
featureScale = [
    'aggregateRating/ratingValue',
    'aggregateRating/reviewCount',  
    'offers/price',
    'depth',
    'width',
]
scaler = StandardScaler()
X_train[featureScale] = scaler.fit_transform(X_train[featureScale])
X_test[featureScale] = scaler.transform(X_test[featureScale])

logisticRegressModel = LogisticRegression()
logisticRegressModel.fit(X_train[['depth', 'width']], y_train)

plt.figure(figsize=(10, 6))
plt.scatter(X_train['depth'], X_train['width'], c=X_train['offers/price'], cmap='viridis', marker='o', label='Train')

plt.scatter(X_test['depth'], X_test['width'], c=X_test['offers/price'], cmap='viridis', marker='x', label='Test')

w = logisticRegressModel.coef_[0]
b = logisticRegressModel.intercept_
xx = np.linspace(-2, 2, 100)
yy = -w[0] / w[1] * xx - b / w[1]
plt.plot(xx, yy, '-r', label='Decision Boundary')

plt.xlabel('depth')
plt.ylabel('width')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.colorbar(label='offers/price')
plt.grid(True)
plt.show()