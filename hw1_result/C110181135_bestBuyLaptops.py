import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
   subst = [
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
    'width'
]

for columns in columnsFill:
    meanValue = bestBuyLaptops[columnsFill].mean()
    bestBuyLaptops[columnsFill].fillna(meanValue, inplace = True)

#print(bestBuyLaptops.head(5))


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

#Split data
X = bestBuyLaptops.drop(columns=['aggregateRating/ratingValue'])
y = bestBuyLaptops['aggregateRating/ratingValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Transforming features for brands
le = LabelEncoder()
X_train['brand'] = le.fit_transform(X_train['brand'])
X_test['brand'] = le.transform(X_test['brand'])

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




