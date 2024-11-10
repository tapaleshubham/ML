import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('uber.csv')
df

df.shape
df.info()
df.dtypes
df.describe()

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['year'] = df.pickup_datetime.dt.year
df['month'] = df.pickup_datetime.dt.month
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['day'] = df['pickup_datetime'].dt.day
df['hour'] = df['pickup_datetime'].dt.hour

df = df.drop(columns=['Unnamed: 0', 'key','pickup_datetime'])
df

df.isnull().sum()
#Filling Null Values
columns = ['dropoff_longitude','dropoff_latitude']
for column in columns:
    df.fillna(df[column].median(),axis = 0,inplace = True)

df.isnull().sum()

#Calculating Distance travelled using Haversine Formula
from math import radians, sin, cos, sqrt, asin

def distance_formula(longitude1, latitude1, longitude2, latitude2):
    travel_dist = []
    for pos in range(len(longitude1)):
        lon1, lat1, lon2, lat2 = map(radians, [longitude1[pos], latitude1[pos], longitude2[pos], latitude2[pos]])
        dist_lon = lon2 - lon1
        dist_lat = lat2 - lat1
        
        a = sin(dist_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dist_lon / 2) ** 2
        
        # Radius of Earth in kilometers (6371)
        c = 2 * asin(sqrt(a)) * 6371
        travel_dist.append(c)
        
    return travel_dist

df['dist_travel_km'] = distance_formula(df.pickup_longitude.to_numpy(), df.pickup_latitude.to_numpy(), 
                                        df.dropoff_longitude.to_numpy(), df.dropoff_latitude.to_numpy())


fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(18, 8))
axes = axes.flatten()
for i, column in enumerate(df.columns):
    df[column].plot(kind='box', ax=axes[i])
    axes[i].set_title(column)
plt.tight_layout()
plt.show()

def remove_outlier(df1 , col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1-1.5*IQR
    upper_whisker = Q3+1.5*IQR
    df[col] = np.clip(df1[col] , lower_whisker , upper_whisker)
    return df1
def treat_outliers_all(df1 , col_list):
    for c in col_list:
        df1 = remove_outlier(df , c)
    return df1

df = treat_outliers_all(df , df.iloc[: , 0::])

fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(18, 8))
axes = axes.flatten()
for i, column in enumerate(df.columns):
    df[column].plot(kind='box', ax=axes[i])
    axes[i].set_title(column)
plt.tight_layout()
plt.show()


correlation_matrix = df.corr()
fig = px.imshow(
    correlation_matrix,
    text_auto=True,  
    color_continuous_scale='RdBu_r',  
    title='Correlation Matrix Heatmap'
)
fig.update_layout(
    title_text='Correlation Matrix Heatmap',
    title_x=0.5,
    width=1080,  
    height=700  
)
fig.show()


X = df.drop(columns=['fare_amount'])  
y = df['fare_amount']
X
y

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
# Implement Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)


# Implement Ridge Regression
ridge_model = Ridge(alpha=3.2)  
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)


# Implement Lasso Regression
lasso_model = Lasso(alpha=0.1)  
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)


def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - R2 Score: {r2:.2%}, RMSE: {rmse:.2f}")


evaluate_model(y_test, y_pred_linear, "Linear Regression")
evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
evaluate_model(y_test, y_pred_lasso, "Lasso Regression")


# Visualizing the models' predictions vs actual values
plt.figure(figsize=(18, 6))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.title('Linear Regression')
plt.xlabel('Actual Fares')
plt.ylabel('Predicted Fares')

# Ridge Regression
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_ridge, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.title('Ridge Regression')
plt.xlabel('Actual Fares')
plt.ylabel('Predicted Fares')

# Lasso Regression
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_lasso, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.title('Lasso Regression')
plt.xlabel('Actual Fares')
plt.ylabel('Predicted Fares')

plt.tight_layout()
plt.show()
