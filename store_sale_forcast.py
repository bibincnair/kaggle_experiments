import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    store = pd.read_csv("stores.csv")
    transaction = pd.read_csv("transactions.csv")
    oil = pd.read_csv("oil.csv")
    holidays_events = pd.read_csv("holidays_events.csv")
    
    # Convert any infinite values to NaN and then handle them
    train.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    test.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    oil.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    holidays_events.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
        
    # Ensure proper date conversion
    train['date'] = pd.to_datetime(train['date'], errors='coerce')
    test['date'] = pd.to_datetime(test['date'], errors='coerce')
    oil['date'] = pd.to_datetime(oil['date'], errors='coerce')
    holidays_events['date'] = pd.to_datetime(holidays_events['date'], errors='coerce')
    return train, test, store, transaction, oil, holidays_events


def print_data_info(train, test, store, transaction, oil):
    print(f"train info : {train.info()}, train head : {train.head()}")
    print(f"test info : {test.info()}, test head : {test.head()}")
    print(f"store info : {store.info()}, store head : {store.head()}")
    print(
        f"transaction info : {transaction.info()}, transaction head : {transaction.head()}"
    )
    print(f"oil info : {oil.info()}, oil head : {oil.head()}")
    # Basic EDA
    # Check for missing values
    print(f"train missing values : {train.isnull().sum()}")
    print(f"test missing values : {test.isnull().sum()}")
    print(f"store missing values : {store.isnull().sum()}")
    print(f"transaction missing values : {transaction.isnull().sum()}")
    print(f"oil missing values : {oil.isnull().sum()}")

    # Descibe the data
    print(f"train describe : {train.describe()}")
    print(f"test describe : {test.describe()}")
    print(f"store describe : {store.describe()}")
    print(f"transaction describe : {transaction.describe()}")
    print(f"oil describe : {oil.describe()}")


def print_missing_values(train, test, store, transaction, oil):
    print(f"train missing values : {train.isnull().sum()}")
    print(f"test missing values : {test.isnull().sum()}")
    print(f"store missing values : {store.isnull().sum()}")
    print(f"transaction missing values : {transaction.isnull().sum()}")
    print(f"oil missing values : {oil.isnull().sum()}")


def merge_data(train, test, stores, transaction, oil, holiday_events):
    oil['dcoilwtico'] = oil['dcoilwtico'].ffill()
    train = train.merge(stores, on="store_nbr", how="left")
    test = test.merge(stores, on="store_nbr", how="left")
    train = train.merge(oil, on="date", how="left")
    test = test.merge(oil, on="date", how="left")
    train = train.merge(holiday_events, on="date", how="left")
    test = test.merge(holiday_events, on="date", how="left")
    
    earthquake_end_date = datetime(2016, 4, 16) + pd.DateOffset(days=14)
    train = train[~((train['date'] > datetime(2016, 4, 16)) & (train['date'] <= earthquake_end_date))]


    print(train.head())
    print(test.head())
    return train, test

def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

def feature_engineering(train, test):
    # Feature Engineering
    train["year"] = train["date"].dt.year
    train["month"] = train["date"].dt.month
    train["day"] = train["date"].dt.day
    train["day_of_week"] = train["date"].dt.dayofweek

    test["year"] = test["date"].dt.year
    test["month"] = test["date"].dt.month
    test["day"] = test["date"].dt.day
    test["day_of_week"] = test["date"].dt.dayofweek
    return train, test


def train_model(train, test):
    print("Training the model...")
    features = ['store_nbr', 'family', 'onpromotion', 'year', 'month', 'day', 'day_of_week', 'dcoilwtico']
    X = train[features]
    y = train['sales']
    X = pd.get_dummies(X, columns=['family'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, verbose=3, n_jobs=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    X_test = test[features]
    X_test = pd.get_dummies(X_test, columns=['family'])
    X_test = X_test.reindex(columns = X_train.columns, fill_value=0)
    test['sales'] = model.predict(X_test)
    submission = test[['id', 'sales']]
    submission.to_csv('submission.csv', index=False)
    rmsle_value = rmsle(y_val, y_pred)
    
    
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    metrics = {'MAE': mae, 'R²': r2, 'RMSE': rmse}
    names = list(metrics.keys())
    values = list(metrics.values())
    plt.figure(figsize=(10, 5))

    plt.bar(names, values, color=['blue', 'green', 'red'])
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Test Metrics: MAE, R², and RMSE')

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

    plt.savefig("metrics.png")
    


# main function
def main():
    train, test, store, transaction, oil, holiday_events = load_data()
    # print_data_info(train, test, store, transaction, oil)
    # print_missing_values(train, test, store, transaction, oil)
    train, test = merge_data(train, test, store, transaction, oil, holiday_events)
    train, test = feature_engineering(train, test)
    train_model(train, test)

# python boilerplate
if __name__ == "__main__":
    main()
