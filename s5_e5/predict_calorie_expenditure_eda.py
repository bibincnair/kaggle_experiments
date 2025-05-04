import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data_path = "data/s5_e5/"
train_path = data_path + "train.csv"
test_path = data_path + "test.csv"


def plot_correlation_matrix(df):
    """
    Plot the correlation matrix of the DataFrame.
    """
    # label encode sex column
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    

def print_unique_values(df):
    """
    Print unique values for each column in the DataFrame.
    """
    for column in df.columns:
        unique_values = df[column].unique()
        print(f"Column: {column}, Unique Values: {unique_values}")
        
def basic_data_analysis(df):
    """
    Perform basic data analysis on the DataFrame.
    """
    # Dataset dimensions
    print(f"Dataset dimensions: {df.shape}")
    # Data types
    print(f"Data types:\n{df.dtypes}")
    # missing values
    print(f"Missing values:\n{df.isnull().sum()}")
    # basic statistics
    print(f"Basic statistics:\n{df.describe()}")
    # Remove id column and boxplot
    df.drop(columns=['id'], inplace=True)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df)
    plt.title("Boxplot of Features")
    plt.show()
    
def target_variable_analysis(df: pd.DataFrame):
    """Analyze the target variable distribution.
    1. Distribution of calories burned, hist plot and box plot
    2. Check for skewness and kurtosis
    3. Log transformation analysis
    """
    # # Drop id column
    # df.drop(columns=['id'], inplace=True)
    # Encode Sex column
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    
    # Distribution of calories burned
    plt.figure(figsize=(12, 8))
    sns.histplot(df['Calories'], bins=30, kde=True)
    plt.title("Distribution of Calories Burned")
    plt.show()
    
    # Box plot of calories burned
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=df['Calories'])
    plt.title("Boxplot of Calories Burned")
    plt.show()
    # Check for skewness and kurtosis
    skewness = df['Calories'].skew()
    kurtosis = df['Calories'].kurtosis()
    print(f"Skewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f}")
    
    # Print top correlated features with Calories
    corr = df.corr()['Calories'].sort_values(ascending=False)
    print("correlated features with Calories:")
    print(corr)
    
    # Log transformation
    log_cal = np.log1p(df['Calories'])
    plt.figure(figsize=(12, 8))
    sns.histplot(log_cal, bins=30, kde=True, color='orange')
    plt.title("Distribution of Log(1 + Calories Burned)")
    plt.show()
    
    log_skew = log_cal.skew()
    log_kurt = log_cal.kurtosis()
    print(f"Log Skewness: {log_skew:.3f}, Log Kurtosis: {log_kurt:.3f}")
    
    # Q–Q plot to check normality
    import scipy.stats as stats
    plt.figure(figsize=(6, 6))
    stats.probplot(df['Calories'], dist="norm", plot=plt)
    plt.title("Q–Q Plot of Calories Burned")
    plt.show()
    # Log transformation
    
    #pair plot
    plt.figure(figsize=(12, 8))
    sns.pairplot(df, diag_kind='kde')
    plt.title("Pairplot of Features")
    plt.show()
    

if __name__ == "__main__":
    # Load the data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    #print column names
    print("Train Data Columns: ", train_df.columns)

    # Check for missing values in the training data
    print(train_df.isnull().sum())

    # Check for missing values in the test data
    print(test_df.isnull().sum())

    # Display basic statistics of the training data
    print(train_df.describe())

    # Display basic statistics of the test data
    print(test_df.describe())
    
    # print_unique_values(train_df)
    
    basic_data_analysis(train_df)
    plot_correlation_matrix(train_df)
    target_variable_analysis(train_df)
