import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns

df = pd.read_csv("your_dataset.csv") print(df.head(), "\n", df.tail()) print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}") print(df.dtypes) print(df.isnull().sum()) print(df.describe())

missing_values_before = df.isnull().sum().sum() for col in df.columns: if df[col].dtype in ['int64', 'float64']: df[col].fillna(df[col].mean(), inplace=True) else: df[col].fillna(df[col].mode()[0], inplace=True) missing_values_after = df.isnull().sum().sum() print(f"Missing values before: {missing_values_before}, after: {missing_values_after}")

df = df.drop_duplicates() print(f"Duplicate rows: {df.duplicated().sum()}")

for col in df.select_dtypes(include=['float64', 'int64']).columns //Detect and handle outliers q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75) iqr = q3 - q1 lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr df[col] = np.clip(df[col], lower_bound, upper_bound)

// Graphical Visualisation //Distribution Analysis via histogram for col in df.select_dtypes(include=['float64', 'int64']).columns: plt.figure(figsize=(8, 4)) sns.histplot(df[col], kde=True) plt.title(f'Distribution - {col}') plt.show()

//Box Plot for col in df.select_dtypes(include=['float64', 'int64']).columns: plt.figure(figsize=(8, 4)) sns.boxplot(x=df[col]) plt.title(f'Box Plot - {col}') plt.show()

//Line PLot for col in df.select_dtypes(include=['float64', 'int64']).columns: plt.figure(figsize=(8, 4)) plt.plot(df[col]) plt.title(f'Line Plot - {col}') plt.xlabel('Index') plt.ylabel(col) plt.show()

//Scatter PLot if len(df.select_dtypes(include=['float64', 'int64']).columns) >= 2: plt.figure(figsize=(8, 4)) sns.scatterplot(x=df[df.select_dtypes(include=['float64', 'int64']).columns[0]], y=df[df.select_dtypes(include=['float64', 'int64']).columns[1]]) plt.title('Scatter Plot') plt.show()
