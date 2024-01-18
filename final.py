import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def read_clean_transpose_data(file_path):
    """
    Read, clean, and transpose the input data.

    Parameters:
    - file_path (str): Path to the CSV file containing the data.

    Returns:
    - original_data (pd.DataFrame): Original data as read from the file.
    - cleaned_data (pd.DataFrame): Cleaned data with numeric columns.
    - transposed_data (pd.DataFrame): Transposed data for further analysis.
    """
    # Load data
    original_data = pd.read_csv(file_path)

    # 1. Data Preprocessing
    # Convert relevant columns to numeric
    numeric_columns = ['Access to electricity (% of population) [EG.ELC.ACCS.ZS]' ,
                        'Agricultural land (% of land area) [AG.LND.AGRI.ZS]' ,
                        'Agricultural methane emissions (% of total) [EN.ATM.METH.AG.ZS]' ,
                        'Arable land (% of land area) [AG.LND.ARBL.ZS]' ,
                        'CO2 emissions from electricity and heat production, total (% of total fuel combustion) [EN.CO2.ETOT.ZS]']

    original_data[numeric_columns] = original_data[numeric_columns].apply(pd.to_numeric ,
                                                                          errors = 'coerce')

    # Drop unnecessary columns for clustering and fitting
    cleaned_data = original_data[numeric_columns]

    # Handle any remaining missing values
    cleaned_data = cleaned_data.dropna()

    # Transpose data for further analysis
    transposed_data = cleaned_data.T

    return original_data , cleaned_data , transposed_data


def linear_model(x , a , b):
    """
        Compute the linear model based on the equation: y = a * x + b.

        Parameters:
        - x (array-like): Independent variable values.
        - a (float): Slope parameter.
        - b (float): Intercept parameter.

        Returns:
        - array-like: Dependent variable values based on the linear model.
        """
    return a * x + b


def clusterVisualization():
    """
        Visualize clustering analysis by creating a scatter plot of Arable Land vs. CO2 Emissions
        with different colors representing different clusters and red markers for cluster centers.

        The function uses data from the global variables: 'data', 'kmeans'.

        Parameters:
        None

        Returns:
        None
        """
    plt.scatter(data['Arable land (% of land area) [AG.LND.ARBL.ZS]'] , data[
        'CO2 emissions from electricity and heat production, total (% of total fuel combustion) [EN.CO2.ETOT.ZS]'] ,
                c = data['Cluster'] , cmap = 'viridis')
    plt.scatter(kmeans.cluster_centers_[: , 3] , kmeans.cluster_centers_[: , 4] ,
                s = 300 , c = 'red' , marker = 'X' ,
                label='Cluster Centers')
    plt.xlabel('Arable Land (% of Land Area)')
    plt.ylabel('CO2 Emissions from Electricity and Heat Production (% of Total)')
    plt.title('Clustering Analysis')
    plt.legend()
    plt.show()


def curvePlot():
    """
        Visualize linear model fitting with confidence intervals by creating a scatter plot
        of Arable Land vs. CO2 Emissions, fitted linear model, and 95% confidence interval.

        The function uses data from the global variables: 'x_data', 'y_data', 'linear_model', 'params',
        'lower_bound', and 'upper_bound'.

        Parameters:
        None

        Returns:
        None
        """
    plt.scatter(x_data , y_data , label = 'Original Data')
    plt.plot(x_data , linear_model(x_data , *params) , color = 'red' ,
             label = 'Fitted Linear Model')
    plt.fill_between(x_data , linear_model(x_data , *lower_bound) ,
                     linear_model(x_data , *upper_bound) , color = 'grey' ,
                     alpha = 0.2 , label = '95% Confidence Interval')
    plt.xlabel('Arable Land (% of Land Area)')
    plt.ylabel('CO2 Emissions from Electricity and Heat Production (% of Total)')
    plt.title('Linear Model Fitting with Confidence Intervals')
    plt.legend()
    plt.show()


# Call the function to get the data
data , data_for_clustering , transposed_data = \
    read_clean_transpose_data('35f7cb22-a76b-4e10-9c3d-eadd3a486824_Data.csv')

# 2. Clustering Analysis
# Normalize data
normalized_data = (data_for_clustering - data_for_clustering.mean()) / data_for_clustering.std()

# Apply k-means clustering
kmeans = KMeans(n_clusters = 3 , random_state = 42)
cluster_labels = kmeans.fit_predict(normalized_data)

# Evaluate clustering performance using silhouette score
silhouette_avg = silhouette_score(normalized_data , cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')

# Assign cluster labels to the original DataFrame
data['Cluster'] = np.nan  # Create a new column with NaN values
data.loc[data_for_clustering.index , 'Cluster'] = cluster_labels

# 3. Model Fitting
x_data = data['Arable land (% of land area) [AG.LND.ARBL.ZS]']
y_data = \
    data['CO2 emissions from electricity and heat production, total (% of total fuel combustion) [EN.CO2.ETOT.ZS]']

# Drop rows with NaN values in x or y
non_nan_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
x_data = x_data[non_nan_indices]
y_data = y_data[non_nan_indices]

# Fit the linear model
params , covariance = curve_fit(linear_model , x_data , y_data)

# Calculate confidence intervals
sigma = np.sqrt(np.diag(covariance))
lower_bound = params - 1.96 * sigma
upper_bound = params + 1.96 * sigma

# 4. Visualization of Clusters
clusterVisualization()

# 5. Visualization of Fitted Model with Confidence Intervals
curvePlot()

# Add predictions for the years 2024, 2025, and 2026
future_years = np.array([2024 , 2025 , 2026])
predicted_values = linear_model(future_years , *params)

print(future_years)
print(predicted_values)
