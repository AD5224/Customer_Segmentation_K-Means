# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r"C:\Users\Aditi\Downloads\Mall_Customers (1).csv")  # Update the path to your dataset

# Display the first few rows of the dataset
print(df.head())

# Data Cleaning
# Check for missing values
print(df.isnull().sum())

# Remove any duplicates if necessary
df.drop_duplicates(inplace=True)

# Feature Selection
# Select relevant features for clustering (Example: Age, Annual Income, Spending Score)
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Data Normalization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the results of the Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Fit the K-Means model with the optimal number of clusters (e.g., from the Elbow method)
optimal_clusters = 5  # Replace this with your chosen number of clusters from the plot
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis', label='Cluster')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()

# Analyze the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(cluster_centers, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
print("Cluster Centers:")
print(centers_df)

# Save the segmented data to a new CSV file
df.to_csv('segmented_customers.csv', index=False)
