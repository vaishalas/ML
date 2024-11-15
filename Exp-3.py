import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate synthetic dataset (representing patient medical records)
def create_synthetic_data(n_samples=300, random_state=42):
    np.random.seed(random_state)
    # Features: [Age, Blood Pressure, Cholesterol Level]
    data = np.vstack([
        np.random.multivariate_normal([30, 120, 200], [[10, 0, 0], [0, 20, 0], [0, 0, 30]], n_samples),
        np.random.multivariate_normal([60, 140, 250], [[10, 0, 0], [0, 25, 0], [0, 0, 35]], n_samples),
        np.random.multivariate_normal([50, 130, 180], [[8, 0, 0], [0, 18, 0], [0, 0, 25]], n_samples)
    ])
    return data

# Load and preprocess the data
data = create_synthetic_data()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# Apply EM Algorithm (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(data_scaled)

# Compare the results using Silhouette Score
kmeans_silhouette = silhouette_score(data_scaled, kmeans_labels)
gmm_silhouette = silhouette_score(data_scaled, gmm_labels)

print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")
print(f"EM (GMM) Silhouette Score: {gmm_silhouette:.3f}")

# Visualization of the Clusters
def plot_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("Age (scaled)")
    plt.ylabel("Blood Pressure (scaled)")
    plt.show()

# Visualize K-Means and GMM Clustering
plot_clusters(data_scaled, kmeans_labels, "K-Means Clustering")
plot_clusters(data_scaled, gmm_labels, "EM (GMM) Clustering")

# Interpretation of Clusters for Personalized Treatment
def interpret_clusters(data, labels, scaler, method_name):
    # Reverse the scaling for interpretation
    data_original = scaler.inverse_transform(data)
    df = pd.DataFrame(data_original, columns=['Age', 'Blood Pressure', 'Cholesterol'])
    df['Cluster'] = labels

    # Calculate and display the average characteristics of each cluster
    cluster_summary = df.groupby('Cluster').mean()
    print(f"\n{method_name} Clustering - Cluster Interpretation:")
    print(cluster_summary)
    print("\nInterpretation:")
    for cluster_id, row in cluster_summary.iterrows():
        age, bp, chol = row
        print(f"Cluster {cluster_id} patients -")
        if age < 40:
            age_group = "younger adults"
        elif age < 55:
            age_group = "middle-aged adults"
        else:
            age_group = "older adults"
        
        treatment_recommendation = (
            f"Patients in Cluster {cluster_id} are mostly {age_group}, "
            f"with an average blood pressure of {bp:.1f} and cholesterol level of {chol:.1f}. "
        )

        # Suggest general treatment plan based on clusters (hypothetical)
        if chol > 220:
            treatment_recommendation += "A high cholesterol management plan including diet and medication may be recommended."
        elif bp > 135:
            treatment_recommendation += "Blood pressure control through lifestyle changes and possible medication is advisable."
        else:
            treatment_recommendation += "A general wellness and preventative health plan is suggested."

        print(treatment_recommendation)

# Interpret clusters based on K-Means and GMM results
interpret_clusters(data_scaled, kmeans_labels, scaler, "K-Means")
interpret_clusters(data_scaled, gmm_labels, scaler, "EM (GMM)")