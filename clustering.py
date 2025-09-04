```python
"""
clustering.py: text clustering for topic modeling
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class DocumentClusterer:
  def __init__(
    self,
    embedding_model_name: str = 'thenlper/gte-small', # sentence transformer model
    n_components: int = 5, # number of dimensions to be reduced to
    min_cluster_size: int = 50,
    random_state: int = 42
  ):
    self.embedding_model = SentenceTransformer(embedding_model_name)
    self.n_components = n_components
    self.min_cluster_size = min_cluster_size
    self.random_state = random_state
    # The following will be initialized later
    self.umap_model = None
    self.hdbscan_model = None
    self.embeddings = None
    self.reduced_embeddings = None
    self.clusters = None
  
  def generate_embeddings(
    self,
    texts: List[str],
    show_progress: bool = True
  ) -> np.ndarray:
    """
    Args:
          texts: list of abstracts
          show_progress: show progress bar
  
    Returns:
            Array of embeddings with shape (n_documents, embedding_dim)
    """
    self.embeddings = self.embedding_model.encode(texts, show_progress_bar=show_progress)
    print(f"Generated embeddings with shape: {self.embeddings.shape}")
    return self.embeddings
  
  def reduce_dimensions(
    self,
    embeddings: Optional[np.ndarray] = None
  ) -> np.ndarray:
    """UMAP is great for dimensionality reduction because it preserves the local and global structure, and cosine similarity works well with normalized text embeddings.
    Reduced embeddings have the shape (n_documents, n_components)"""
    print(f"Reducing dimensions from {embeddings.shape[1] to {self.ncomponents}...")
    
    self.umap_model = UMAP(
      n_components=self.n_components,
      min_dist=0.0,
      metric='cosine',
      random_state=self.random_state
    )
  
    self.reduced_embeddings = self.umap_model.fit_transform(embeddings)
    print(f"Reduced to shape: {self.reduced_embeddings.shape}")
    return self.reduced_embeddings
  
  def cluster_documents(self, reduced_embeddings: Optional[np.ndarray] = None) -> np.ndarray:
    """Clustering using HDBSCAN as no need to specify number of clusters, handles varying cluster densities, and identifies outliers"""
    if reduced_embeddings is None:
      reduced_embeddings = self.reduced_embeddings
  
    print(f"Clustering {len(reduced_embeddings)} documents...")
  
    self.hdbscan_model = HDBSCAN(
      min_cluster_size=self.min_cluster_size, metric='euclidean', cluster_selection_method='eom'
    )
  
    self.clusters = self.hdbscan_model.fit_predict(reduced_embeddings)
    n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
    n_outliers = (self.clusters == -1).sum()
          
    print(f"Found {n_clusters} clusters with {n_outliers} outliers")
    return self.clusters
  
  def fit(self, texts: List[str]) -> 'DocumentClusterer':
    """Clustering pipeline"""
    self.generate_embeddings(texts)
    self.reduce_dimensions()
    self.cluster_documents()
    return self
  
  def visualize_clusters_2d(
    self, texts: Optional[List[str]] = None, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)
  ) -> None:
    """texts: Optional text labels, save_path: path to save figure"""
    
    # Reduce to 2D for visualization
    if self.reduced_embeddings.shape[1] != 2:
      print("Reducing to 2d for visualization")
      embeddings_2d = UMAP(
        n_components=2, min_dist=0.0, metric='cosine', random_state=self.random_state
      ).fit_transform(self.embeddings)
    else:
      embeddings_2d = self.reduced_embeddings
  
    df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df["cluster"] = self.clusters
  
    # Separate outliers from clusters
    clusters_df = df[df.cluster != -1]
    outliers_df = df[df.cluster == -1]
  
    plt.figure(figsize=figsize)
    # Plot outliers with grey
    if len(outliers_df) > 0:
      plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey", label=f"Outliers ({len(outliers_df)})")
  
    # Plot clusters with colors
    scatter = plt.scatter(clusters_df.x, clusters_df.y, c=clusters_df.cluster.astype(int), alpha=0.6, s=5, cmap='tab20b')
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f"Document Clusters (n={len(set(self.clusters))-1})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
          
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
    else:
        plt.show()
  
  
  def get_cluster_characteristics(self) -> pd.DataFrame:
    """Returns datadrame with cluster sizes and percentages"""
    cluster_counts = pd.Series(self.clusters).value_counts().sort_index()
  
    stats = pd.DataFrame({
        'Cluster': cluster_counts.index,
        'Size': cluster_counts.values,
        'Percentage': (cluster_counts.values / len(self.clusters) * 100).round(2)
    })
  
    # Outlier labels
    stats['Label'] = stats['Cluster'].apply(lambda x: 'Outliers' if x == -1 else f'Cluster {x}')
    return stats
  
  
  def inspect_cluster(self, cluster_id: int, texts: List[str], n_samples: int = 3) -> List[str]:
    """Inspect sample documents from a specific cluster. cluster_id is the id of the cluster, texts are the documents, and n_samples are the number of samples to return"""
  
    cluster_indices = np.where(self.clusters == cluster_id)[0]
  
    if len(cluster_indices) == 0:
      return [f"No documents found in cluster {cluster_id}"]
    # Sample random documents from the cluster
    sample_size = min(n_samples, len(cluster_indices))
    sample_indices = np.random.choice(
      cluster_indices, size=sample_size, replace=False)
  
    samples = []
    for idx in sample_indices:
      text = texts[idx][:300] + "..." if len(texts[idx]) > 300 else texts[idx]
      samples.append(text)
  
    return samples
    )

def run_clustering_example():
  ""Example of how to use the DocumentClusterer class"""

  from datasets import load_dataset
  print("Arxiv dataset loaded")
  dataset = load_dataset("maartengr/arxiv_nlp")["train"]
  abstracts = dataset["Abstracts"]
  titles = dataset["Titles"]

  # Initialize and run clustering
  clusterer = DocumentClusterer(
      embedding_model_name='thenlper/gte-small',
      n_components=5,
      min_cluster_size=50
  )
  clusterer.fit(abstracts) # to fit the model

  # Get stats
  stats = clusterer.get_cluster_statistics()
  print("\nClustering Stats")
  print(stats)

  # Inspect a cluster
  print("\nSample documents from Cluster 0")
  samples = clusterer.inspect_cluster(0, abstracts, n_samples=3)
  for i, sample in enumerate(samples, 1):
      print(f"\nDocument {i}:")
      print(sample)

  # Visualize clusters
  clusterer.visualize_clusters_2d(save_path="clusters.png")
  return clusterer

if __name__ = "__main__":
    clusterer = run_clustering_example()
