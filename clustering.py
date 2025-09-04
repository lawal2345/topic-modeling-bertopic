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
) -> np.ndarray
