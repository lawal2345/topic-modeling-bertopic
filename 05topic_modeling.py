"""
Topic modeling is done with BERTopic, including KeyBERT-inspired and MMR
"""

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from umap import UMAP
from hdbscan import HDBSCAN
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from copy import deepcopy
