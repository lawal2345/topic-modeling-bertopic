# topic-modeling-bertopic

## Project Summary:
1. I built a topic modeling system using BERTopic on the ArXiv dataset.
2. UMAP and HDBSCAN were used for dimensionality reduction and clustering respectively.
3. 161 distinct clusters were identified from ~40,000 research papers.
4. Representation models KeyBERT inspired and MMR were used to create topics for the 161 clusters.

## Pipeline
Abstracts -> Embeddings -> Dimensionality Reduction -> Clustering -> Topic Extraction -> Representation Learning

## Visualizations
### Topic Hierarchy
<img width="1000" height="2600" alt="hierarchichal clustering" src="https://github.com/user-attachments/assets/78e8b4a4-5bc8-4d45-b6e9-546f8ad9f05e" />
Hierarchical clustering showing topic relationships

### Documents and their Topics
<img width="1200" height="750" alt="documents and topics" src="https://github.com/user-attachments/assets/a5eeeb47-28f1-4045-b955-052a6305dfad" />
Documents positioned by their topic assignments
