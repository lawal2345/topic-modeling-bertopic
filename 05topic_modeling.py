```python
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

class TopicModeler:
    def __init__(
        self,
        embedding_model_name: str = 'thenlper/gte-small',
        n_components: int = 5,
        min_cluster_size: int = 50,
        random_state: int = 42,
        verbose: bool = True # show progress info
    ):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Initialize UMAP for dim. reduction
        self.umamp_model = UMAP(
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=random_state
        )

        # Initialize clustering HDBSCAN
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean'
            cluster_selection_method='eom'
        )

        # Initialize BERTopic with custom models
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model, umap_model=self.umap_model, hdbscan_model=self.hdbscan_model, verbose=verbose
        )  
        
        # Store original representations for comparison
        self.original_topics = None
        self.documents = None
        self.embeddings = None

    def fit(self, documents: List[str]) -> 'TopicModeler':
        """Fit the topic model on clusters. Arguments are the text abstracts"""
        print(f"Fitting topic model on {len(documents)} documents...")

        # Generate embeddings
        self.documents = documents
        self.embeddings = self.embedding_model.encode(documents, show_progress_bar=True)

        # Fit bertopic
        self.topic_model.fit(documents, self.embeddings)

        self.original_topics = deepcopy(self.topic_model.topic_representations_)
        # Print basic statistics
        topic_info = self.topic_model.get_topic_info()
        n_topics = len(topic_info) - 1  # Exclude outlier topic -1
        print(f"Found {n_topics} topics")

        return self

    def apply_keybert_representation(self) -> pd.DataFrame
        """Applying keybert-inspired to improve topic keywords"""
        representation_model = KeyBERTInspired()
        self.topic_model.update_topics(self.documents, representation=representation_model)
        return self.calculate_topic_differences("KeyBERT-Inspired")

    def apply_mmr_representation(self, diversity: float = 0.5) -> pd.DataFrame:
        """Applying MMR for diverse topic keywords. MMR balances relevance (similarity to topic) and diversity (redundant keyword reduction)"""
        representation_model = MaximalMarginalRelevance(diversity=diversity)
        self.topic_model.update_topics(self.documents, representation_model=representation_model)
        return self.calculate_topic_differences(f"MMR (diversity={diversity})")

    def apply_generative_representation(self, model_name: str = 'google/flan-t5-small') -> pd.DataFrame:
        """
        Using a generative model to create human-readable topic descriptions.
        """
        print(f"\nApplying generative representation with {model_name}...")
        
        # Define prompt template for the LLM
        prompt = """I have a topic that contains the following documents:
        [DOCUMENTS]
  
        The topic is described by the following keywords: '[KEYWORDS]'.
        Based on the documents and keywords, what is this topic about? Give a concise label."""
        
        # Initialize generator
        generator = pipeline('text2text-generation', model=model_name)
        
        # Create representation model
        representation_model = TextGeneration(
            generator, 
            prompt=prompt, 
            doc_length=50,  # Max tokens per document
            tokenizer="whitespace"
        )
        
        # Update topics
        self.topic_model.update_topics(
            self.documents,
            representation_model=representation_model
        )
        return self._calculate_topic_differences(f"Generative ({model_name})")

    def _calculate_topic_differences(
        self, method_name: str, n_topics: int = 5, n_words: int = 5
    ) -> pd.DataFrame:
        """
        Calculate differences between original and updated topic representations.
        Args:
            method_name: Name of the representation method
            n_topics: Number of topics to compare
            n_words: Number of words per topic to show
        """
        df_data = []
        
        for topic in range(n_topics):
            # Get original words
            if topic in self.original_topics:
                og_words = " | ".join([
                    word for word, _ in self.original_topics[topic][:n_words]
                ])
            else:
                og_words = "N/A"
            
            # Get new words
            new_topic = self.topic_model.get_topic(topic)
            if new_topic:
                new_words = " | ".join([
                    word for word, _ in new_topic[:n_words]
                ])
            else:
                new_words = "N/A"
            
            df_data.append({
                'Topic': topic,
                'Original': og_words,
                method_name: new_words
            })
        
        return pd.DataFrame(df_data)

    def get_topic_info(self) -> pd.DataFrame:
        """
        Get comprehensive information about all topics.
         """
        return self.topic_model.get_topic_info()
    
    def get_topic(self, topic_id: int) -> List[Tuple[str, float]]:
        """
        Get keywords and scores for a specific topic.
        
        Args:
            topic_id: Topic identifier"""
        return self.topic_model.get_topic(topic_id)

    def find_topics(self, query: str, top_n: int = 5) -> Tuple[List[int], List[float]]:
        """
        Find topics most similar to a query string.
        
        Args:
            query: Search query
            top_n: Number of topics to return
            
        Returns:
            Tuple of (topic_ids, similarity_scores)
        """
        topics, similarities = self.topic_model.find_topics(query, top_n=top_n)
        
        print(f"\nTop {top_n} topics for query '{query}':")
        for topic, sim in zip(topics, similarities):
            topic_words = self.get_topic(topic)
            if topic_words:
                words = " | ".join([w for w, _ in topic_words[:3]])
                print(f"  Topic {topic} (similarity: {sim:.3f}): {words}")
        
        return topics, similarities
    
    def visualize_documents(self, titles: Optional[List[str]] = None, width: int = 1200, height: int = 750
    ):
        """
        Create interactive visualization of documents and topics."""
        if titles is None:
            titles = [f"Doc {i}" for i in range(len(self.documents))]
            
        fig = self.topic_model.visualize_documents(
            titles,
            width=width,
            height=height,
            hide_annotations=True
        )
        # Update font for better readability
        fig.update_layout(font=dict(size=14))
        return fig
    
    def visualize_hierarchy(self):
        """
        Visualize hierarchical topic structure."""
        return self.topic_model.visualize_hierarchy()
    
    def visualize_barchart(self, top_n_topics: int = 8):
        """
        Visualize top words per topic as bar charts.
        Args:
            top_n_topics: Number of topics to show"""
        return self.topic_model.visualize_barchart(top_n_topics=top_n_topics)
    
    def export_results(self, output_dir: str = "results"):
        """
        Export all results and visualizations.
        
        Args:
            output_dir: Directory to save results
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save topic info
        topic_info = self.get_topic_info()
        topic_info.to_csv(f"{output_dir}/topic_info.csv", index=False)
        print(f"Saved topic info to {output_dir}/topic_info.csv")
        
        # Save detailed topic representations
        with open(f"{output_dir}/topic_keywords.txt", "w") as f:
            for topic_id in range(len(topic_info) - 1):
                keywords = self.get_topic(topic_id)
                f.write(f"\nTopic {topic_id}:\n")
                for word, score in keywords[:10]:
                    f.write(f"  {word}: {score:.4f}\n")
        print(f"Saved topic keywords to {output_dir}/topic_keywords.txt")
        
        print(f"\nResults exported to {output_dir}/")

def compare_representation_methods(documents: List[str]) -> pd.DataFrame:
    """
    Compare different topic representation methods on the same dataset.
    """
    print("=" * 80)
    print("COMPARING TOPIC REPRESENTATION METHODS")
    print("=" * 80)
    
    # Initialize and fit base model
    modeler = TopicModeler()
    modeler.fit(documents)
    
    # Store results
    results = {
        'Original (TF-IDF)': modeler._calculate_topic_differences("Original")
    }
    
    # Test KeyBERT-inspired
    keybert_diff = modeler.apply_keybert_representation()
    results['KeyBERT-Inspired'] = keybert_diff
    
    # Reset to original
    modeler.topic_model.topic_representations_ = deepcopy(modeler.original_topics)
    
    # Test MMR with different diversity values
    for diversity in [0.3, 0.5, 0.7]:
        mmr_diff = modeler.apply_mmr_representation(diversity=diversity)
        results[f'MMR (div={diversity})'] = mmr_diff
        modeler.topic_model.topic_representations_ = deepcopy(modeler.original_topics)
    
    # Test generative approach
    gen_diff = modeler.apply_generative_representation()
    results['Generative (Flan-T5)'] = gen_diff
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    for method, df in results.items():
        print(f"\n{method}:")
        print(df.to_string(index=False))
    
    return results
def run_complete_analysis():
    """
    Run a complete topic modeling analysis on the ArXiv dataset.
    """
    from datasets import load_dataset
    
    # Load dataset
    print("Loading ArXiv NLP dataset...")
    dataset = load_dataset("MaartenGr/arxiv_nlp")["train"]
    abstracts = dataset["Abstracts"]
    titles = dataset["Titles"]
    
    # 5000 documents
    abstracts = abstracts[:5000]
    titles = titles[:5000]
    
    # Initialize modeler
    modeler = AdvancedTopicModeler(
        embedding_model_name='thenlper/gte-small',
        n_components=5,
        min_cluster_size=50
    )
    
    # Fit the model
    modeler.fit(abstracts)
    
    # Get topic information
    print("\nTopic Information:")
    topic_info = modeler.get_topic_info()
    print(topic_info.head(10))
    
    # Analyze specific topics
    print("\nDetailed look at Topic 0:")
    topic_0 = modeler.get_topic(0)
    for word, score in topic_0[:10]:
        print(f"  {word}: {score:.4f}")
    
    # Search for topics
    modeler.find_topics("transformer attention mechanism")
    modeler.find_topics("legal court law")
    modeler.find_topics("speech recognition")
    
    # Apply different representations
    print("\n" + "=" * 80)
    print("Testing Different Representation Methods:")
    print("=" * 80)
    
    # KeyBERT
    keybert_results = modeler.apply_keybert_representation()
    print("\nKeyBERT Results:")
    print(keybert_results)
    
    # MMR
    mmr_results = modeler.apply_mmr_representation(diversity=0.5)
    print("\nMMR Results:")
    print(mmr_results)
    
    # Export results
    modeler.export_results("results")
    
    # Create visualizations (these would be saved as HTML files)
    try:
        # Document visualization
        fig_docs = modeler.visualize_documents(titles=list(titles))
        fig_docs.write_html("results/documents_topics.html")
        print("Saved document visualization to results/documents_topics.html")
        
        # Hierarchy
        fig_hierarchy = modeler.visualize_hierarchy()
        fig_hierarchy.write_html("results/topic_hierarchy.html")
        print("Saved hierarchy to results/topic_hierarchy.html")
        
        # Bar charts
        fig_bars = modeler.visualize_barchart()
        fig_bars.write_html("results/topic_barchart.html")
        print("Saved barchart to results/topic_barchart.html")
    except Exception as e:
        print(f"Visualization export note: {e}")
    
    return modeler


if __name__ == "__main__":
    # Run the complete analysis
    modeler = run_complete_analysis()
