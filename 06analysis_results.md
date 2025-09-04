# Topic Modeling Results

## Brief Summary

More than **40,000+ ArXiv NLP research papers** were analyzed, discovering **161 distinct research topics**.

---

## Topic Representation Comparison

### Overview
I compared four different topic representation models to evaluate their effectiveness in generating interpretable and meaningful topic keywords.

### 1. Baseline: TF-IDF Representation
Traditional frequency-based approach that identifies keywords based on their importance within topics relative to the entire dataset.

| Topic | Keywords |
|-------|----------|
| **0 - Speech Processing** | speech \| asr \| recognition \| end \| acoustic |
| **1 - Medical NLP** | medical \| clinical \| biomedical \| patient \| health |
| **2 - Text Summarization** | summarization \| summaries \| summary \| abstractive \| document |
| **3 - Hate Speech Detection** | hate \| offensive \| speech \| detection \| toxic |
| **4 - Machine Translation** | translation \| nmt \| machine \| neural \| bleu |

### 2. KeyBERT-Inspired Representation
Ability to produce more semantically coherent and relevant keywords to each cluster.

| Topic | Original Keywords | KeyBERT-Inspired Keywords | Improvement |
|-------|------------------|---------------------------|-------------|
| **0** | speech \| asr \| recognition \| end \| acoustic | speech \| transcription \| encoder \| phonetic \| acoustic | More specific technical terms |
| **1** | medical \| clinical \| biomedical \| patient \| health | nlp \| ehr \| ehrs \| clinical \| medicine | Domain-specific terminology (EHR) |
| **2** | summarization \| summaries \| summary \| abstractive \| document | summarization \| summarizers \| summaries \| summary \| abstractive | Better term consistency |
| **3** | hate \| offensive \| speech \| detection \| toxic | hate \| hateful \| language \| offensive \| classification | Clearer task framing |
| **4** | translation \| nmt \| machine \| neural \| bleu | translation \| translate \| translations \| multilingual \| machine | Broader translation aspects |

### 3. Maximal Marginal Relevance (MMR)
Balances relevance with diversity (diversity=0.5), preventing redundant keywords while maintaining topic coherence.

| Topic | Original Keywords | MMR Keywords | Diversity Benefit |
|-------|------------------|--------------|-------------------|
| **0** | speech \| asr \| recognition \| end \| acoustic | speech \| asr \| error \| model \| automatic | Includes error handling & modeling |
| **1** | medical \| clinical \| biomedical \| patient \| health | clinical \| biomedical \| patient \| healthcare \| medical | Balanced medical terminology |
| **2** | summarization \| summaries \| summary \| abstractive \| document | summarization \| extractive \| factual \| document \| abstractive | Covers both paradigms |
| **3** | hate \| offensive \| speech \| detection \| toxic | hate \| toxic \| abusive \| platforms \| dataset | Broader problem scope |
| **4** | translation \| nmt \| machine \| neural \| bleu | nmt \| bleu \| multilingual \| parallel \| source | Technical diversity |

### 4. Generative AI (Flan-T5)
Uses language model to generate natural language descriptions of topics based on the dataset.

| Topic | Original Keywords | Generative Description | Analysis |
|-------|------------------|------------------------|----------|
| **0** | speech \| asr \| recognition \| end \| acoustic | Speech-to-speech | Concise, high-level description |
| **1** | medical \| clinical \| biomedical \| patient \| health | Science/Tech | Too generic, needs refinement |
| **2** | summarization \| summaries \| summary \| abstractive \| document | Abstractive summarization | Accurate technical term |
| **3** | hate \| offensive \| speech \| detection \| toxic | Science/Tech | Insufficient specificity |
| **4** | translation \| nmt \| machine \| neural \| bleu | Science/Tech | Requires better prompting |

**Generative Model Improvement**: Open source generative models require more careful prompt engineering such as in-context or tree-based prompting for better results.

---

## Visualization Analysis

### Figure 1: Document Clustering in 2D Space
<img width="1200" height="750" alt="documents and topics" src="https://github.com/user-attachments/assets/96356dcf-fff3-44ad-9a48-83c71a5c1769" />


### Figure 2: Topic Word Importance
<img width="1000" height="500" alt="topic word scores" src="https://github.com/user-attachments/assets/9bc17209-cc8a-4e4d-a770-397ac198ccd7" />

