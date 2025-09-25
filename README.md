# codebert-clone-detection
This project demonstrates how to generate embeddings of code snippets using the CodeBERT model and visualize them with t-SNE.  
We sample 5 pairs (10 snippets) from the CodeXGLUE BigCloneBench dataset, embed them using CodeBERT, and plot the embeddings in 2D.  
The goal is to observe whether clone pairs (semantically similar code) appear closer together than non-clone pairs.

---

## Execution Environment

This project was tested in **Google Colab** with the following environment:

- Python 3.10
- CUDA 12.1
- GPU: NVIDIA Tesla T4 (via Colab runtime)

---

## Dependencies

- Python 3.9+ (tested on Google Colab)
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Matplotlib
- NumPy
- Pandas

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Findings

- The t-SNE visualization of 5 sampled pairs (10 snippets) did not show a clear separation between clone pairs (`label=1`) and non-clone pairs (`label=0`). While some clone pairs appeared close to each other in the 2D space, others were scattered, and several non-clone snippets also appeared nearby.  


- An additional Cosine similarity analysis was done which showed that embeddings for both clone and non-clone pairs had very high similarity values (~0.99). This suggests that, for such a small sample, CodeBERT’s `[CLS]` embeddings are not strongly discriminating between semantically similar and dissimilar functions.  

- Possible explanations:
  - **Small sample size:** Only 10 snippets are not enough to reveal consistent clustering patterns.
  - **Pooling choice:** Using only the `[CLS]` token may lose semantic detail; mean pooling across tokens might yield better separation.
  - **t-SNE limitations:** With very few points, t-SNE can produce distorted 2D layouts.

- **Conclusion:** CodeBERT embeddings do capture semantic information, but in this experiment with limited data and `[CLS]` pooling, clone and non-clone pairs were not well distinguished. Larger sample sizes or alternative pooling strategies could provide clearer results.
