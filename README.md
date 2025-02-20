# DeepSeek-R1 Embedding Generation with Ollama

## Overview
This guide walks you through generating document embeddings using DeepSeek-R1 with Ollama for efficient similarity-based document retrieval. We will utilize DeepSeek-R1 to create high-dimensional semantic embeddings and retrieve relevant document chunks based on a user query.

## Prerequisites
Ensure you have the following installed before proceeding:
- Python 3.8+
- Ollama installed ([Installation Guide](https://ollama.com))
- ChromaDB for document retrieval (`pip install chromadb`)
- Faiss (optional for large-scale retrieval) (`pip install faiss-cpu`)

## Step 1: Install DeepSeek-R1
DeepSeek-R1 requires a version greater than **1.5B** for optimal performance. Install a suitable version via Ollama:
(i'm not sure from it double check
)
```sh
ollama pull deepseek-r1:7b  # Example: Using the 7B model
```

Available model sizes: `7B`, `8B`, `14B`, `32B`, `70B`, `671B`. Replace `X` in `deepseek-r1:X` with your preferred size.

## Step 2: Generating Embeddings
We use the DeepSeek-R1 model through Ollama to generate embeddings for document chunks. 

### Example Code:
```python
from concurrent.futures import ThreadPoolExecutor
import ollama

def generate_embedding(text, model="deepseek-r1:7b"):
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Example usage
document_chunks = ["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"]

with ThreadPoolExecutor() as executor:
    embeddings = list(executor.map(generate_embedding, document_chunks))
```

This parallelized approach speeds up the embedding generation process compared to sequential execution.

## Step 3: Context Retrieval with ChromaDB
Once the embeddings are generated, we store them in ChromaDB and retrieve the most relevant document chunks.

```python
import chromadb

def retrieve_context(query, model="deepseek-r1:7b", retriever):
    query_embedding = generate_embedding(query, model)
    results = retriever.query(query_embedding, n_results=5)
    return " ".join([doc["text"] for doc in results])
```

## Optimizations
For improved efficiency, consider the following optimizations:
- **Chunk Size Tuning**: Adjust `chunk_size` and `chunk_overlap` for optimal retrieval performance.
- **Smaller Models**: Use `deepseek-r1:7b` or `deepseek-r1:8b` for lower resource consumption.
- **Faiss for Scaling**: Use Faiss for faster large-scale document retrieval.
- **Batch Processing**: Process document chunks in batches to enhance performance.

## Conclusion
By using DeepSeek-R1 with Ollama, you can efficiently generate embeddings and perform similarity-based document retrieval. Experiment with different model sizes and retrieval techniques to optimize performance for your use case.
