# Simple RAG - Retrieval Augmented Generation

## Requirements (recommended version)

Install the following libraries from conda forge using
```bash
conda install -c conda-forge LIBRARY_NAME
```

- python (3.12.8)
- langchain (0.3.13)
- pypdf (5.1.0)
- chromadb (0.5.23)

## Usage

- Clone the repository and make sure you are using a python environment meeting the requirements above.
- Open a terminal and navigate to the root directory of the project 
- Index PDF chunks to ChromaDB by running (check the file `[database/pdf2chroma.py](database/pdf2chroma.py)` for more options):
```bash
python -m database.pdf2chroma
```
- Start a simple RAG-based chatbot (check the file [query.py](query.py) for more options):
```bash
python -m query
```