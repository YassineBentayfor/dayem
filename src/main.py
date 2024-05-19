import os
from urllib.request import urlretrieve
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch
import gradio as gr

class DocumentProcessingPipeline:
    """
    A class to represent a document processing pipeline.

    Attributes:
    - loader (PyPDFDirectoryLoader): A loader object to load PDF files from a local directory.
    - docs_before_split (list): A list of document objects before text splitting.
    - docs_after_split (list): A list of document objects after text splitting.
    - avg_char_before_split (int): The average number of characters per document before splitting.
    - avg_char_after_split (int): The average number of characters per chunk after splitting.
    - huggingface_embeddings (HuggingFaceBgeEmbeddings): Embeddings object for generating document embeddings.
    - vectorstore (FAISS): Vector store for similarity searching.
    - retriever (RetrievalQA): Retrieval question-answering model.
    """

    def __init__(self, directory_path):
        """
        Initialize the DocumentProcessingPipeline with the directory path containing PDF files.

        Args:
        - directory_path (str): The path to the directory containing PDF files.
        """
        self.loader = PyPDFDirectoryLoader(directory_path)
        self.docs_before_split = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        self.docs_after_split = self.text_splitter.split_documents(self.docs_before_split)
        self.avg_char_before_split = self._calculate_average_character_length(self.docs_before_split)
        self.avg_char_after_split = self._calculate_average_character_length(self.docs_after_split)
        self._initialize_embeddings()
        self._initialize_vectorstore()
        self._initialize_retriever()

    def _calculate_average_character_length(self, docs):
        """
        Calculate the average number of characters per document.

        Args:
        - docs (list): A list of document objects.

        Returns:
        - avg_length (int): The average number of characters per document.
        """
        avg_length = sum([len(doc.page_content) for doc in docs]) // len(docs)
        return avg_length

    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings for document representation."""
        self.huggingface_embeddings = HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={'normalize_embeddings': True}
        )

    def _initialize_vectorstore(self):
        """Initialize FAISS vector store for similarity searching."""
        self.vectorstore = FAISS.from_documents(self.docs_after_split, self.huggingface_embeddings)

    def _initialize_retriever(self):
        """Initialize the RetrievalQA model for question answering."""
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def process_documents(self, query):
        """
        Process documents and retrieve relevant information.

        Args:
        - query (str): The query for information retrieval.

        Returns:
        - relevant_documents (list): A list of relevant document content.
        """
        relevant_documents = self.retriever.similarity_search(query)
        return relevant_documents


# Initialize DocumentProcessingPipeline
document_pipeline = DocumentProcessingPipeline("./docs")

# Process documents and retrieve relevant information
query = """I'm a farmer from Morocco, can you give recommendations based on the weather on what crops I can grow?"""
relevant_docs = document_pipeline.process_documents(query)

# Display relevant document content
for doc in relevant_docs:
    print(doc.page_content)

# Define the function to answer the question
def answer_question(prompt):
    # Call the QA chain with the provided prompt
    result = document_pipeline.process_documents(prompt)
    return result

# Create Gradio Interface
gr.Interface(
    fn=answer_question,
    inputs=gr.inputs.Textbox(lines=5, label="Enter Prompt"),
    outputs="text",
    title="Document Processing and QA System",
    description="Enter a prompt to retrieve relevant information from the documents."
).launch()
