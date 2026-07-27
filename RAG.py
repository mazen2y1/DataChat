import os
import torch
from dotenv import load_dotenv
from LLMProvider import get_llm, get_text
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

def build_RAG(texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    combined_text = "\n\n".join(texts)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(
        combined_text
    )

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={
            "device": device
        }
    )

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embedding
    )
    return vectorstore

def ask_RAG(vectorstore, question):
    documents = vectorstore.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    llm = get_llm()

    prompt = f"""
You are a precise question-answering assistant.

Answer the user's question using ONLY the information
in the context below.

Instructions:

- Base your answer strictly on the provided context.
- Do not use outside knowledge.
- If the context contains the answer, respond clearly and concisely.
- If the context contains only a partial answer, provide what is available and note what is missing.
- If the answer is not in the context, respond exactly with:
  "I could not find the answer in the uploaded documents."
- Do not guess or fabricate information.

Context:

{context}

Question:

{question}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content