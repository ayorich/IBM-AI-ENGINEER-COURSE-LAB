"""
RAG QA Bot using free, local models (no API keys required).
- Embeddings: sentence-transformers (runs locally)
- LLM: Ollama (local) or Hugging Face pipeline fallback (flan-t5-small)
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFacePipeline
import transformers

import gradio as gr

# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")


def get_llm():
    """Use Ollama if available (free, local). Else fallback to small Hugging Face model (CPU)."""
    try:
        return Ollama(
            model="llama3.2",
            temperature=0.5,
            num_predict=256,
        )
    except Exception:
        pass
    # Fallback: small model that runs on CPU, no API key
    model_id = "google/flan-t5-small"
    pipe = transformers.pipeline(
        "text2text-generation",
        model=model_id,
        max_new_tokens=256,
        device=-1,  # CPU
    )
    return HuggingFacePipeline(pipeline=pipe)


def get_embeddings():
    """Local embeddings via sentence-transformers (no API key)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def document_loader(file):
    loader = PyPDFLoader(file)
    return loader.load()


def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(data)


def vector_database(chunks):
    embedding_model = get_embeddings()
    return Chroma.from_documents(chunks, embedding_model)


def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()


def retriever_qa(file, query):
    if file is None:
        return "Please upload a PDF file first."
    if not query or not query.strip():
        return "Please enter a question."
    try:
        llm = get_llm()
        retriever_obj = retriever(file)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=True,
        )
        response = qa.invoke({"query": query})
        return response["result"]
    except Exception as e:
        if "ollama" in str(e).lower() or "connection" in str(e).lower():
            return (
                "Ollama is not running or not installed. "
                "Install from https://ollama.com and run: ollama pull llama3.2"
            )
        return f"Error: {e}"


rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=[".pdf"],
            type="filepath",
        ),
        gr.Textbox(
            label="Input Query",
            lines=2,
            placeholder="Type your question here...",
        ),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="RAG QA Bot (Free Models)",
    description=(
        "Upload a PDF and ask questions. Uses local models only (no API keys). "
        "For best quality, install Ollama and run: ollama pull llama3.2"
    ),
)

if __name__ == "__main__":
    rag_application.launch(server_name="0.0.0.0", server_port=7860)
