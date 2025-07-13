

import os
import pandas as pd



from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
# from langchain_core.documents import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings





def load_documents(doc_dir):
    documents = []
    # Ensure the directory exists
    if not os.path.exists(doc_dir):
        print(f"Warning: Document directory '{doc_dir}' not found.")
        return []

    for filename in os.listdir(doc_dir):
        file_path = os.path.join(doc_dir, filename)

        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    content = ", ".join(str(v) for v in row.values)
                    documents.append(Document(page_content=content))
            except Exception as e:
                print(f"Error loading CSV file {filename}: {e}")

        elif filename.endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading TXT file {filename}: {e}")
    return documents


def build_vectorstore(docs):
    # Ensure there are documents to process
    if not docs:
        print("No documents loaded to build vectorstore.")
        return None

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=128,
        temperature=0.3,
        do_sample=False,
        device=-1  # CPU
    )
    return HuggingFacePipeline(pipeline=pipe)


def build_qa_chain():
    docs = load_documents("docs")
    if not docs:
        raise Exception("No documents found. Please ensure the 'docs' directory exists and contains files.")

    vs = build_vectorstore(docs)
    if vs is None:
        raise Exception("Failed to build vector store. Check document loading.")

    llm = load_llm()

    retriever = vs.as_retriever(search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa