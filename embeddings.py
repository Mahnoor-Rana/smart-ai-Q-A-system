import os 
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.huggingface import HuggingFaceEmbeddings
from constants import CHROMA_SETTINGS

# persistant_directory = "db "

# def main ():
#     for root, dirs, files in os.walk("us_census"):
#         for file in files:
#             if file.endswith(".pdf"):
#                    print(file)
#                    loader = PDFMinerLoader(os.path.join(root,file))

#     doc = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=500)
#     texts = text_splitter.split_documents(doc)
#     # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
   
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = Chroma.from_documents(texts,embeddings,persist_directory=persistant_directory, client_settings = CHROMA_SETTINGS)
#     # vectore_store = Chroma(embeddings,persist_directory=persistant_directory)
#     # vectore_store.save()
#     # db.save()
#     db.persist()
#     # db = None 
# if __name__ == "__main__" :
#      main()

from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def main():
  
    for root, dirs, files in os.walk("./us_census"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Processing: {file}")
                loader = PDFMinerLoader(os.path.join(root, file))
    doc = loader.load()  
    if not doc:
        print("No documents were loaded from the PDF!")
    else:
        print(f"Loaded {len(doc)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    
    texts = text_splitter.split_documents(doc)
    if not texts:
        print("Text splitting failed!")
    else:
        print(f"Split into {len(texts)} chunks.")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if embeddings:
        print("Embeddings created successfully.")
    else:
        print("Failed to create embeddings.")
    db = Chroma.from_documents(texts, embeddings, persist_directory="./db", client_settings=CHROMA_SETTINGS)
    print("Chroma DB created and stored successfully.")
    print(texts[0].page_content)
    # db.persist()
    # db = None 
if __name__ == "__main__":
    main()
