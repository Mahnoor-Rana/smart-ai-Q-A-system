from flask import Flask, request, jsonify
import os
import streamlit as st 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.embeddings import SentenceTransformerEmbeddings
from constants import CHROMA_SETTINGS
from transformers import pipeline


# torch.mps.empty_cache()
# export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9
device = torch.device("mps" if torch.has_mps else "cpu")
checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
basemodel = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map =None,
    torch_dtype = torch.float16,
).to(device)

@st.cache_resource 
def llm_pipeline():
    pipe = pipeline(
        "text2text-generation",
        model = basemodel,
        tokenizer = tokenizer,
        max_length=128,

        do_sample= True,
        temperature = 0.3,
        top_p= 0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource 
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(
        model_name = "all-MiniLM-L6-v2"
    )
    db = Chroma(persist_directory="db",embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever= retriever,
        # return_source_document = True
    )
    return qa 
def process_answer(instruuction):
    response = ''
    instruuction = instruuction
    qa = qa_llm()
    generated_text = qa(instruuction)
    answer = generated_text['result']
    return answer, generated_text

def main():
    st.title("Search in your PDF ")
    with st.expander("Smart Document Answering System"):
        st.markdown(
            """
        This is a generative ai quuesion aswering app that responds to questions about your PDF file"""
        )
    question = st.text_area("Enter your Question")
    if st.button("Search"):
        st.info("Your Question: " + question)
        st.info("Your answer ")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)


if __name__ == "__main__" :
     main()