from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import streamlit as st

DB_FAISS_PATH = 'vectorstore/db_faiss'
 
llm=Ollama(model="llama3")

prompt= ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
<context>
{context}
<content>
Question:{input}                                     
 """)

document_chain=create_stuff_documents_chain(llm,prompt)

embeddings = OllamaEmbeddings(model="llama3",
                                       model_kwargs={'device': 'cpu'})
db=FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
retriver=db.as_retriever()
retrieval_chain=create_retrieval_chain(retriver,document_chain)



st.title("Ayurveda health and wellness navigator with Ollama")
input_text=st.text_input("Ask me anything")
response=retrieval_chain.invoke({"input":input_text})
if input_text:
    st.write(response['answer'])