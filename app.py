import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

def process_input(urls, question):
    
    # 1. Load Documents
    urls_list = urls.split("\n")
    loader = WebBaseLoader(urls_list)
    documents = loader.load()
    
    # 2. Split Documents into Chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(documents)
    
    # 3. Select Embeddings
    embeddings = OllamaEmbeddings(model="mistral")
    
    # 4. Create a Vector Store
    vector_store = Chroma.from_documents(
        documents = doc_splits,
        embedding = embeddings,
        collection_name = "rag-chroma",
    )
    
    # 5. Create Retriever Interface
    retriever = vector_store.as_retriever()
    
    # 6. Perform the RAG
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = Ollama(model="mistral")
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 7. Run Query
    return rag_chain.invoke(question)
    
# streamlit UI
st.title("document query with Ollama")
st.write("enter urls(one per line) and a question to query the documents")

# UI for input fields
urls = st.text_area("Enter URLs separated by new lines", height=150)
question = st.text_input("Question")

# button to process input
if st.button ('query documents'):
    with st.spinner('Processing...'):
        answer = process_input(urls, question)
        st.text_area("Answer", value=answer, height=300, disabled=True)
