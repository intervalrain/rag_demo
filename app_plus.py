import streamlit as st
import os
from auth_provider import AuthProvider
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from typing import Dict, List
from langdetect import detect

def get_user_permissions(db: AuthProvider) -> Dict[str, List[str]]:
    results = db.execute_query("select * from dual")
    permissions = {}
    for row in results:
        permissions[row['user_id']] = row['permissions'].split(',')
    return permissions

def process_input(vector_store, question, selected_pdfs):
    filters = [{"source": pdf} for pdf in selected_pdfs]
    combined_filter = {"$or": filters} if len(filters) > 1 else filters[0]
    
    retriever = vector_store.as_retriever(
        search_kwargs={"filter": combined_filter}
    )

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

    return rag_chain.invoke(question)

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # 預設為英語


# Streamlit UI
st.set_page_config(layout="wide", page_title="UMC - DSM Bot")

# 使用 sidebar 來放置文件選擇功能
with st.sidebar:
    st.title("DSM Documents")

    connection_string = os.environ.get("DB_CONNECTION_STRING", "fakeconnection")
    db = AuthProvider(connection_string)

    user_permissions = get_user_permissions(db)
    user_identity = "admin"  # 固定為 admin 用戶
    st.write(f"Current user: {user_identity}")

    persist_directory = "./vectordb"
    embeddings = OllamaEmbeddings(model="mistral")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    available_pdfs = user_permissions[user_identity]

    pdf_types = {"All": available_pdfs}
    for pdf in available_pdfs:
        pdf_type = pdf[:4]
        if pdf_type not in pdf_types:
            pdf_types[pdf_type] = []
        pdf_types[pdf_type].append(pdf)

    selected_type = st.selectbox("Filter PDF by type", options=list(pdf_types.keys()))

    filtered_pdfs = pdf_types[selected_type]

    st.write("Available PDFs:")
    selected_pdfs = st.multiselect(
        "Select PDFs for querying",
        options=filtered_pdfs,
        default=filtered_pdfs,
        format_func=lambda x: x
    )

# 主要聊天界面
st.title("DSM Bot")

# if 'messages' not in st.session_state:
#     st.session_state.messages = [
#         {"role": "system", "content": "You are a helpful assistant. Always respond in the same language as the user's question."}
#     ]

# # 創建一個固定高度的容器來顯示聊天記錄
# chat_container = st.container()
# chat_container.markdown("## Chat")
# chat_display = chat_container.empty()

# def update_chat_display():
#     chat_history = ""
#     for message in st.session_state.messages[1:]:  # 跳過系統消息
#         if message["role"] == "user":
#             chat_history += f'<div style="text-align: right; margin: 5px; padding: 5px;">{message["content"]}</div>'
#         else:
#             chat_history += f'<div style="text-align: left; margin: 5px; padding: 5px;">{message["content"]}</div>'
    
#     chat_display.markdown(f"""
#         <div id="chat-container" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; border-radius: 5px;">
#             {chat_history}
#         </div>
#         <script>
#             var chatContainer = document.getElementById('chat-container');
#             chatContainer.scrollTop = chatContainer.scrollHeight;
#         </script>
#     """, unsafe_allow_html=True)

# def on_input_change():
#     user_input = st.session_state.user_input
#     if user_input:
#         language = detect_language(user_input)
        
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         update_chat_display()
        
#         # 顯示 "機器人正在輸入" 的效果
#         with st.spinner('Assistant is typing...'):
#             response = process_input(vector_store, user_input, selected_pdfs)
        
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         update_chat_display()

# # 在底部創建輸入框
# st.text_input("Your question", key="user_input", on_change=on_input_change)

# # 初始顯示聊天記錄
# update_chat_display()

# 自定義 CSS
st.markdown("""
<style>
.chat-container {
    height: 400px;
    overflow-y: auto;
    display: flex;
    flex-direction: column-reverse;
    border: 1px solid var(--text-color);
    padding: 10px;
    border-radius: 5px;
}
.message {
    margin: 5px;
    padding: 10px;
    border-radius: 15px;
    max-width: 70%;
    word-wrap: break-word;
}
.user-message {
    align-self: flex-end;
    background-color: var(--primary-color);
    color: white;
    border-bottom-right-radius: 0;
}
.assistant-message {
    align-self: flex-start;
    background-color: var(--secondary-background-color);
    border-bottom-left-radius: 0;
}
</style>
""", unsafe_allow_html=True)


if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant. Always respond in the same language as the user's question."}
    ]

# 創建一個固定高度的容器來顯示聊天記錄
chat_container = st.container()
chat_container.markdown("## Chat")
chat_display = chat_container.empty()

def update_chat_display():
    chat_history = ""
    for message in st.session_state.messages[1:]:  # 跳過系統消息
        if message["role"] == "user":
            chat_history = f'<div class="message user-message">{message["content"]}</div>' + chat_history
        else:
            chat_history = f'<div class="message assistant-message">{message["content"]}</div>' + chat_history
    
    chat_display.markdown(f"""
        <div class="chat-container">
            {chat_history}
        </div>
        <script>
            var chatContainer = document.querySelector('.chat-container');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        </script>
    """, unsafe_allow_html=True)

def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.user_input = ""
    if user_input:
        language = detect_language(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        update_chat_display()
        
        # 顯示 "機器人正在輸入" 的效果
        with st.spinner('Assistant is typing...'):
            response = process_input(vector_store, user_input, selected_pdfs)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        update_chat_display()

# 在底部創建輸入框
st.text_input("Your question", key="user_input", on_change=on_input_change)

# 初始顯示聊天記錄
update_chat_display()