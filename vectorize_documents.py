import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def vectorize_documents(pdf_dir, persist_directory):
    embeddings = OllamaEmbeddings(model="mistral")
    
    logging.info("正在加載或創建向量資料庫...")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # 加載已處理文件列表
    processed_files_path = os.path.join(persist_directory, "processed_files.txt")
    processed_files = set()
    if os.path.exists(processed_files_path):
        with open(processed_files_path, "r") as f:
            processed_files = set(f.read().splitlines())

    updated = False
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf') and filename not in processed_files:
            logging.info(f"處理新文件：{filename}")
            try:
                file_path = os.path.join(pdf_dir, filename)
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                
                doc_type = filename[:4]
                for page in pages:
                    page.metadata['type'] = doc_type
                    page.metadata['source'] = filename

                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
                splits = text_splitter.split_documents(pages)

                vector_store.add_documents(splits)
                processed_files.add(filename)
                updated = True
                logging.info(f"文件 {filename} 已添加到向量資料庫")
            except Exception as e:
                logging.error(f"處理文件 {filename} 時發生錯誤：{str(e)}")

    if updated:
        vector_store.persist()
        # 更新已處理文件列表
        with open(processed_files_path, "w") as f:
            for filename in processed_files:
                f.write(f"{filename}\n")
        logging.info("向量存儲已更新並保存")
    else:
        logging.info("沒有新文件需要處理")

if __name__ == "__main__":
    pdf_dir = "./docs"
    persist_directory = "./vectordb"
    vectorize_documents(pdf_dir, persist_directory)