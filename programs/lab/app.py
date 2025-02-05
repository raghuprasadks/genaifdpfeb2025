import os
import requests
import PyPDF2
import streamlit as st
import cohere
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()


# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to download and extract IPC data
def download_and_extract_ipc():
    pdf_path = "IPC_186045_removed_removed.pdf"
    
    if not os.path.exists(pdf_path):
        url = "https://www.indiacode.nic.in/repealedfileopen?rfilename=A1860-45.pdf"
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with open(pdf_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            st.success("✅ IPC Document Downloaded Successfully!")
        else:
            st.error("❌ Failed to download the IPC PDF.")
            return None

    text = extract_text_from_pdf(pdf_path)
    
    if text.strip():
        with open("Indian_Penal_Code.txt", "w", encoding="utf-8") as text_file:
            text_file.write(text)
        st.success("✅ Text extracted and saved successfully!")
        return text
    else:
        st.error("❌ No extractable text found in the PDF.")
        return None

# Initialize chatbot using Cohere
def create_chatbot():
    cohere_key = os.getenv("COHERE_API_KEY")  # Fetch from Colab secrets

    
    if not cohere_key:
        st.error("❌ Cohere API Key not found. Set it in Colab secrets.")
        return None

    text = download_and_extract_ipc()
    
    if not text:
        st.error("❌ No text available for chatbot training.")
        return None

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    embeddings = CohereEmbeddings(model="embed-english-v2.0", cohere_api_key=cohere_key)
    vector_store = FAISS.from_texts(texts, embeddings, normalize_L2=True)

    llm = ChatCohere(model="command-r-plus", temperature=0.3, cohere_api_key=cohere_key)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain

# Streamlit UI
st.set_page_config(page_title="IPC Chatbot", page_icon="⚖️", layout="wide")

st.title("⚖️ Indian Penal Code (IPC) Chatbot")
st.write("Ask legal questions related to the Indian Penal Code (IPC).")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = create_chatbot()

user_input = st.text_input("Enter your question:", placeholder="e.g., What is Section 302 in IPC?")
if st.button("Ask"):
    if user_input and st.session_state.qa_chain:
        response = st.session_state.qa_chain.invoke(user_input)
        st.subheader("Response:")
        st.write(response["result"])
    else:
        st.error("Chatbot is not ready. Please check API key or reload the app.")

st.markdown("---")
st.caption("Powered by Cohere and LangChain")
