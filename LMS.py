import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import tempfile

# --------------------- CONFIG & STYLING ---------------------
st.set_page_config(page_title="LMS ChatBot", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f4f6f9;}
        .stButton>button {
            background-color: #0d6efd;
            color: white;
            border-radius: 6px;
            padding: 0.4rem 1rem;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            height: 40px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Gemini-Powered LMS ChatBot")
st.write("Upload your course PDFs and interact with them using Google Gemini AI.")

# --------------------- API KEY SETUP ---------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyBtzXpY89p1bB6nhlHF422fsQzCItACQUo"  # Replace with your actual key

# --------------------- PDF Upload ---------------------
uploaded_files = st.file_uploader("📚 Upload Course PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing uploaded PDFs..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            all_docs.extend(docs)

        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs = text_splitter.split_documents(all_docs)

        # Vector Store Creation
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma.from_documents(split_docs, embedding=embeddings)

        # Retrieval Chain Setup
        retriever = db.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

        # Prompt Template
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful and knowledgeable tutor.
        Use only the information in the context below to answer the question in a clear, step-by-step manner.
        If unsure, say "I'm not sure based on the document."

        <context>
        {context}
        </context>

        Question: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # --------------------- Chat Input ---------------------
        st.markdown("## 💬 Ask a Question about the Course Material")
        query = st.text_input("Ask your question here:")

        if query:
            with st.spinner("Finding the best answer..."):
                try:
                    response = retrieval_chain.invoke({"input": query})
                    st.success("✅ Answer")
                    st.markdown(f"**Answer:** {response['answer']}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
