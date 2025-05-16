"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings

from chromadb import Settings

# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional
# from chromadb import PersistentClient

# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF and MD RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None
if "use_rag" not in st.session_state:
    st.session_state["use_rag"] = True
if "markdown_files_content" not in st.session_state:
    st.session_state["markdown_files_content"] = []
if "uploaded_file_names" not in st.session_state:
    st.session_state["uploaded_file_names"] = []


def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info: Response from ollama.list()

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    try:
        # The new response format returns a list of Model objects
        if hasattr(models_info, "models"):
            # Extract model names from the Model objects
            model_names = tuple(model.model for model in models_info.models)
        else:
            # Fallback for any other format
            model_names = tuple()
            
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


def create_vector_db(file_upload, loader, temp_dir) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    data = loader.load()
    # print("data:", data[0].page_content)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    # Updated embeddings configuration with persistent storage
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client_settings = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    )
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=f"pdf_{hash(file_upload.name)}",  # Unique collection name per file
        client_settings=client_settings
    )
    logger.info("Vector DB created with persistent storage")

    # ##################################################
    # # ❶ point at the folder that holds chroma.sqlite3
    # client = PersistentClient(path=PERSIST_DIRECTORY)  # <- your persist_directory
    #
    # # ❷ list what’s in there
    # for col in client.list_collections():
    #     print(col.name, col.count())
    #
    #     # peek at the first 5 entries
    #     sample = col.peek(5)  # --> {'ids': [...], 'documents': [...], ...}
    #     print(sample["ids"][:3], "...")
    #
    #     # full fetch of one ID with vectors
    #     row = col.get(ids=[sample["ids"][0]],
    #                   include=["embeddings", "documents", "metadatas"])
    #     print(row)


    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db


def add_document_to_vector_db(file_upload, loader, temp_dir) -> None:
    """
    Add a document to the existing vector database.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.
        loader: Document loader for processing the uploaded file.
    """
    logger.info(f"Adding document to vector DB from file upload: {file_upload.name}")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    # Add new documents to the existing vector database
    st.session_state["vector_db"].add_documents(chunks)
    logger.info("New documents added to vector DB")
    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Initialize LLM
    llm = ChatOllama(model=selected_model)
    
    # Query prompt template to generate multiple questions with similar meaning
    # QUERY_PROMPT = PromptTemplate(
    #     input_variables=["question"],
    #     template="""You are an AI language model assistant. Your task is to generate 2
    #     different versions of the given user question to retrieve relevant documents from
    #     a vector database. By generating multiple perspectives on the user question, your
    #     goal is to help the user overcome some of the limitations of the distance-based
    #     similarity search. Provide these alternative questions separated by newlines.
    #     Original question: {question}""",
    # )

    # Set up retriever with 3 different questions to retrieve the matching chunks
    # retriever = MultiQueryRetriever.from_llm(
    #     vector_db.as_retriever(),
    #     llm,
    #     prompt=QUERY_PROMPT
    # )

    if st.session_state["use_rag"]:
        # retriever which uses vector-db chunks retrieved by similarity with original question
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        logger.info("Using vector-db chunks retrieved by similarity with original question")
        # show retrieved chunks
        # docs = retriever.invoke(question)
        # for i, doc in enumerate(docs, start=1):
        #     print(f"=== Chunk {i} ===")
        #     print(doc.page_content.strip())  # der Text‑Abschnitt
        #     if doc.metadata:
        #         print("Metadaten:", doc.metadata)  # z.B. {'source': 'file.md', 'chunk': 2}
        #     print()
    else:
        # retriever which uses string content of the markdown files in list 'markdown_files_content'
        combined_context = "new document\n" + "new document\n".join(st.session_state["markdown_files_content"])
        retriever = RunnableLambda(lambda _: combined_context)
        logger.info("Using string content of the markdown files")


    # RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            # Delete the collection
            vector_db.delete_collection()
            
            # Clear session state
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.subheader("🧠 Ollama PDF and MD files RAG playground", divider="gray", anchor=False)

    # Get available models
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Add checkbox for using RAG
    st.session_state["use_rag"] = col2.toggle(
        "Use RAG",
        key="rag_checkbox",
        value=True
    )

    # Model selection
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓",
            available_models,
            key="model_select"
        )

    # Regular file upload with unique key
    file_upload = col1.file_uploader(
        "Upload a PDF or MD file ↓",
        type=["pdf", "md"],
        accept_multiple_files=False,
        key="pdf_uploader"
    )

    temp_dir = tempfile.mkdtemp()
    if file_upload is not None:
        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
            logger.info(f"File saved to temporary path: {path}")

        if file_upload.name.endswith(".pdf"):
            loader = UnstructuredPDFLoader(path)
            file_type = "pdf"
        elif file_upload.name.endswith(".md"):
            loader = UnstructuredMarkdownLoader(path)
            file_type = "md"
        else:
            st.error("Unsupported file type. Please upload a PDF or Markdown file.")
            return
    else:
        return

    if file_upload and file_upload.name not in st.session_state["uploaded_file_names"]:
        path = os.path.join(temp_dir, file_upload.name)
        if file_type == "md":
            with open(path, "r", encoding="utf-8") as f:
                st.session_state["markdown_files_content"].append(f.read())
        if st.session_state["vector_db"] is None:
            with st.spinner(f"Processing uploaded {file_type} file..."):
                st.session_state["vector_db"] = create_vector_db(file_upload, loader, temp_dir)
        else:
            with st.spinner(f"Adding new {file_type} file to existing vector DB..."):
                add_document_to_vector_db(file_upload, loader, temp_dir)
        st.session_state["uploaded_file_names"].append(file_upload.name)
        # Store the uploaded file in session state
        st.session_state["file_upload"] = file_upload
        # Extract and store PDF pages
        if file_type == "pdf":
            with pdfplumber.open(file_upload) as pdf:
                st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]

    if st.session_state.uploaded_file_names:
        col1.markdown("---")
        col1.markdown("### Uploaded Files:")
        for file_name in st.session_state.uploaded_file_names:
            col1.write(file_name)
        col1.markdown("---")

    # Display PDF if pages are available
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        # PDF display controls
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=700, 
            step=50,
            key="zoom_slider"
        )

        # Display PDF pages
        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

    # Delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection", 
        type="secondary",
        key="delete_button"
    )

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Chat input and processing
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                # Process and display assistant response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                # Add assistant response to chat history
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file or use the sample PDF to begin chat...")


if __name__ == "__main__":
    main()