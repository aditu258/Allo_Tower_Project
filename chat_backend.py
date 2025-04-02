import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import firebase_admin
from firebase_admin import credentials, db
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import SystemMessage
from typing import List, Dict, Any, Optional
import time
from dotenv import load_dotenv
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
INDEX_NAME = os.getenv("INDEX_NAME", "test")  # Default value is "test"
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
DATABASE_URL = os.getenv("DATABASE_URL")

class ChatSystem:
    def __init__(self, model, retriever, ref, chat_history=None):
        self.model = model
        self.retriever = retriever
        self.ref = ref
        self.chat_history = chat_history or []

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred, {
                'databaseURL': DATABASE_URL
            })
        return db.reference('/')
    except Exception as e:
        raise RuntimeError(f"Firebase initialization failed: {str(e)}")

def get_pinecone_client() -> Pinecone:
    """Initialize and return Pinecone client"""
    try:
        return Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        raise RuntimeError(f"Pinecone client initialization failed: {str(e)}")

def check_index_has_data(pc: Pinecone, index_name: str) -> bool:
    """Check if Pinecone index contains data"""
    try:
        if index_name not in [index["name"] for index in pc.list_indexes()]:
            return False
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        return stats['total_vector_count'] > 0
    except Exception as e:
        print(f"Warning: Error checking index data: {str(e)}")
        return False

def load_documents(documents_dir: str) -> list:
    """Load documents from directory"""
    try:
        documents = []
        book_files = [f for f in os.listdir(documents_dir) if f.endswith(".txt")]
        print(f"Found {len(book_files)} text files in directory")
        
        for book_file in book_files:
            file_path = os.path.join(documents_dir, book_file)
            loader = TextLoader(file_path)
            book_docs = loader.load()
            for doc in book_docs:
                doc.metadata = {"source": book_file}
                documents.append(doc)
        return documents
    except Exception as e:
        raise RuntimeError(f"Error loading documents: {str(e)}")

def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 0) -> list:
    """Split documents into chunks"""
    try:
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        raise RuntimeError(f"Error splitting documents: {str(e)}")

def initialize_pinecone_index(pc: Pinecone, index_name: str, dimension: int = 384) -> None:
    """Initialize Pinecone index if it doesn't exist"""
    try:
        existing_indexes = [index["name"] for index in pc.list_indexes()]
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists. Skipping creation.")
    except Exception as e:
        raise RuntimeError(f"Error initializing Pinecone index: {str(e)}")

def create_vector_store(docs: list, embeddings, index_name: str, skip_if_exists: bool = True):
    """Create Pinecone vector store"""
    try:
        pc = get_pinecone_client()
        
        if skip_if_exists and check_index_has_data(pc, index_name):
            print("Index already contains data - skipping document insertion.")
            os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
            return PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embeddings
            )
        
        print("Creating new vector store with documents...")
        return PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=index_name
        )
    except Exception as e:
        raise RuntimeError(f"Error creating vector store: {str(e)}")

def initialize_chat_model():
    """Initialize the chat model with Gemini"""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_output_tokens=4048,
            timeout=120,
            max_retries=2,
            google_api_key=GOOGLE_API_KEY
        )
    except Exception as e:
        raise RuntimeError(f"Error initializing chat model: {str(e)}")

def deserialize_chat_history(stored_chat_history: List[Dict[str, Any]]) -> List[Any]:
    """Convert stored chat history to LangChain message objects"""
    try:
        deserialized_messages = []
        for message in stored_chat_history:
            role = message.get("role")
            content = message.get("content", "")
            if content:
                if role == "human":
                    deserialized_messages.append(HumanMessage(content=content))
                elif role == "ai":
                    deserialized_messages.append(AIMessage(content=content))
                elif role == "system":
                    deserialized_messages.append(SystemMessage(content=content))
        return deserialized_messages
    except Exception as e:
        print(f"Warning: Error deserializing chat history: {str(e)}")
        return [SystemMessage(content="You are a helpful assistant")]

def serialize_chat_history(chat_history: list) -> list:
    """Convert LangChain message objects to serializable format"""
    try:
        return [
            {"role": "human", "content": msg.content} if isinstance(msg, HumanMessage)
            else {"role": "ai", "content": msg.content}
            for msg in chat_history
            if isinstance(msg, (HumanMessage, AIMessage))
        ]
    except Exception as e:
        print(f"Warning: Error serializing chat history: {str(e)}")
        return []

def load_chat_history(ref):
    """Load chat history from Firebase if it exists"""
    try:
        stored_chat_history = ref.get() or []
        chat_history = deserialize_chat_history(stored_chat_history)
        
        if not any(isinstance(msg, SystemMessage) for msg in chat_history):
            chat_history.insert(0, SystemMessage(content="You are a helpful assistant"))
        
        return chat_history
    except Exception as e:
        print(f"Warning: Error loading chat history: {str(e)}")
        return [SystemMessage(content="You are a helpful assistant")]

def initialize_chat_system() -> ChatSystem:
    """Initialize the complete chat system"""
    try:
        ref = initialize_firebase()
        os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
        pc = get_pinecone_client()
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        initialize_pinecone_index(pc, INDEX_NAME)
        
        if not check_index_has_data(pc, INDEX_NAME):
            documents = load_documents(DOCUMENTS_DIR)
            docs = split_documents(documents)
            print(f"Processing {len(docs)} document chunks")
            vector_store = create_vector_store(docs, embeddings, INDEX_NAME, skip_if_exists=False)
        else:
            print("Using existing Pinecone index data")
            vector_store = PineconeVectorStore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=embeddings
            )
        
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        model = initialize_chat_model()
        chat_history = load_chat_history(ref)
        
        return ChatSystem(model, retriever, ref, chat_history)
    except Exception as e:
        print(f"Error initializing chat system: {str(e)}")
        raise

def get_chat_response(chat_system: ChatSystem, query: str) -> str:
    """Get response from the chat system"""
    try:
        retrieved_docs = chat_system.retriever.invoke(query)
        retrieved_texts = "\n".join([doc.page_content for doc in retrieved_docs])

        chat_system.chat_history.append(HumanMessage(content=query))
        chat_system.chat_history.append(SystemMessage(content=f"Retrieved Context:\n{retrieved_texts}"))

        result = chat_system.model.invoke(chat_system.chat_history)
        
        if result.content.strip():
            chat_system.chat_history.append(AIMessage(content=result.content))
            serialized = serialize_chat_history(chat_system.chat_history)
            chat_system.ref.set(serialized[-50:])  # Keep last 50 messages
            return result.content
        
        return "I couldn't generate a response. Please try again."
    except Exception as e:
        print(f"Error getting chat response: {str(e)}")
        return "Sorry, I encountered an error. Please try again."

def reset_chat_history(chat_system: ChatSystem):
    """Reset chat history while preserving system message"""
    try:
        system_messages = [msg for msg in chat_system.chat_history if isinstance(msg, SystemMessage)]
        chat_system.chat_history = system_messages if system_messages else [SystemMessage(content="You are a helpful assistant")]
        chat_system.ref.set(serialize_chat_history(chat_system.chat_history))
    except Exception as e:
        print(f"Error resetting chat history: {str(e)}")
        raise