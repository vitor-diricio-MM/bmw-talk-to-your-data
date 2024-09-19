# BMW TALK TO YOUR DATA - Streamlit Application

# Import necessary libraries
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# configure page
st.set_page_config(page_title="BMW TALK TO YOUR DATA", page_icon="💬")

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not set in environment variables.")
    st.stop()

# Set the OpenAI API key in the environment variable for OpenAI library use
os.environ["OPENAI_API_KEY"] = openai_api_key

# Directly specify the path to your Google Cloud service account JSON file
# Replace 'service_account.json' with the path to your service account file
google_credentials_path = "service_account.json"  # Update this path as needed

# Check if the service account file exists
if not os.path.exists(google_credentials_path):
    st.error(
        f"Google Application Credentials file not found at {google_credentials_path}."
    )
    st.stop()


# Initialize BigQuery client without caching to avoid authentication token issues
def get_bigquery_client():
    try:
        # Load credentials from the service account file
        credentials = service_account.Credentials.from_service_account_file(
            google_credentials_path
        )
        # Initialize the BigQuery client with the loaded credentials
        client = bigquery.Client(
            credentials=credentials, project=credentials.project_id
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize BigQuery client: {e}")
        st.error(f"Error initializing BigQuery client: {e}")
        return None


# Create the BigQuery client
client = get_bigquery_client()


# Load data from BigQuery without caching to ensure fresh data and avoid credential issues
def load_data():
    try:
        query = """
        SELECT * FROM `bmw-recommendation-engine.talk_to_your_data.bmw_posts_test`
        """
        query_job = client.query(query)
        results = query_job.result()
        data = [dict(row.items()) for row in results]  # Each row as a dict
        # Optionally display data for debugging
        # st.text(data)
        return data
    except Exception as e:
        logger.error(f"Failed to load data from BigQuery: {e}")
        st.error(f"Error loading data from BigQuery: {e}")
        return []


# Load the data from BigQuery
data = load_data()


# Initialize OpenAI embeddings and FAISS vector store
def initialize_vector_store(data):
    try:
        embeddings = OpenAIEmbeddings()
        documents = []
        for row in data:
            # Create a text representation of the row by concatenating key-value pairs
            content = " ".join(
                [f"{key}: {value}" for key, value in row.items() if value is not None]
            )
            doc = Document(page_content=content, metadata=row)
            documents.append(doc)
        # Create a FAISS vector store from the documents
        db = FAISS.from_documents(documents, embeddings)
        return db
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        st.error(f"Error initializing vector store: {e}")
        return None


# Initialize the vector store with the loaded data
db = initialize_vector_store(data)


# Define a function to retrieve information based on similarity search
def retrieve_info(query, k=5):
    try:
        similar_responses = db.similarity_search(query, k=k)
        return similar_responses  # Return the list of similar documents
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        return []


# Setup ConversationBufferMemory to keep track of conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup the language model for generating responses
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-16k-0613",
    max_tokens=1000,
    openai_api_key=openai_api_key,
)

# Define the prompt template for the assistant
template = """
Eres un asistente de inteligencia artificial que proporciona soporte a los empleados de Banco Santa Fe. Banco Santa Fe es un banco argentino.

Se te proporcionará una lista de información relacionada con Banco Santa Fe. Responde a la pregunta del empleado basándote únicamente en la información proporcionada. Si la pregunta no está cubierta en la información, responde con "Lo siento. No tengo esta información."

Sigue estas directrices:
- Siempre responde en español.
- Si no estás seguro de la respuesta, di "No sé" y pide una pregunta más detallada.
- Proporciona respuestas completas e informativas sin ser excesivamente verboso.

**Historial de conversación:**
{chat_history}

**Pregunta del empleado:**
{message}

**Información:**
{info}

**Respuesta:**
"""

# Create a PromptTemplate with the specified input variables
prompt = PromptTemplate(
    input_variables=["chat_history", "message", "info"], template=template
)

# Create the LLMChain with the language model, prompt, and memory
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)


# Function to generate a response using the language model and retrieved information
def generate_response(message):
    try:
        docs = retrieve_info(message)
        if not docs:
            return "Lo siento. No tengo esta información."
        # Concatenate the content of retrieved documents
        info_text = ""
        for doc in docs:
            info_text += doc.page_content + "\n"
        # Generate the response using the language model chain
        response = chain.run(message=message, info=info_text)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Error obteniendo la respuesta."


# Build the Streamlit application interface
def main():
    st.header("BMW TALK TO YOUR DATA")
    st.write("Talk to your posts data...")

    # Initialize session state for conversation history if not already initialized
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Display the conversation history
    for chat in st.session_state.conversation:
        if chat["role"] == "user":
            st.markdown(f"**You:** {chat['content']}")
        else:
            st.markdown(f"**Chatbot:** {chat['content']}")

    # Get user input from a text input widget
    message = st.text_input("Write a question:")

    # When the "Send" button is clicked and the message is not empty
    if st.button("Send") and message.strip() != "":
        with st.spinner("Generating answer..."):
            # Generate a response and update the conversation history
            response = generate_response(message)
            st.session_state.conversation.append({"role": "user", "content": message})
            st.session_state.conversation.append(
                {"role": "assistant", "content": response}
            )
            st.rerun()

    # Clear the conversation history when the "Clear conversation" button is clicked
    if st.button("Clear conversation"):
        st.session_state.conversation = []
        st.rerun()


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
