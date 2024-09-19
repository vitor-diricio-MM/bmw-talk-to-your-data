import os
import logging
import streamlit as st
from dotenv import load_dotenv
from google.cloud import bigquery
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()

# Set OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI"]["API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# Set Google Application Credentials from Streamlit secrets
google_credentials = st.secrets["GOOGLE"]["CREDENTIALS"]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials


# Initialize BigQuery client
@st.cache_resource
def get_bigquery_client():
    try:
        client = bigquery.Client()
        return client
    except Exception as e:
        logger.error(f"Failed to initialize BigQuery client: {e}")
        st.error("Error initializing BigQuery client.")
        return None


client = get_bigquery_client()


# Load FAQs from BigQuery
@st.cache_data
def load_faqs():
    try:
        query = """
        SELECT faq_text FROM `your_project.your_dataset.faq_table`
        """
        query_job = client.query(query)
        results = query_job.result()
        faqs = [row["faq_text"] for row in results]
        return faqs
    except Exception as e:
        logger.error(f"Failed to load FAQs from BigQuery: {e}")
        st.error("Error loading FAQs from BigQuery.")
        return []


faqs = load_faqs()


# Initialize OpenAI embeddings and FAISS vector store
@st.cache_resource
def initialize_vector_store(faqs):
    try:
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(faqs, embeddings)
        return db
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        st.error("Error initializing vector store.")
        return None


db = initialize_vector_store(faqs)


# Define retrieval function
def retrieve_info(query, k=5):
    try:
        similar_responses = db.similarity_search(query, k=k)
        return [doc.page_content for doc in similar_responses]
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        return []


# Setup ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup LLMChain & prompts
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-16k-0613",
    max_tokens=1000,
    openai_api_key=openai_api_key,
)

template = """
Eres un asistente de inteligencia artificial que proporciona soporte a los empleados de Banco Santa Fe. Banco Santa Fe es un banco argentino.

Se te proporcionará una lista de preguntas frecuentes relacionadas con Banco Santa Fe. Responde a la pregunta del empleado basándote únicamente en las preguntas frecuentes proporcionadas. Si la pregunta no está cubierta en las preguntas frecuentes, responde con "Lo siento. No tengo esta información."

Sigue estas directrices:
- Siempre responde en español.
- Si no estás seguro de la respuesta, di "No sé" y pide una pregunta más detallada.
- Proporciona respuestas completas e informativas sin ser excesivamente verboso.

**Historial de conversación:**
{chat_history}

**Pregunta del empleado:**
{message}

**Preguntas Frecuentes:**
{faq}

**Respuesta:**
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "message", "faq"], template=template
)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)


# Retrieval augmented generation with context
def generate_response(message):
    try:
        faq = retrieve_info(message)
        if not faq:
            return "Lo siento, no pude encontrar información relevante."
        faq_text = "\n".join(faq)
        response = chain.run(message=message, faq=faq_text)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Lo siento, ocurrió un error al procesar tu solicitud."


# Build the Streamlit app
def main():
    st.set_page_config(page_title="Chatbot Banco Santa Fe", page_icon="💬")

    st.header("Chatbot Banco Santa Fe")
    st.write(
        "Interactúa con el chatbot para obtener respuestas basadas en las preguntas frecuentes de Banco Santa Fe."
    )

    # Initialize session state for conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Display conversation history
    for chat in st.session_state.conversation:
        if chat["role"] == "user":
            st.markdown(f"**Tú:** {chat['content']}")
        else:
            st.markdown(f"**Chatbot:** {chat['content']}")

    # User input
    message = st.text_input("Escribe tu pregunta:", key="input")

    if st.button("Enviar") and message.strip() != "":
        with st.spinner("Generando una respuesta..."):
            response = generate_response(message)
            st.session_state.conversation.append({"role": "user", "content": message})
            st.session_state.conversation.append(
                {"role": "assistant", "content": response}
            )
            st.experimental_rerun()

    # Clear conversation
    if st.button("Limpiar conversación"):
        st.session_state.conversation = []
        st.experimental_rerun()


if __name__ == "__main__":
    main()
