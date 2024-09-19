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
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# configure page
st.set_page_config(page_title="BMW TALK TO YOUR DATA", page_icon="💬")


# Retrieve the OpenAI API key from environment variables
openai_api_key = st.secrets["OPENAI_API_KEY"]
google_credentials_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
if not openai_api_key or not google_credentials_json:
    st.error("OpenAI API or Google Credentials keys not set in environment variables.")
    st.stop()


# Initialize BigQuery client without caching to avoid authentication token issues
def get_bigquery_client():
    try:
        # Load credentials from the service account file
        credentials = service_account.Credentials.from_service_account_info(
            google_credentials_json
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
        SELECT
            post_url,
            post_caption,
            post_id,
            `model`,
            POST_REACTIONS_ANGER,
            POST_REACTIONS_HAHA,
            POST_REACTIONS_LOVE,
            POST_REACTIONS_LIKE,
            POST_REACTIONS_PRIDE,
            POST_REACTIONS_SORRY,
            POST_REACTIONS_THANKFUL,
            POST_REACTIONS_WOW,
            POST_COMMENTS,
            avg_sentiment_score
            background_description,
            car_angle,
            car_as_main_subject,
            car_color,
            car_movement,
            car_view,
            ev_features,
            human_presence,
        FROM
            `bmw-recommendation-engine.talk_to_your_data.bmw_posts_test`
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
    model="gpt-4o",
    max_tokens=1000,
    openai_api_key=openai_api_key,
)

# Define the prompt template for the assistant
template = """
You are an artificial intelligence assistant and should act as a media data analyst. You will deal with data from Facebook posts. You will have access to the post ID, URL, metrics such as reactions and their respective quantities, number of comments, and the average sentiment of the comments on that post (i.e., performing a sentiment analysis of all the comments on the post, with a score from -1 to 1, where -1 is very negative and 1 is very positive). You should also consider the number of comments, as a post with only 1 comment and a sentiment score of 1 is not that impactful.

The posts are from a car salesperson, so you will also have access to information such as:

- The angle of the car in the post image
- Whether there are humans in the post image
- Description of the image background
- Color of the car in the image
- Whether the car is moving or stationary in the image
- Whether there is a car in the image
- Whether the car is the main focus of the image
- Overall features of the image
- tags_market, which in this case represents the country of the post
- The model of the car in the image
- The post description
- Answer user questions and requests based on the information I will provide as a knowledge base. If a question is asked and you cannot answer it with this information, you should respond with "I don't have this information."

**Conversation history:**
{chat_history}

**Question/request:**
{message}

**Your knowledge base:**
{info}

**Answer:**
"""

# Create a PromptTemplate with the specified input variables
prompt = PromptTemplate(
    input_variables=["chat_history", "message", "info"], template=template
)

# Create the LLMChain with the language model, prompt, and memory
chain = prompt | llm


# Function to generate a response using the language model and retrieved information
def generate_response(message):
    try:
        docs = retrieve_info(message)
        if not docs:
            return "Sorry, i don't have this information"
        info_text = "\n".join([doc.page_content for doc in docs])
        response = chain.invoke(
            {
                "message": message,
                "info": info_text,
                "chat_history": memory.chat_memory.messages,
            }
        )
        return response.content
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {e}"


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
