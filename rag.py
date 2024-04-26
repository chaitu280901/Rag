import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to create RAG Sage pipeline
def create_rag_sage_pipeline(retriever, chat_model, user_input):
    # Define chat prompt template
    chat_template = ChatPromptTemplate.from_messages([
        # System Message Prompt Template
        SystemMessage(content="""You are a helpful assistant, trained to provide accurate and relevant information based on the context provided.
        Your answers should be formatted in markdown for better readability. """),
        # Human Message Prompt Template
        HumanMessagePromptTemplate.from_template("""Context:
    {context}

    Question:
    {question}

    Answer: """)
    ])

    # Define output parser
    output_parser = StrOutputParser()

    # Define RAG Sage pipeline
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )

    # Invoke RAG Sage pipeline with user input
    return rag_chain.invoke(user_input)

# Set up Streamlit app
st.title("RAG Sage: The Document Genius")

# Load PDF document
loader = PyPDFLoader('arxiv.pdf')
data = loader.load()
chunks = loader.load_and_split()

# Create embeddings for document chunks
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")

# Set up connection with ChromaDB
retriever = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model).as_retriever(search_kwargs={"k": 5})

# Get user input
user_input = st.text_input("Enter your question here...")

# Generate response button
if st.button("Generate Response"):
    response = create_rag_sage_pipeline(retriever, ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-pro-latest", convert_system_message_to_human=True), user_input)
    st.write(response)
