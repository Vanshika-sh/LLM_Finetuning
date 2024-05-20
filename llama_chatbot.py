import streamlit as st
from utils.rag_agent_utils import get_doc_tools
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core.objects import ObjectIndex
import ollama  # Importing the hypothetical Ollama library

st.title("RAG Chatbot - Llama3")

# Initialize the Ollama model
model = ollama.load_model("llama3")

# Function to generate responses using Ollama
def generate_response(prompt):
    return ollama.generate(model, prompt)

# Allow users to upload multiple PDF files
uploaded_files = st.file_uploader("Choose PDF files", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    paper_to_tools_dict = {}
    for uploaded_file in uploaded_files:
        # Save uploaded file to disk
        with open(Path(uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Get document tools for each uploaded document
        vector_tool, summary_tool = get_doc_tools(Path(uploaded_file.name).as_posix(), Path(uploaded_file.name).stem)
        paper_to_tools_dict[uploaded_file.name] = [vector_tool, summary_tool]

    # Creating an object index and retriever from these tools
    all_tools = [tool for tools in paper_to_tools_dict.values() for tool in tools]
    obj_index = ObjectIndex.from_objects(all_tools, index_cls=VectorStoreIndex)
    obj_retriever = obj_index.as_retriever(similarity_top_k=3)

    # Initialize conversation history
    if 'conversations' not in st.session_state:
        st.session_state.conversations = []

    # Function to handle user question submission
    def submit_query():
        query = st.session_state.query_input
        tools = obj_retriever.retrieve(query)
        agent_worker = FunctionCallingAgentWorker.from_tools(
            tool_retriever=obj_retriever,
            llm=generate_response,  # Use the Ollama model response function
            system_prompt=""" \
            You are an agent designed to answer queries over a set of given papers.
            Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
            """,
            verbose=True
        )
        agent = AgentRunner(agent_worker)
        response = agent.query(query)
        st.session_state.conversations.append(f"User: {query}")
        st.session_state.conversations.append(f"Agent: {response}")
        st.session_state.query_input = ""

    # Text input for user queries
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""

    # Display all past conversations using markdown for automatic text wrapping
    for exchange in st.session_state.conversations:
        st.markdown(f"**{exchange}**")

    # Create a text input field that allows the user to enter their question
    user_input = st.text_input("Ask a question about the documents:", 
                               value=st.session_state.query_input, 
                               key="query_input",
                               on_change=submit_query)

    # Submit button
    submit_button = st.button("Submit", on_click=submit_query)
else:
    st.warning("Please upload PDF files to proceed.")
    st.stop()
