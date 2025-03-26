import os
import streamlit as st
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

groq_api_key = st.secrets["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"]
llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-specdec")

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an educational assistant expert, help in notes, quiz, explaining of concepts and other things. Listen to user's query and provide solution accordingly"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)


st.title("Shiksha AI")
st.write("Welcome! I'm here to help. Feel free to share your doubts")

# Sidebar with resources
with st.sidebar:
    st.header("Shiksha AI")
    st.write("Tidal Techies")


# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("How are you feeling today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate bot response
    with st.chat_message("assistant"):
        response = conversation.invoke({"input": prompt})
        st.markdown(response['text'])

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
# Add this at the BOTTOM of your code (after the chat input section)

# Footer with developer credit
# Add this at the BOTTOM of your code
# Add this at the BOTTOM of your code
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        right: 0;
        bottom: 0;
        margin-right: 1rem;
        margin-bottom: 1rem;
        z-index: 1000;
        background-color: transparent;
        text-align: right;
    }
    </style>
    <div class="footer">
        Developed by <a href="https://pratirath06.github.io/" target="_blank">Pratirath Gupta</a>
    </div>
    """,
    unsafe_allow_html=True
)
