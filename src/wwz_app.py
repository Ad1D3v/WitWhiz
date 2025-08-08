import streamlit as st
from langchain_core.messages import HumanMessage
import time

# Import Custom Module
from wwz_core import chatbot

# Set the Config for Session State 
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# Define Interface Elements
st.set_page_config(page_title="WitWhiz", page_icon="🧠")
st.title("WitWhiz")
st.subheader("Chat Smart. Chat Swift.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Empty placheolder for custom streaming effect
        response_placeholder = st.empty()
        response = ""
        for ai_message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=prompt)]},
                config={'configurable': {'thread_id': 'thread-1'}},
                stream_mode='messages'
        ):
            if ai_message_chunk.content and metadata['langgraph_node'] == "chat_node":
                # Accumulate chunks of the response
                response += ai_message_chunk.content
                # Update the UI with new chunk
                response_placeholder.markdown(response) 
                # Simulate typing speed
                time.sleep(0.02)  

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})