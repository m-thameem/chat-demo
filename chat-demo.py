import getpass
import streamlit as st
import time
import random
import os
from dotenv import load_dotenv
from typing import Sequence

import chardet

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.messages import HumanMessage, AIMessage


import json

st.title("Foolhu Claude Test")
anthropic_key = st.sidebar.text_input('Anthropic API Key', type='password')

class Transliterator:
    def __init__(self):
        self.dictionary = self.load_dictionary()
        self.all_akuru_fili = self.load_config('all_akuru_fili')
        self.fili_fix = self.load_config('fili_fix')
        self.punctuations = self.load_config('punctuations')
        self.text = ""

    def load_dictionary(self):
        with open('dictionary/translations.json', 'r') as file:
            return json.load(file)

    def load_config(self, config_name):
        with open(f'configs/{config_name}.json', 'r') as file:
            return json.load(file)

    def replace_zero_width_non_joiners(self):
        # replace zero-width non-joiners
        self.text = self.text.replace('\u200C', '')

    def replace_from_text(self, replacements):
        # replace text based on replacements
        for key, value in replacements.items():
            self.text = self.text.replace(key, value)

    def capitalize_first_letter_of_new_sentence(self):
        # capitalize the first letter of a new sentence
        sentences = self.text.split('. ')
        sentences = [s.capitalize() for s in sentences]
        self.text = '. '.join(sentences)

    def transliterate(self, input_text: str) -> str:
        self.text = input_text
        
        self.replace_zero_width_non_joiners()

        # fix words that normally don't translate well
        # like names and English words.
        self.replace_from_text(self.dictionary)

        # fili_fix first before replacing akuru and fili
        self.replace_from_text(self.fili_fix)

        # akuru and fili
        self.replace_from_text(self.all_akuru_fili)

        # punctuations
        self.replace_from_text(self.punctuations)

        # capitalize every letter AFTER a full-stop (period).
        self.capitalize_first_letter_of_new_sentence()

        return self.text.capitalize()

transliterator = Transliterator()

if not anthropic_key:
    raise ValueError("Please enter your Anthropic API key in the sidebar")
os.environ["ANTHROPIC_API_KEY"] = anthropic_key


model = ChatAnthropic(model="claude-3-5-sonnet-20240620")


with open("system_prompt.txt", "r") as file:
    system_prompt = file.read()

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

workflow = StateGraph(state_schema=State)

def call_model(state: State):
    trimmed_messages = state["messages"]
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

def chat_with_bot(user_message: str) -> str:
    # Convert session messages to LangChain message format
    history_messages = []
    for msg in st.session_state.messages[:-1]:  # Exclude the latest message
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        else:
            history_messages.append(AIMessage(content=msg["content"]))
    
    # Add the current message
    history_messages.append(HumanMessage(content=user_message))
    
    response_content = ""
    for chunk, metadata in app.stream(
        {"messages": history_messages},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            response_content += chunk.content
            
    return response_content

def response_generator():
    response = chat_with_bot(st.session_state.messages[-1]["content"])
    print(response)
    print(chardet.detect(response.encode()).get("encoding"))

    if chardet.detect(response.encode()).get("encoding") == "utf-8":
        response = transliterator.transliterate(response)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})