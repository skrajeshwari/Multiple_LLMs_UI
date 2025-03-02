
# Import required packages

import os
import openai
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from google.colab import userdata
import gradio as gr
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Read OpenAI key from Codespaces Secrets

#api_key = os.environ['OPENAI_NEWKEY']             # <-- change this as per your Codespaces secret's name
#os.environ['OPENAI_API_KEY'] = api_key
#openai.api_key = os.getenv('OPENAI_API_KEY')


from dotenv import load_dotenv

load_dotenv()

# Debug print to check available environment variables
print("Available environment variables:", os.environ)

# Read OpenAI key from Codespaces Secrets
api_key = os.environ.get('OPENAI_NEWKEY')
print(f"Read OpenAI API key: {api_key}")

if api_key:
    print("OpenAI API key loaded successfully.")
else:
    print("Failed to load OpenAI API key.")

os.environ['OPENAI_API_KEY'] = api_key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Verify that OpenAI API key is set correctly
print(f"OpenAI API Key: {openai.api_key}")

# Load LLM Models

llm1_4o = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
llm2_3p5 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm3_4 = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

llms = {"gpt-4o-mini": llm1_4o, "gpt-3.5-turbo": llm2_3p5, "gpt-4-turbo": llm3_4}
llm_names = list(llms.keys())

# Prompt Template
template = """Answer the question given at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Question: {question}
Generate Answer:"""
PROMPT = PromptTemplate(input_variables=["question"], template=template)

# Chat Prompt Template for Conversational Memory
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

# Chat History

# Store to keep track of session histories
session_histories = {}

# Function to Retrieve or Initialize Session History
def get_session_history(session_id: str):
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

# Function to generate a response
def generate_response(prompt, session_id, llm_name=None):
    # Format the prompt with conversation history
    memory = get_session_history(session_id)
    conversation_history = "\n".join([msg.content for msg in memory.messages])
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=f"{conversation_history}\nUser: {prompt}")
    ]

    # Generate response based on selected LLM or all LLMs
    if llm_name:
        llm = llms.get(llm_name, llms['gpt-4o-mini'])  # Default to 'gpt-4o-mini' if llm_name is invalid
        response = llm.invoke(messages)
        memory.add_message(HumanMessage(content=prompt))
        memory.add_message(AIMessage(content=response.content))
        return response.content

    # Otherwise, get responses from all LLMs
    output = ""
    for name in llm_names:
        response = llms[name].invoke(messages)
        output += f"{name}: {response.content}\n"
        memory.add_message(HumanMessage(content=prompt))
        memory.add_message(AIMessage(content=response.content))
    return output

# Clear chat history
def clear_history(session_id):
    session_histories[session_id] = ChatMessageHistory()
    return [("System", "Chat history cleared.")]

# Gradio UI
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## LLM Chat Interface")

        # Session ID and LLM selection
        session_id = gr.Textbox(value="user1", label="Session ID", interactive=True)
        llm_selector = gr.Dropdown(choices=llm_names + ["All"], value="All", label="Choose LLM")

        # User input and chat history display
        user_input = gr.Textbox(label="Enter your query here")
        response_display = gr.Chatbot(label="Assistant Response")

        # Clear chat button
        clear_button = gr.Button("Clear Chat")

        # Define input-output interactions
        def respond(session_id, prompt, llm_name):
            llm = llm_name if llm_name != "All" else None
            response = generate_response(prompt, session_id, llm)
            return [(prompt, response)]

        # Bind Enter key for user input to trigger the response
        user_input.submit(
            respond,
            inputs=[session_id, user_input, llm_selector],
            outputs=response_display,
            queue=False,
        )

        # Clear chat button interaction
        clear_button.click(
            lambda session_id: clear_history(session_id),
            inputs=[session_id],
            outputs=response_display,
        )

    demo.launch()

# Call main to launch Gradio interface
main()
