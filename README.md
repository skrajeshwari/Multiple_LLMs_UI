# Multiple_LLMs_UI

An interface with multiple options of LLMs provided to the user. 

# Add OPENAI_API_KEY in codespaces

# Run in Bash

pip install openai langchain-openai langchain-core langchain-community sentence-transformers
pip install gradio
python main.py

# Install Dependancies

openai: This package allows access to OpenAI’s models and APIs.

langchain-openai, langchain-core, langchain-community: These packages are part of LangChain, a framework to build applications using language models. They provide tools for working with OpenAI models, core utilities, and additional community-contributed tools.

sentence-transformers: A library for computing dense vector embeddings of text, useful for tasks like semantic search or similarity matching.

pip install gradio: Installs Gradio, a Python library for creating a user interface (UI) for machine learning models. Gradio makes it easy to design interactive web-based UIs without needing front-end development skills.

# Import required packages

# Set OpenAI API Key

api_key = userdata.get('OPENAI_KEY'):

This retrieves the API key for OpenAI from user data. It assumes that userdata is a dictionary-like object containing the key under 'OPENAI_KEY'.

os.environ['OPENAI_API_KEY'] = api_key:

This line stores the API key in the environment variable OPENAI_API_KEY, which makes it accessible to any process that needs it and is a common way to handle sensitive information securely.

openai.api_key = os.getenv('OPENAI_API_KEY'):

Here, the API key is set for use with the openai package by retrieving it from the environment variable using os.getenv(). This statement ensures the openai library can authenticate API requests with the correct key.

# Loading the LLM Models 

Defining and initializing three different language models from OpenAI, setting them up with specific configurations for later use:

Model Definitions:

llm1_4o = ChatOpenAI(model_name="gpt-4o-mini", temperature=0):

Creates an instance of the ChatOpenAI model with model_name="gpt-4o-mini" and temperature=0.
"gpt-4o-mini" seems to be a custom or experimental model variant based on GPT-4 but simplified for specific tasks.
temperature=0 ensures deterministic, consistent responses, making it ideal for precise and non-random output.
llm2_3p5 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0):

This defines an instance of gpt-3.5-turbo (a highly efficient model in the GPT-3.5 series) with a temperature of 0 for deterministic behavior.
llm3_4 = ChatOpenAI(model_name="gpt-4-turbo", temperature=0):

This initializes the "gpt-4-turbo" model, another advanced variant of GPT-4 designed for optimized performance and efficient responses. Again, temperature=0 ensures consistent answers.
Creating a Dictionary of Models:

llms = {"gpt-4o-mini": llm1_4o, "gpt-3.5-turbo": llm2_3p5, "gpt-4-turbo": llm3_4}:
Stores the initialized models in a dictionary, where each model’s name is a key, allowing for easy access by name. This is useful for dynamically selecting a model by its name.
Extracting Model Names:

llm_names = list(llms.keys()):
Extracts the keys (model names) from the llms dictionary into a list. This list of model names can be used to present options to a user or select a model by its name.
This setup enables easy switching between multiple LLMs based on their configurations and names.

# Prompt Template 

This code defines a prompt template for generating responses using the language models. Here’s a breakdown of each part:

Template Definition:

template = """...""":
This multi-line string defines the structure of the prompt that will be fed into the language model.
It instructs the model on how to respond to a specific question. The template contains two key parts:
A directive for the model to provide an answer based on the question.
A specific instruction that if the model doesn't know the answer, it should state that instead of fabricating a response.
Prompt Construction:

Question: {question}:
This is a placeholder for the actual question that will be inserted into the prompt when it is used. The {question} will be replaced with the input question at runtime.
PROMPT = PromptTemplate(...):

PromptTemplate(input_variables=["question"], template=template):
This line creates an instance of the PromptTemplate class, which is likely a part of the LangChain library.
The input_variables parameter specifies that the template will take one variable, question, which will replace the placeholder in the template.
The template parameter passes the defined prompt structure.
This prompt template can be reused to generate consistent and structured prompts for the language models, ensuring that the model responds appropriately to questions.

# Chat Prompt Template

This code sets up a chat prompt template designed for a conversational memory system, enabling the model to remember previous interactions. Here’s a breakdown of each part:

ChatPromptTemplate.from_messages([...]):

This method is likely part of the LangChain library, which constructs a prompt for a chat-based interaction. It takes a list of messages to define the structure of the conversation.
Messages List:

The list provided to from_messages contains tuples, each representing a different role in the conversation:

("system", "You are a helpful assistant. Answer all questions to the best of your ability."):

This message sets the system’s behavior, establishing the model as a helpful assistant. It indicates that the assistant should aim to provide accurate and informative answers.
("placeholder", "{chat_history}"):

This is a placeholder for the chat history. When the prompt is used, {chat_history} will be replaced with the previous messages exchanged in the conversation. This allows the model to consider context from prior interactions, enhancing its ability to provide relevant responses.
("human", "{input}"):

This message represents the user input. The placeholder {input} will be replaced with the current input from the user when the prompt is generated. It allows the model to respond to the latest user query while taking the previous context into account.

Overall, this chat prompt template is structured to facilitate ongoing dialogue by integrating the chat history and the user’s current input, allowing the assistant to maintain context and provide more coherent and contextually aware responses.

# Chat History

# Generate Response

This function generates responses to user prompts while keeping track of conversation history. Here’s a concise breakdown:

def generate_response(prompt, session_id, llm_name=None)::

Defines a function to generate a response based on the given prompt, session ID, and optionally a specific language model (LLM).
memory = get_session_history(session_id):

Retrieves the chat history for the specified session using the previously defined function.
conversation_history = "\n".join([msg.content for msg in memory.messages]):

Compiles the conversation history into a single string for context in the response.
messages = [...]:

Prepares the messages for the LLM:
A system message indicating the assistant's role.
A human message that includes the conversation history and the current user prompt.
if llm_name::

Checks if a specific LLM is requested.
Retrieves the corresponding LLM from the llms dictionary, defaulting to 'gpt-4o-mini' if the name is invalid.
Generates a response using the selected LLM and updates the session history with the user's prompt and the AI's response.
else::

If no specific LLM is provided, iterates over all available LLMs.
For each LLM, it generates a response, appends it to the output string, and updates the session history with the user's prompt and the AI's response.

output:

Returns the combined responses from all LLMs if none was specified, or the single response from the chosen LLM if one was specified.

# Building the chat interface using Gradio

 components of code:

with gr.Blocks() as demo::

Initializes a Gradio interface using blocks, allowing for flexible layout and component organization.
User Interface Elements:

gr.Markdown("## LLM Chat Interface"): Displays a title for the interface.
session_id = gr.Textbox(...): A textbox for the user to input a session ID, with a default value of "user1".
llm_selector = gr.Dropdown(...): A dropdown menu for selecting the language model (LLM), including an option to choose "All" models.
user_input = gr.Textbox(...): A textbox for the user to enter their query.
response_display = gr.Chatbot(...): A chat display area for showing the assistant's responses.
clear_button = gr.Button("Clear Chat"): A button to clear the chat history.
Response Handling:

def respond(session_id, prompt, llm_name)::
Defines a function to generate a response based on the session ID, user prompt, and selected LLM.
If "All" is selected, it passes None to the generate_response function.
Input-Output Binding:

user_input.submit(...): Binds the Enter key press to trigger the respond function, taking inputs from the session ID, user input, and LLM selector, and displaying the response in response_display.
Clear Button Interaction:

clear_button.click(...): Binds the clear button to a function that clears the chat history for the specified session, updating the response_display.
demo.launch():

Launches the Gradio interface, making it accessible for user interaction.
This code creates an interface that allows users to interact with multiple LLMs and maintain session history in a chat format.
