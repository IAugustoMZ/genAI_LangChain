"""
This is the implementation of the custom chatgpt app.
as the first project of the course
"""
#%%
import os
import warnings
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

# define the path to the .env file
env_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'env', '.env')

# load the environment variables
load_dotenv(dotenv_path=env_path, override=True)

# ignore the warnings
warnings.filterwarnings("ignore")

# %%
# instantiate the chatgpt model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)

# create the chat history
history = FileChatMessageHistory(
    file_path='chat_history.json'
)

# message memory
memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=history,
    return_messages=True
)

# create the prompt template
prompt = ChatPromptTemplate(
    input_variables=['content'],
    messages=[
        SystemMessage(content='You are a chatbot having a conversation with a human.'),
        MessagesPlaceholder(variable_name='chat_history'),  # where the memory will be inserted
        HumanMessagePromptTemplate.from_template('{content}')
    ]
)

# create the chat chain
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# %%
# create the loop for the chat
while True:
    content = input("You: ")

    if content in ['exit', 'quit', 'bye']:
        print('Chatbot: Goodbye!')

        break

    response = chain.run({'content': content})
    print(f'Chatbot: {response}')
    print('-'*50)