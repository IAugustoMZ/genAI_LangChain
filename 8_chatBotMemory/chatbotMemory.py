import os
import uuid
import time
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# define path to the .env file
env_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
   '.env'
)
load_dotenv(dotenv_path=env_path, override=True)

# ignore the warnings
warnings.filterwarnings("ignore")

# create the chatbot
chatbot = ChatGroq(
    model='llama-3.1-70b-versatile',
    temperature=0,
    max_tokens=None,
    max_retries=2,
)

# create the chatbot memory
chatbotMemory = {}

def get_chatbot_memory(session_id) -> BaseChatMessageHistory:
    """
    Get the chatbot memory for the given session_id
    """
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]

# create the chatbot with memory
chatbotWithMemory = RunnableWithMessageHistory(
    chatbot,
    get_chatbot_memory
)

# create the chatbot session
session1 = {'configurable': {'session_id': str(uuid.uuid4())}}

# # get the first response
# firstMsg = 'I am a chemical engineer'
# print(f'[USER]: {firstMsg}')
# response = chatbotWithMemory.invoke(
#     [HumanMessage(firstMsg)],
#     config=session1
# )

# print('[CHATBOT]: ', response.content)

# # get the second response
# secondMsg = 'My fiancee is a teacher'
# print(f'[USER]: {secondMsg}')
# response = chatbotWithMemory.invoke(
#     [HumanMessage(secondMsg)],
#     config=session1
# )

# print('[CHATBOT]: ', response.content)

# # get the third response
# thirdMsg = 'What is my profession?'

# print(f'[USER]: {thirdMsg}')
# response = chatbotWithMemory.invoke(
#     [HumanMessage(thirdMsg)],
#     config=session1
# )

# print('[CHATBOT]: ', response.content)

# # let's try to change the session
# session2 = {'configurable': {'session_id': str(uuid.uuid4())}}

# # ask my fiancee's profession
# fourthMsg = 'What is my fiancee profession?'
# print(f'[USER]: {fourthMsg}')
# response = chatbotWithMemory.invoke(
#     [HumanMessage(fourthMsg)],
#     config=session2
# )

# print('[CHATBOT]: ', response.content)

# # important to manage the memory
# print(chatbotMemory)

print('='*100)

# let's try to limit the memory
def limited_chatbot_memory(messages, num_messages=2):
    """
    Limit the chatbot memory to the given number of messages
    """
    return messages[-num_messages:]

# create a prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a very helpful industrial data analyst and your job is query industrial data. Suppose you have access to a map between the names of the variables and their tags; and also to the process history database, which is queried through the tag. The industrial database has three columns: timestamp, tag and value. Explain, step-by-step, your thinking process to answer the user question. If you dont know the answer or cannot identify the variable, just answer "I cannot answer your question", but dont make any comments.'),
        MessagesPlaceholder(variable_name='messages')
    ]
)

# create the chain
limitedChatBotChain = (
    RunnablePassthrough.assign(messages=lambda x: limited_chatbot_memory(x['messages']))
    | prompt
    | chatbot
)

# create the chatbot session
chatbotLimitedMemory = RunnableWithMessageHistory(
    limitedChatBotChain,
    get_chatbot_memory,
    input_messages_key='messages'
)

# create the chatbot list of messages
listMessages = [
    HumanMessage(content='Qual o maior valor de temperatura do forno 1 na última semana?'),
    HumanMessage(content='E a média?'),
    HumanMessage(content='E a média da semana anterior?'),
    HumanMessage(content='E o valor mínimo?'),
    HumanMessage(content='E o valor mínimo da semana anterior?'),
]

# create the chatbot session
def create_session():
    return {'configurable': {'session_id': str(uuid.uuid4())}}

session = create_session()

for msg in listMessages:
    print(f'[USER]: {msg.content}')
    response = chatbotLimitedMemory.invoke(
        {'messages':[msg]},
        config=session
    )
    print('[CHATBOT]: ', response.content)
    time.sleep(1)