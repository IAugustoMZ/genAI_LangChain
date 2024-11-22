import os
import uvicorn
import warnings
from fastapi import FastAPI
from dotenv import load_dotenv
from langserve import add_routes
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

# create messages to the LLM
messagestoLLM = [
    ('system', 'You are a teacher and chemical engineering expert. You have to help student about chemical engineering knowledge. So take the provided context and explain the concept to the student, in a simple and concise way. Assume the student has never got in touch with it. If you don\'t know the answer, you can say "I don\'t know".If the concept is not related to chemical engineering, please inform the user. Respond in the same language of the user.'),
    ('user', '{concept}')
]
prompt = ChatPromptTemplate.from_messages(messagestoLLM)

# create the chain
chain = prompt | chatbot | StrOutputParser()

# create the app
app = FastAPI(
    title='Chemical Engineering Chatbot',
    description='A chatbot that answers questions about chemical engineering.',
    version='1.0.0'
)

# add the routes
add_routes(app, chain, path='/chat')

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)