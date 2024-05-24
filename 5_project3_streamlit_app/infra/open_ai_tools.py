import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# define the path to the .env file
env_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    ), 'env', '.env'
)
load_dotenv(dotenv_path=env_path)

class OpenAITools:
    def __init__(self):
        """
        class that connects to the OpenAI API
        """
        self.embeddings = self.create_embeddings_model()
        self.chat = self.create_chat_model()

    def create_embeddings_model(self):
        """
        create a model that generates embeddings
        """
        return OpenAIEmbeddings(
            model='text-embedding-3-small',
            dimensions=1536
        )
    
    def create_chat_model(self):
        """
        create a model that generates responses
        """
        return ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0.5
        )
        