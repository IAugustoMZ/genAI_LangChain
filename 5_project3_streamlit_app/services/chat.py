import os
from dotenv import load_dotenv
from infra.open_ai_tools import OpenAITools
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ( ChatPromptTemplate, SystemMessagePromptTemplate, 
                               HumanMessagePromptTemplate )

# define the path to the .env file
env_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    ), 'env', '.env'
)
load_dotenv(dotenv_path=env_path)

class ChatAcademicAdvisor:

    # create the templates
    system_template = """
    Use the following pieces of information to answer the user's question:
    Whenever you find a reference to another paper, you'll include it in the answer.
    --------------------------------------------------------------
    Context: {context}
    """
    user_template = """
    Question: {question}
    Chat History: {chat_history}
    """

    def __init__(self):
        """
        class that manages the chatbot
        """
        self.openai = OpenAITools()

    def ask_and_get_questions(self, q: str, store: object, top_k: int=10):
        """
        Ask a question and get the answer

        Parameters:
        ----------
        q: str
            The question to be asked

        store: object
            The object that stores the embeddings
        
        top_k: int, optional
            The number of retrievals to be returned, by default 10
        
        Returns:
        -------
        str
            The answer to the question
        """
        # get the llm model
        llm = self.openai.chat

        # create the retriever
        retriever = store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': top_k}
        )

        # create the memory
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # create the messages
        messages = [
            SystemMessagePromptTemplate.from_template(template=self.system_template),
            HumanMessagePromptTemplate.from_template(template=self.user_template)
        ]

        # create the chat prompt
        qa_prompt = ChatPromptTemplate.from_messages(messages=messages)

        # create the chain
        crc = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type='stuff',
            verbose=True,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': qa_prompt}
        )

        # get the answer
        answer = crc.invoke({'question': q})

        return answer['answer']