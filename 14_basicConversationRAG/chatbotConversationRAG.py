import os
import uuid
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.chains.combine_documents import create_stuff_documents_chain

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

# load the document
document_path = '<files_path>'
files = os.listdir(document_path)

# create the document loader
documents = []
for file in files:
    document_loader = UnstructuredMarkdownLoader(document_path + '/' + file)
    documents.extend(document_loader.load())

# split the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
split_documents = splitter.split_documents(documents)

# create the vectorstore
vectorstore = Chroma.from_documents(documents=split_documents, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# create the prompt
system_prompt = (
    """
    You are an assistant for question answering tasks. Your knowledge is foccused on Power BI. Use the following pieces of retrieved context to answer the question. If you don't know the answer, you can say "I don't know. Keep the answers concise.\n\n{context}"
    """
)

contextualize_q_system_prompt = (
    """
    Given a chat history and the latest user question, which might reference context in chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed. Otherwise return it as is.
    """
)

# main prompt
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ('system', system_prompt),
#         ('human', '{input}')
#     ]
# )
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ]
)

# contextualize question prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ]
)

# create the chain
history_aware_retriever = create_history_aware_retriever(chatbot, retriever, contextualize_q_prompt)
question_answering_chain = create_stuff_documents_chain(chatbot, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)

# define a store for the memory
store = {}

def get_session_history(session_id: str, history_to_retrieve: int=3) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    
    return ChatMessageHistory(messages=store[session_id].messages[-history_to_retrieve:])

def chat(question, session_id=None):
    # output = rag_chain.invoke({'input': question, 'chat_history': chat_history})
    output = conversational_rag_chain.invoke(
        {
            'input': question
        },
        config={
            'configurable': {'session_id': session_id},
        }
    )
    return output['answer']

# configure a memory retrieval
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

if __name__ == '__main__':
    print('Seja bem vindo ao chatbot de Power BI e Business Intelligence!')
    session_id = str(uuid.uuid4())
    while True:
        chat_history = []
        question = input('\nYou: ')
        answer = chat(question, session_id=session_id)
        print(f'[BOT]: {answer}')
        # chat_history.extend(
        #     [
        #         HumanMessage(content=question),
        #         AIMessage(content=answer)
        #     ]
        # )
