import os
import time
import warnings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

# define the PDF path
pdfpath = 'C:\\Users\\izelioli\\OneDrive - Deloitte (O365D)\\Documents\\Pessoais\\UsefulProjects\\genAI_LangChain\\12_pdfDocumentQA\\PrincipaisEtapasEngenheirVirtual.pdf'

# create the document loader
doc_loader = PyPDFLoader(pdfpath)
docs = doc_loader.load()

# create the splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# create the vectorstore
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# create the retriever
retriever = vectorstore.as_retriever()

# create system prompt
system_prompt = ("""You are an assistant for question-answering tasks.\nUse the following pieces of retrieved context to answer the question. If you don't know the answer, say you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}""")

# create the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}')
    ]
)

# create the chain
question_answer_chain = create_stuff_documents_chain(chatbot, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# create list of questions about how to build a virtual engineer
questions = [
    'What are the main steps to build this system?',
    'Which steps have the hight count of tasks?',
    'What needs to be done in the FrontEnd step?',
    'What is the main goal of the API/Backend step?',
    'What is the main goal of the Guardrails step?',
    'And what about the Monitoring step?',
    'What is the Active Flow step?',
]

# run the chain
for question in questions:
    print(f'[USER]: {question}')
    response = rag_chain.invoke({'input': question})
    print(response['answer'])
    print('---'*50)
    time.sleep(1)
