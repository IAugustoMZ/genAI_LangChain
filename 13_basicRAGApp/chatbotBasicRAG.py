import os
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader



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
document_path = '<documents_path>'
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

# create prompt
prompt1  = ChatPromptTemplate(
    input_variables=['context', 'question'],
    metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, 
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'],
                template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to build a multiple question regarding the context. Outout a single question and the multiple alternatives. Don't answer the question. Don't create ambiguous answers. Each alternative should be distinct in meaning from another\nContext: {context}. \nQuestion:"
            )
        )
    ]
)


prompt2  = ChatPromptTemplate(
    input_variables=['context', 'user_input'],
    metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, 
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'user_input'],
                template="You are an assistant for question-answering tasks. You'll  receive the alternative chosen by the user. Use the question, the context and the information provided by the user to respond if the user is right or wrong. \nUser Answer: {user_input} \nContext: {context}\nYour diagnostic:"
            )
        )
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# create the chains
rag1_chain = (
    {'context': retriever | format_docs}
    | prompt1
    | chatbot
    | StrOutputParser()
)

rag2_chain = (
    {'context': retriever | format_docs, 'user_input': RunnablePassthrough()}
    | prompt2
    | chatbot
    | StrOutputParser()
)

# run the chain
list_mgs = [
    'BI',
    'Power BI',
    'Big Data',
    'Data Science',
    'Power Platform',
    'Power BI family',
    'Power BI interface',
    'Data Modeling',
    'Types of Data Modeling',
    'Data Modeling BI',
    'Data Modeling benefits',
    'Laboratory 2',
    'Relationships types'
    
]

print('---'*100)
for msg in list_mgs:
    print('[USER]:', msg)
    response = rag1_chain.invoke(msg)
    print('[BOT]:', response)
    print('---'*100)
    user_input = input('What is the correct answer: ')
    user_input = response + '\nUser chose:' + user_input
    answer = rag2_chain.invoke(user_input)
    print('[BOT]:', answer)
    print('---'*100)
