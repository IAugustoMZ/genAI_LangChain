"""
This is the implementation of the QA answering system for private documents.
as the second project of the course
"""
# %%
import os
import time
import warnings
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ( ChatPromptTemplate, SystemMessagePromptTemplate, 
                               HumanMessagePromptTemplate )


# define path to the .env file
env_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'env', '.env'
)

# load the environment variables
load_dotenv(dotenv_path=env_path, override=True)

# ignore the warnings
warnings.filterwarnings("ignore")

# define data path
data_path = os.path.join(
    os.path.dirname(__file__),
    'data'
)

# %%
# define a function to load the documents
def load_document(file_path):
    """
    Load the document from the file path

    Parameters:
    ----------
    file_path: str
        The path to the document
    
    """

    # check if the document is a pdf
    if file_path.endswith('.pdf'):
        from langchain.document_loaders import PyPDFLoader

        # instantiate the document loader
        loader = PyPDFLoader(file_path=file_path)
    elif file_path.endswith('.docx'):
        from langchain.document_loaders import Docx2txtLoader

        # instantiate the document loader
        loader = Docx2txtLoader(file_path=file_path)
    else:
        raise ValueError("The document file type is not supported")

    # load the document
    document = loader.load()

    return document

# define other functions for searching
def load_from_wikipedia(query, lang='en', load_max_docs=2):

    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()

    return data

# define chunking function
def chunk_data(data, chunk_size=256, chunk_overlap=0):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(data)
   
    return chunks

# define a function to print the embedding cost
def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])

    print(f'Total tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')

# # function to a vector database
# def insert_or_fetch_embeddings(index_name, chunks):
#     import pinecone
#     from pinecone import PodSpec
#     from langchain_openai import OpenAIEmbeddings
#     from langchain_community.vectorstores import Pinecone

#     # create pinecone client
#     pc = pinecone.Pinecone()
#     embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

#     if index_name in pc.list_indexes().names():
        
#         print(f'Index {index_name} already exists. Loading embeddings...', end='')
#         vector_store = Pinecone.from_existing_index(index_name, embedding=embeddings)

#     else:
#         print(f'Creating index {index_name}...', end='')
#         pc.create_index(
#             name=index_name,
#             dimension=1536,
#             metric='cosine',
#             spec=PodSpec('gcp-starter')
#         )
#         vector_store = Pinecone.from_documents(
#             chunks,
#             embedding=embeddings,
#             index_name=index_name
#         )

#     return vector_store

# # function to delete a index
# def delete_index(index_name='all'):
#     import pinecone

#     # create pinecone client
#     pc = pinecone.Pinecone()

#     if index_name == 'all':
#         print('Deleting all indexes...')
#         for index in pc.list_indexes().names():
#             pc.delete_index(index)
#     else:
#         print(f'Deleting index {index_name}...', end='')
#         pc.delete_index(index_name)

# define a function to use Chroma as the vector store
def create_embedding_store(chunks, persist_directory='./data/chroma.db'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # define the embeddings
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # create vector score
    vector_store = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vector_store

# define function to load the vector store
def load_embedding_chroma(persist_directory='./data/chroma.db'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # define the embeddings
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vector_store = Chroma(persist_directory=persist_directory, embedding=embeddings)

    return vector_store

# function to ask a question
def ask_and_get_answer(vector_store, q):
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA
    
    # create the QA model
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    # create the retriever
    retriever = vector_store.as_retriever(
        serach_type='similarity',
        search_kwargs={'k': 10}
    )

    # create the memory
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

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

    # create the messages
    messages = [
        SystemMessagePromptTemplate.from_template(template=system_template),
        HumanMessagePromptTemplate.from_template(template=user_template)
    ]

    # create the chat prompt
    qa_prompt = ChatPromptTemplate.from_messages(messages=messages)

    # create the chain
    # chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type='stuff')
    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type='stuff',
        verbose=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )

    # get the answer
    # answer = chain.run(q)
    answer = crc.invoke({'question': q})

    return answer

# %%
# get all the documents
document_list = os.listdir(data_path)

text_list = []
for document in document_list:
    print('Loading document:', document)
    text_list.append(load_document(os.path.join(data_path, document)))

# %%
# test the chunking function
all_chunks = []
for text in text_list:
    all_chunks.append(chunk_data(text, chunk_size=256, chunk_overlap=0))

# %%
# print the embedding cost
for chunks in all_chunks:
    print_embedding_cost(chunks)

# %%
# # deleting all the indexes
# delete_index()

# # %%
# # insert the embeddings
# insert_or_fetch_embeddings(index_name='articles-test', chunks=all_chunks[0])

# # %%
# q = 'What is provided document about?'
# vector_store = insert_or_fetch_embeddings('articles-test', chunks=None)

#%%
# answer = ask_and_get_answer(vector_store, q)

# print(answer)

# create Chroma vector store
vector_store = create_embedding_store(all_chunks[1])

# %%
# creating the "app"
print('Write Quit or Exit to quit.')
i = 1
while True:
    q = input(f'Question {i}: ')
    i += 1
    if q.lower() in ['quit', 'exit']:
        print('Quitting...Bye bye!')
        time.sleep(2)
        break
    
    print('Question: ', q)
    answer = ask_and_get_answer(vector_store, q)
    print('Answer: ', {answer['answer']})
    print(f'\n{"-"*50}\n')