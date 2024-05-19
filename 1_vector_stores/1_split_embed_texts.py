#%%
import os
import warnings
import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# set path to dotenv file
PATH = os.path.join('../', 'env', '.env')

# loat the dotenv file
load_dotenv(PATH, override=True)

# ignore warnings
warnings.filterwarnings('ignore')

# %%
# create the text splitter text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len
)

# %%
# open the file
with open('./control_valves.txt', 'r') as file:
    text = file.read()

# %%
# create the chunks
chunks = text_splitter.create_documents([text])

# %%
print(chunks[:5])

# %%
# how many chunks do we have
print(len(chunks))

# %%
# lets check the cost of embedding
def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])

    print(f'Total tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')

# %%
print_embedding_cost(chunks)

# %%
embeddings = OpenAIEmbeddings()

# %%
# try the embeddings
vector = embeddings.embed_query('Hey, my name is Icaro')
vector

# %%
# create pinecone index
pc = pinecone.Pinecone()

# check if there are indexes, and delete them
for index in pc.list_indexes().names():
    pc.delete_index(index)

# %%
index_name = 'control-valves'
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=pinecone.PodSpec(
                        environment='gcp-starter'
                    ))
    
# %%
# upload the embeddings
vector_store = Pinecone.from_documents(
    chunks,
    embeddings,
    index_name=index_name
)

# %%
# if I want to load from an existing index
# vector_store = Pinecone.from_existing_index(index_name=index_name,
#                                             embeddings=embeddings)

# %%
# do some similarity search
query = 'What is a control valve?'

result = vector_store.similarity_search(query, k=10)
print(result)

# %%
for res in result:
    print(res.page_content)
    print('-' * 50)

# %%
# create the LLM model
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

# create the retriever
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 10}
)

# create a chain to answer questions
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever
)

# %%
answer = chain.run(query)

print(answer)

# %%
# try another question
query = 'What are the main parts of a control valve?'
answer = chain.run(query)

print(answer)
# %%
query = 'Why do I need a control valve?'
answer = chain.run(query)

print(answer)
# %%
