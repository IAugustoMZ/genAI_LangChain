#%%
import os
import random
import warnings
from pinecone import Pinecone
from dotenv import load_dotenv
from pinecone import ServerlessSpec


# set path to dotenv file
PATH = os.path.join('../', 'env', '.env')

# loat the dotenv file
load_dotenv(PATH, override=True)

# %%
# create a Pinecone vector store
pinecone = Pinecone()

# check connection
print(pinecone.list_indexes())

#%%
# create a new index
index_name = 'langchain'
if index_name not in pinecone.list_indexes():
    print('Creating index:', index_name)
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1',
        )
    )
    print('Index created:', index_name)
else:
    print('Index already exists:', index_name)

# %%
# analyze the index
print(pinecone.describe_index(index_name))

# %%
# get the index stats
index = pinecone.Index(index_name)
print(index.describe_index_stats())

# %%
# try to upsert vectors
vectors = [[random.random() for _ in range(1536)] for _ in range(10)]

# create a list of ids
ids = [f'vector_{i}' for i in range(10)]

# upsert vectors
index.upsert(vectors=zip(ids, vectors))

# %%
# update vectors
index.upsert(vectors=[('vector_0', [random.random() for _ in range(1536)])])

# %%
# fetch vectors
index.fetch(ids=['vector_0'])

# %%
# delete vectors
index.delete(ids=['vector_0', 'vector_1'])

# %%
index.describe_index_stats()

# %%
# perform a search
query_vector = [random.random() for _ in range(1536)]

index.query(
    vector=query_vector,
    top_k=3,
    include_values=False
)

# %%
# create a new namespace
vectors = [[random.random() for _ in range(1536)] for _ in range(3)]
ids = [f'vector_{i}' for i in range(3)]
index.upsert(vectors=zip(ids, vectors), namespace='namespace_1')

# %%
index.describe_index_stats()

# %%
# delete all vectors in a namespace
index.delete(namespace='namespace_1', delete_all=True)

# %%
index.describe_index_stats()

# %%
