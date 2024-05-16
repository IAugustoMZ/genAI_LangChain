#%%
import os
from dotenv import load_dotenv
from datetime import datetime as dt
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache, SQLiteCache
from langchain_experimental.utilities import PythonREPL
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

# set path
PATH = os.path.join('../', 'env', '.env')

# find the dot env file
load_dotenv(PATH, override=True)

# %%
# intereacting with OpenAI AOU
llm = ChatOpenAI()
output = llm.invoke('What is the meaning of life?')
print(output.content)

# %%
# create messages
messages = [
    SystemMessage(content='You are a physicist and respond only in German.'),
    HumanMessage(content='Explain quantum mechanics in a single sentence')
]
output = llm.invoke(messages)
print(output.content)

# %%
# Caching LLMs - In-memory
set_llm_cache(InMemoryCache())
prompt = 'Explain quantum mechanics to a 7 year old child'

# first call - without cache
t0 = dt.now()
llm.invoke(prompt)
t1 = dt.now()
print('First call (without cache):', (t1-t0).total_seconds(), 's')

# %%
# second call
t0 = dt.now()
llm.invoke(prompt)
t1 = dt.now()
print('Second call (in-memory cache):', (t1-t0).total_seconds(), 's')

# %%
# Caching LLMs - SQLite
# set SQLite caching
set_llm_cache(SQLiteCache(database_path='.langchain.db'))
prompt = 'Explain dark matter to a French 6 six old child'

# first call - without cache
t0 = dt.now()
llm.invoke(prompt)
t1 = dt.now()
print('First call (without cache):', (t1-t0).total_seconds(), 's')

# %%
t0 = dt.now()
llm.invoke(prompt)
t1 = dt.now()
print('Second call (SQLite cache):', (t1-t0).total_seconds(), 's')

# %%
# Streaming LLMs
# no-streaming response
prompt = 'Write a rock song about the Moon and a Raven'
print(llm.invoke(prompt).content)

# %%
# streaming response
for chunk in llm.stream(prompt):
    print(chunk.content, end='', flush=True)

# %%
# Prompt Templates
# create a template
template = """You are an experienced virologist.
Write a few sentences about the following virus "{virus}" in "{language}"
"""
prompt_template = PromptTemplate.from_template(template)

prompt = prompt_template.format(virus='hiv', language='french')
prompt

# %%
response = llm.invoke(prompt)

print(response.content)
# %%
from langchain_core.messages import SystemMessage
# ChatPromptTemplates
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content='You respond only in the JSON format'),
    HumanMessagePromptTemplate.from_template(
        'Top {n} countries in {area} by population'
    )
])

messages = chat_template.format_messages(n='10', area='America')
print(messages)
# %%
output = llm.invoke(messages)
print(output.content)

# %%
# simple chains
# create a template
template = """You are an experienced virologist.
Write a few sentences about the following virus "{virus}" in "{language}"
"""
prompt_template = PromptTemplate.from_template(template)

# %%
# define a chain
chain = LLMChain(
    llm = llm,
    prompt = prompt_template,
    verbose=True
)

# invoke the chain
output = chain.invoke({'virus': 'COVID-19', 'language': 'Italian'})

print(output)
# %%
# another example with user input
template = 'What is the capital of {country}? List the top {n} places to visit in this city. Use bullet points'
prompt_template = PromptTemplate.from_template(template)

# define a chain
chain = LLMChain(
    llm = llm,
    prompt = prompt_template,
    verbose=True
)

# get inputs
country = input('Which country you want to visit?')
n = input('How many places do you want to visit?')

# invoke chain
output = chain.invoke({'country': country, 'n': n})
print(output['text'])

# %%
# Simple Sequential Chains

# define 1st chain
llm1 = ChatOpenAI(temperature=0.2)
prompt_template1 = PromptTemplate.from_template(
    template="""You are an experienced Python programmer and Chemical Engineer.
    Write a function that implements the concept of {concept}.
    You can assume any data you need.
"""
)
chain1 = LLMChain(llm=llm1, prompt=prompt_template1)

# define 2nd chain
llm2 = ChatOpenAI(temperature=1.5)
prompt_template2 = PromptTemplate.from_template(
    template="""You are a very experience Web developer.
    Write a JavaScript function that expects user input and executes the function {function}.
    The user should input the same arguments of the function.
"""
)
chain2 = LLMChain(llm=llm2, prompt=prompt_template2)

# concatenate the two chains
overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

# let's see the results
output = overall_chain.invoke('isothermal CSTR')

print(output['output'])

# %%
# LangChain Agents - Python REPL

# test python REPL
python_repl = PythonREPL()
python_repl.run('print([n for n in range(1, 100) if n % 13 == 0])')

#%%
# create the LLM
llm = ChatOpenAI(model='gpt-4-turbo-preview', temperature=0)

# create the agent
agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)

# invoke agent
agent_executor.invoke('Calculate the Pressure of an ideal gas of 1 mol at 9 cubic meters and 300 K')

#%%
# lets try another query
agent_executor.invoke("""
                      How many mols of an ideal gas do I need to put in 9 cubic meters and 300 K
                      to get a pressure of 30 MPa?
                      """)