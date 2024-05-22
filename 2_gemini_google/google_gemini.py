#%%
import os
import warnings
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# set path to dotenv file
PATH = os.path.join('../', 'env', '.env')

# loat the dotenv file
load_dotenv(PATH, override=True)

# ignore warnings
warnings.filterwarnings('ignore')

# configure the client
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

# %%
for model in genai.list_models():
    print(model)

# %%
# let's integrate Gemini with Langchain
llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.9)

response = llm.invoke('What is the purpose of a backpressure turbine?')

#%%
print(response.content)

#%%
# use a prompt template
prompt = PromptTemplate.from_template('You are a technical content creator. Write a LinkedIn post about {topic}')

# create the chain
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)

# define topic
topic = 'applications of Generative AI in the industry'

# get the response
response = chain.run(topic=topic)

# print the response
print(response)
# %%
# let's try some prompt composition
llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.9, convert_system_message_to_human=True)

# create a prompt
output = llm.invoke([
    SystemMessage('Respond only YES or NO in Spanish'),
    HumanMessage('Is nuclear fusion a good source of energy?'),
])

print(output.content)

# %%
# let's try streaming
prompt = 'Write a scientific article introduction about Industry 4.0'
response = llm.invoke(prompt)
print(response.content)

# %%
for chunk in llm.stream(prompt):
    print(chunk.content)
    print('-' * 100)