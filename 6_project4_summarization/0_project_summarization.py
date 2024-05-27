# %%
import os
import warnings
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.agents import initialize_agent, Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# define path to the .env file
env_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'env', '.env'
)

# load the environment variables
load_dotenv(dotenv_path=env_path, override=True)

# ignore the warnings
warnings.filterwarnings("ignore")

# %%
# work with a basic text summarization
text = """
The first industrial revolution (1750 - 1860) introduced the steam engine, a technological breakthrough that facilitated the shift from artisanal to mechanical manufacturing. This revolution profoundly altered production methods, leading to significant economic and societal changes. The second industrial revolution (1850 - 1960) was marked by the advent of the internal combustion engine and electric energy, which enabled mass production and the development of production lines. This period saw innovations originating in Europe and the USA that transformed industries on a large scale. The third industrial revolution (1940 - 2000) brought about the use of nuclear energy, oil, and gas, along with the modernization of transportation and automation of processes. The creation of the first industrial robots for the automobile industry and the implementation of programmable logic controllers were pivotal during this era \cite{XU2018}.

The fourth industrial revolution, or Industry 4.0, represents a significant departure from previous industrial revolutions. It is characterized by the integration of advanced computation, communication technologies (such as the Internet), and data processing. These technologies enable a deep transformation in production systems, focusing on efficiency, scalable quality, and cost reduction through the collection and analysis of vast amounts of data \cite{JAN2023}. This era is defined by cyber-physical systems, where machines and the physical world interact seamlessly through instruments, sensors, and user interactions \cite{ZHOU2015}.

As technological advancements continue to unfold, the question arises whether we are entering a Fifth Industrial Revolution (I5.0). Historically, industrial revolutions have been marked by significant technological breakthroughs that transform production, economies, and societies. While Industry 4.0 is still evolving and its full potential has not yet been realized, the term Industry 5.0 has emerged in various discussions, hinting at another paradigm shift \cite{COELHO2023}. However, unlike previous revolutions, I5.0 lacks a clear and universally agreed-upon disruptive technology or framework, raising questions about its validity as a distinct revolution. Concepts associated with I5.0, such as human-centric technology integration and collaboration between humans and machines, seem to build upon rather than replace the advancements of I4.0. Therefore, it is more plausible to view these developments as an extension of the ongoing fourth industrial revolution rather than a separate fifth revolution \cite{COELHO2023}.

Nonetheless, industrial revolutions are not solely about the adoption of new technologies but also about how people work and interact with these technologies. Generative AI, for example, represents a transformative leap in artificial intelligence, profoundly altering the business landscape. This technology enables the creation of high-quality, customized content, such as personalized marketing emails, significantly reducing production costs and enhancing revenue. The broad application of generative AI across various business functions holds the potential to deliver substantial economic benefits, estimated between $2.6 trillion to $4.4 trillion annually. Furthermore, generative AI enhances labor productivity by automating detailed work activities and enabling workers, especially less-experienced ones, to achieve higher efficiency and performance. This capability boosts individual productivity and improves overall business outcomes, such as customer satisfaction and employee retention. While traditional AI technologies continue to add significant value, generative AI opens new frontiers in creativity and innovation, positioning it as a pivotal force in shaping the future of work and business \cite{BRYNJOLFSSON2023, CHUI2023}.

Whether we label the ongoing advancements as part of the fourth or fifth industrial revolution, it is crucial to consider the profound impacts of these technologies. Their potential to transform industrial competitiveness and societal structures underscores the importance of understanding and adapting to these changes, regardless of nomenclature.
"""

# creating messages
messages = [
    SystemMessage(content='You are an expert copyriter with experience in summarizing texts.'),
    HumanMessage(content=f'Please provide a short and concise summary of the following text: {text}'),
]

# create the llm
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

# %%
# get the number of tokens
llm.get_num_tokens(text)

#%%
# run the llm
summary = llm(messages)

print(summary.content)

# %%
# summarizing using prompt templates
template = """
Write a concise and short summary of the following text:
TEXT: {text}
Translate the summaly to {language}
"""

prompt = PromptTemplate(
    input_variables=['text', 'language'],
    template=template
)

# %%
# get the number of tokens
llm.get_num_tokens(prompt.format(text=text, language='Portuguese')) 

# %%
chain = LLMChain(llm=llm, prompt=prompt)

# run the chain
summary = chain.invoke({'text': text, 'language': 'Portuguese'})

print(summary)

# %%
# Stuffing - stuff the text into the
docs = [Document(page_content=text)]

# create the prompt template
prompt_template = "Write a concise and short summary of the following text: {text}"

prompt = PromptTemplate(
    input_variables=['text'],
    template=prompt_template
)

# create the chain
chain = load_summarize_chain(llm=llm, prompt=prompt, chain_type='stuff')

# run the chain
summary = chain.run(docs)
print(summary)

# %%
# create the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = text_splitter.create_documents([text])

# create the chain
chain = load_summarize_chain(llm=llm, chain_type='map_reduce')

# run the chain
summary = chain.run(chunks)
print(summary)

# %%
# using map reduce with custom prompts
map_prompt = """Write a short and concise summary of the following:
Text: {text}
CONCISE SUMMARY:
"""

map_prompt_template = PromptTemplate(
    input_variables=['text'],
    template=map_prompt
)

# combine prompt 
combine_prompt = """Write a concise summary of the following text that covers the key points.
Add a title to the summary.
Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED by bullet points if possible AND end the summary with a CONCLUSION phrase.
Text: {text}
"""

combine_prompt_template = PromptTemplate(
    input_variables=['text'],
    template=combine_prompt
)

# create the chain
summary_chain = load_summarize_chain(
    llm=llm,
    map_prompt=map_prompt_template,
    combine_prompt=combine_prompt_template,
    chain_type='map_reduce'
)

# run the chain
summary = summary_chain.run(chunks)

print(summary)

# %%
# loader for the pdf
loader = PyPDFLoader(file_path="C:\\Users\\izelioli\\OneDrive - Deloitte (O365D)\\Documents\\Pessoais\\Courses\\LearnLangChain\\genAI_LangChain\\4_project2_rag_private_documents\\data\\ApplicationsI40Manufacturing_Zheng2020.pdf")
data = loader.load()

# create the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)

# split the data into chunks
chunks = text_splitter.split_documents(data)

# create the chain
summary_chain = load_summarize_chain(
    llm=llm,
    chain_type='refine'
)

# run the chain
# summary = summary_chain.run(chunks)

# print(summary)

# %%
# refine with custom prompts
prompt_template = """
Write a concise summary of the following text extracting the key points:
Text: {text}
CONCISE SUMMARY:
"""
initial_prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

# refine prompt
refine_prompt = """
Your job is to produce a final summary.
I have provided a summary up to a certain point: {existing_summary}
Please refine the summary with some more context below
--------------
{text}
--------------
Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED by bullet points if possible AND end the summary with a CONCLUSION phrase.
"""
refine_prompt_template = PromptTemplate(template=refine_prompt, input_variables=['existing_summary', 'text'])

# create the chain
summary_chain = load_summarize_chain(
    llm=llm,
    question_prompt=initial_prompt,
    refine_prompt=refine_prompt_template,
    chain_type='refine',
    return_intermediate_steps=False
)

# run the chain
# summary = summary_chain.run(chunks)

# print(summary)

# %%
# summarization using LangChain Agents
# create the wikipedia agent
wikipedia_agent = WikipediaAPIWrapper()

# creating the tools
tools = [
    Tool(name='wikipedia', func=wikipedia_agent.run, description='Get information from Wikipedia')
]

# initialize the agent
agent = initialize_agent(tools=tools, llm=llm, agent='zero-shot-react-description', verbose=True)

# run the agent
output = agent.run('What is the main difference of Industry 4.0 and Industry 5.0?')

print(output)