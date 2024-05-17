#%%
import os
import warnings
from langchain import hub
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import Tool, AgentExecutor, initialize_agent, create_react_agent

# ignore warnings
warnings.filterwarnings('ignore')

# set path to dotenv file
PATH = os.path.join('../', 'env', '.env')

# find the dot env file
load_dotenv(PATH, override=True)

# %%
# set up the agent
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

# %%
# define a template
template = """Answer the following question as best as you can.
Question: {question}
"""
prompt_template = PromptTemplate.from_template(template)

# pull react prompt
prompt = hub.pull('hwchase17/react')

# %%
# checking the react prompting
print(prompt.template)

# %%
# creating the tools
# 1. Python REPL
python_repl = PythonREPLTool()
python_repl_tool = Tool(
    name='PythonREPL',
    func=python_repl.run,
    description='Useful when you need to use Python to answer questions. You should input Python code'
)

# 2. Wikipedia Query
api_wrapper = WikipediaAPIWrapper()
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)
wikipedia_tool = Tool(
    name='WikipediaQuery',
    func=wikipedia.run,
    description='Useful when you need to look up a topic, country, or person on Wikipedia. '
)

# 3. DuckDuckGo Search
duckduckgo = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name='DuckDuckGoSearch',
    func=duckduckgo.run,
    description='Useful when you need to search the web for information that another tool can\'t provide'
)

# collect all the tools in a list
tools = [python_repl_tool, wikipedia_tool, duckduckgo_tool]

# %%
# create the agent
agent = create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# %%
# set-up the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

# %%
# test a programming question
question = 'Implement a function that calculates the Arrhenius equation'
output = agent_executor.invoke({
    'input': prompt_template.format(question=question)
})

# %%
# print the input
print(output['input'])

# print the output
print(output['output'])

# %%
# test a web search
question = """List 20 applications of generative AI in chemical engineering according to recent papers.
Provide the links to the papers and order the papers by descending publication date.
Give me the list of applications and the links to the papers."""
output = agent_executor.invoke({
    'input': prompt_template.format(question=question)
})

# %%
# test a wikipedia search
question = 'What were the capitals of Brazil?'
output = agent_executor.invoke({
    'input': prompt_template.format(question=question)
})
# %%
# print the input
print(output['input'])

# print the output
print(output['output'])