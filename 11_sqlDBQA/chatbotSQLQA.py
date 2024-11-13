import os
import warnings
from dotenv import load_dotenv
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

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

# get the sql database path
db_path = '<database_path>'

# create the database
db = SQLDatabase.from_uri(f'sqlite:///{db_path}')


# create the chain
chain = create_sql_query_chain(chatbot, db)

# run the chain
response = chain.invoke({'question': 'Which projects I have worked so far? Bring all projects'})

# print the response
print(response)

# run the query
print(db.run(response.split('SQLQuery:')[1]))

# check the prompt
print(chain.get_prompts()[0].pretty_print())

print('---'*20)

# create agent that writes and executes the query
execute_query = QuerySQLDataBaseTool(db=db)

full_chain = chain | RunnableLambda(func= lambda x: x.split('SQLQuery:')[1])  | execute_query

response = full_chain.invoke({'question': 'How many hours have I worked in each project? Every project'})

print(response)

print('---'*20)

# agent that writes and executes the query and answers the question
answer_prompt = PromptTemplate.from_template(
    """
    Given a user question, answer it with the result of the query.

    Question: {question}
    SQLResult: {result}
    Answer:
    """
) 

human_chain = (
    RunnablePassthrough.assign(query=chain | RunnableLambda(func= lambda x: x.split('SQLQuery:')[1])).assign(
        result=itemgetter('query') | execute_query
    ) 
    | answer_prompt
    | chatbot
    | StrOutputParser()
)

# answer the question
response = human_chain.invoke({'question': 'How many hours have I worked in each project? Every project. Order by total hours'})

print(response)