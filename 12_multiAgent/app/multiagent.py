import os
import operator
import requests
import functools
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

# define llm
llm = ChatGroq(model='llama-3.3-70b-versatile')

# create a tool
# return_direct=False means that the tool will not return the result directly
# but will pass it to the next tool in the pipeline
@tool('process_search_tool', return_direct=False)
def process_search_tool(url: str) -> str:
    """
    Parse the web content with BeautifulSoup

    Parameters:
    -----------
    url: str
        The URL to parse

    Returns:
    --------
    str
        The parsed content
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

tools = [TavilySearchResults(max_results=2), process_search_tool]

# create an agent
def create_new_agent(llm: ChatGroq, tools: list, system_prompt: str) -> AgentExecutor:
    """
    Create a new agent

    Parameters
    ----------
    llm : ChatGroq
        llm model
    tools : list
        List of tools
    system_prompt : str
        System prompt

    Returns
    -------
    AgentExecutor
        The agent
    """
    # create the prompt 
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name='messages'),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])

    # create the agent
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # create the executor
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# define the agent node
def agent_node(state: dict, agent: AgentExecutor, name: str) -> dict:
    """
    Create an agent node

    Parameters
    ----------
    state : dict
        The state
    agent : AgentExecutor
        The agent
    name : str
        The name of the agent

    Returns
    -------
    dict
        The agent node
    """
    # invoke the agent
    result = agent.invoke(state)

    return {
        'messages': [HumanMessage(content=result['output'], name=name)]
    }

# define the team of nodes
content_writers_team = ['online_researcher', 'blog_manager', 'social_media_manager']

# define the system prompt
system_prompt = """As a content marketing manager, your role is to oversee the insight between these workers: {content_writers_team}. Based on user's request, determine which worker should take the next action. Each worker is responsible for executing a specific task and reporting back their findings and progress. Once all tasks are completed, indicate FINISH
"""
options = content_writers_team + ['FINISH']

# define the route function
function_def = {
    'name': 'route',
    'description': 'Select the next role.',
    'parameters': {
        'title': 'routeSchema',
        'type': 'object',
        'properties': {'next': {'title': 'Next', 'anyOf': [{'enum': options}]}},
        'required': ['next']
    }
}

# define the content marketing manager prompt
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    MessagesPlaceholder(variable_name='messages'),
    ('system', 'Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}')
]).partial(options=str(options), content_writers_team=', '.join(content_writers_team))

# create the marketing manager chain
content_marketing_manager_chain = prompt | llm.bind(functions=[function_def], function_call='required') | JsonOutputFunctionsParser()

# create the agents
online_researcher_agent = create_new_agent(
    llm,
    tools,
    """Your primary role is to function as an intelligent online research assistant, adept at scouring the internet for the latest and most relevant trending stories about chemical and / or process engineering. You possess the capability to access a wide range of online news sources, blogs, and social media platforms to gather real time information."""
)
online_researcher_node = functools.partial(agent_node, agent=online_researcher_agent, name='online_researcher')

blog_manager_agent = create_new_agent(
    llm,
    tools,
    """"You are a Blog Manager. The role of a blog manager emcompasses several critical responsabilities aimed at transforming initial drafts into polished, SEO-optimized blog articles that engage and grow in audience. Starting with drafts provided by online researchers, the Blog Manager must throughly understand the content, ensuring it aligns with blog's tone, target audience, and thematic goals. Key-responsabilities include:
    
    1. Content Enhancement: Elevate the draft's quality by improving clarity, flow and engagement. This involves refininf the narrative, adding compelling headers, and ensuring the article is reader-friendly and informative.
    2. SEO Optimization: Implement best practices for search engine optimization. This includes keyword research and integration, optimizing meta descriptions, and ensuring URL structures and heading tags enhance visibility in search engine results.
    3. Compliance and Best Practices: Ensure the content adheres to legal and ethical standards, including copyright laws and truth in advertising. The Blog Manager must also keep up with evolving SEO strategies and blogging trends to maintain and enhance content effectiveness.
    4. Editorial Oversight: Work closely with writers and contributors to maintain a consistent voice and quality across all blog posts. This may also involve managing a content calendar, scheduling posts for optimal engagement, and coordinating with marketing teams to support promotional activities.
    5. Analytics and Feedback Integration: Regularly review performance metrics to understand audience engagement and preferences. Use this data to refine future content and optimize overall blog strategy.

    In summary, the Blog Manager plays a pivotal role in bridging initial research and the final publication by enhancing content quality, ensuring SEO compatibility, and aligning with the strategic objectives of the blog. This position requires a blend of creative, technical, and analytical skills to sucessfully manage and grow the blog's presence online.
    """
)

blog_manager_node = functools.partial(agent_node, agent=blog_manager_agent, name='blog_manager')

social_media_manager_agent = create_new_agent(
    llm,
    tools,
    """
    You are a Social Media Manager. The role of a Social Media Manager, particularly for managing LinkedIn content, involves transforming research drafts into concise engaging Linkedin posts that ressonate with the audience and adhere to the platform best practices. Upon receiving a draft from an online researcher, the Social Media Manager is tasked with severak critical functions:

    1. Conten Condensation: Distill the core messages of the draft into a concise, engaging LinkedIn post. This involves capturing the essence of the article in a few sentences, highlighting key takeaways, and creating a compelling call-to-action. This requires a sharp focus on getting the message across in a clear and engaging manner.
    2. Engagement Optimization: Craft posts that are tailored to LinkedIn's to maximize engagement. This includes the strategic use of compelling language, relevant hashtags, and timely topics that resonate with the target audience.
    3. Compliance and Best Practices: Ensure that all posts adhere to LinkedIn's community guidelines and best practices. This includes appropriate use of mentions, hashtags, and links. Also, adhere to ethical standards, avoiding misinformation and respecting copyright laws.

    In summary, the Social Media Manager's role is crucial in leveraging Linkedin to disseminate information effectively, engage with followers, and build the brand's presence online. This position combines creative communication skills with strategic planning and analysis to optimize social media impact.
    """
)

social_media_manager_node = functools.partial(agent_node, agent=social_media_manager_agent, name='social_media_manager')

# create the graph state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# create the graph workflow
workflow = StateGraph(AgentState)
workflow.add_node('content_marketing_manager', content_marketing_manager_chain)
workflow.add_node('online_researcher', online_researcher_node)
workflow.add_node('blog_manager', blog_manager_node)
workflow.add_node('social_media_manager', social_media_manager_node)

# all nodes are connected to the content marketing manager
for member in content_writers_team:
    workflow.add_edge(start_key=member, end_key='content_marketing_manager')

conditional_map = {k: k for k in content_writers_team}
conditional_map['FINISH'] = END
workflow.add_conditional_edges('content_marketing_manager', lambda x: x['next'], conditional_map)

# set entry point
workflow.set_entry_point('content_marketing_manager')

# compile the graph
multiagent = workflow.compile()

# define the main function to run the multiagent
def main():
    theme = input('Tell me what you want me to talk about: ')
    template = f"""Write me a report about: {theme}. After the research on {theme}, pass the findings to the blog manager to generate the final blog post. Onde done, pass it to the social media manager to create a LinkedIn post. """
    state = {'messages': [HumanMessage(content=template)]}

    for s in multiagent.stream(state, {'recursion_limit': 50}):
        if not '__end__' in s:
            print(s, end='\n\n--------------------------------------\n\n')

if __name__ == '__main__':
    main()