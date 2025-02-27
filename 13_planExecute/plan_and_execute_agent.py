import os
import sys
import asyncio
import logging
import operator
import functools
import importlib.util
from anyio import to_thread
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

# create logger
logger = logging.getLogger(__name__)

# create the LLM that will drive the agent
llm = AzureChatOpenAI(
    api_key=os.environ.get('AZURE_OPEN_AI_API_SECRET_KEY'),
    api_version=os.environ.get('AZURE_OPEN_AI_API_VERSION'),
    azure_endpoint=os.environ.get('AZURE_OPEN_AI_GPT_3_5_ENDPOINT'),
    model=os.environ.get('AZURE_OPEN_AI_GPT_3_5_MODEL_NAME'),
    temperature=0
)

# define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# define the create agent class
class CreateAgent:
    def __init__(self) -> None:
        """
        creates the agent object
        """
        pass

    def create_agent_executor(self, llm: AzureChatOpenAI, tools: list, system_prompt: str) -> AgentExecutor:
        """
        creates a new agent executor node

        Parameters
        ----------
        llm : AzureChatOpenAI
            the language model that will be used to generate the prompts
        tools : list
            the list of tools that the agent will use
        system_prompt : str
            the system prompt that the agent will use

        Returns
        -------
        AgentExecutor
            the agent executor object
        """
        prompt = ChatPromptTemplate.from_messages([
            ('system', system_prompt),
            MessagesPlaceholder(variable_name='messages'),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])
        agent = create_openai_tools_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools
        )

    def run_agent_node(self, state: AgentState, agent: AgentExecutor, name: str) -> dict:
        """
        creates a new agent node

        Parameters
        ----------
        state : AgentState
            the agent state
        agent : AgentExecutor
            the agent executor object
        name : str
            the name of the agent

        Returns
        -------
        dict
            the agent node object
        """
        logger.info(f"Running agent {name}")
        result = agent.invoke(state)
        return {'messages': [HumanMessage(content=result['output'], name=name)]}

# define planner prompt class
class PlannerPrompt:

    @staticmethod
    def system_prompt() -> str:
        return """
        You are an AI agent expert in chemical engineering. You are the first step in a multiagent system whose purpose is to answer questions about physical and chemical properties of compounds.

        Your job is to come up with a plan to answer the user's question. Your plan should be a sequence of steps named by the agent that will execute them.

        At the end of your plan execution, the user should have the answer to their question. Here you can find an example of plan:

        Question: What is the vapor density of water at 120 degrees Celsius?
        1. DefineCompoundAgent
        2. run_temperature_id_tool
        4. run_property_id_tool
        3. run_property_calculation_tool
        4. FINISH
        """

# define empty tool
class EmptyTool(BaseTool):
    name: str = 'empty_tool'
    description: str = 'an empty tool'

    def _run(self, **kwargs) -> str:
        state = kwargs.get('state', {})
        return ''

# define Planner Node
class PlannerNode:
    def __init__(self, llm: AzureChatOpenAI) -> None:
        """
        creates the planner node object

        Parameters
        ----------
        llm : AzureChatOpenAI
            the language model that will be used to generate the prompts
        """
        self.llm = llm
        self.system_prompt = PlannerPrompt.system_prompt()
        self.tools = [EmptyTool()]

    def create_node(self) -> object:
        """
        creates a new planner node

        Returns
        -------
        object
            the planner node object
        """
        agent = CreateAgent()
        planner_node = functools.partial(
            CreateAgent.run_agent_node,
            agent=agent.create_agent_executor(self.llm, self.tools, self.system_prompt),
            name='PlannerAgent'
        )
        return planner_node

# define the supervisor node
class SupervisorNode:
    def __init__(self, llm: AzureChatOpenAI, members: list) -> None:
        """
        creates the supervisor node object

        Parameters
        ----------
        llm : AzureChatOpenAI
            the language model that will be used to generate the prompts
        members : list
            the list of members of the supervisor node
        """
        self.members = members
        self.llm = llm
        self.total_options = self.members + ['FINISH']
        self.system_prompt = """You are a supervisor AI agent expert in chemical engineering. You are the second step in a multiagent system whose purpose is to answer questions about physical and chemical properties of compounds.

        Your responsabilities include:
        1. Follow a plan created by the planner agent: {plan}
        2. Manage the conversation between the following workers: {members}

        Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status.
        When all workers have finished their tasks, respond with FINISH.
        """

    def functions_definitions(self) -> dict:
        """
        defines the functions for the supervisor node

        Returns
        -------
        dict
            the functions for the supervisor node
        """
        return {
            'name': 'route',
            'description': 'Selects the next worker to act',
            'parameters': {
                'title': 'routeSchema',
                'type': 'object',
                'properties': {
                    'next': {
                        'title': 'Next',
                        'anyOf': [{'enum': self.total_options}]
                    }
                },
                'required': ['next']
            }
        }

    def supervisor_chain(self) -> object:
        """
        creates a new supervisor chain

        Returns
        -------
        object
            the supervisor chain object
        """
        prompt = ChatPromptTemplate.from_messages([
            ('system', self.system_prompt),
            MessagesPlaceholder(variable_name='messages'),
            ('system', 'Given the conversation above, who should act next? Or should we FINISH? Select one of {options}')
        ])
        supervisor_chain = (
            prompt
            | self.llm.bind_functions(functions=[self.functions_definitions()], function_call='route')
            | JsonOutputFunctionsParser()
        )
        return supervisor_chain

# define the graph builder object
class GraphBuilder:
    def __init__(self, llm: AzureChatOpenAI) -> None:
        """
        creates the graph builder object

        Parameters
        ----------
        llm : AzureChatOpenAI
            the language model that will be used to generate the prompts
        """
        self.llm = llm
        self.workers_infos = [
            {'worker': 'DefineCompoundAgent', 'description': 'defines the chemical compound in the question', 'path': '13_planExecute/agents/define_compound_agent.py'},
            {'worker': 'DefineTemperatureAgent', 'description': 'defines the temperature in the question', 'path': '13_planExecute/agents/define_temperature_agent.py'},
            {'worker': 'DefinePropertyAgent', 'description': 'defines the property in the question', 'path': '13_planExecute/agents/define_property_agent.py'},
            {'worker': 'CalculatePropertyAgent', 'description': 'calculates the property in the question', 'path': '13_planExecute/agents/calculate_property_agent.py'}
        ]
        self.workers = [item['worker'] for item in self.workers_infos]
        self.supervisor_chain = SupervisorNode(self.llm, self.workers).supervisor_chain()
        self.planner = PlannerNode(self.llm).create_node()

    def build_graph(self) -> StateGraph:
        """
        builds the graph

        Returns
        -------
        StateGraph
            the state graph object
        """
        workflow = StateGraph(AgentState)

        # add the planner node
        workflow.add_node('planner', self.planner)
        workflow.add_node('supervisor', self.supervisor_chain)

        for worker_info in self.workers_infos:
            worker_name = worker_info['worker']
            system_prompts = worker_info['description']
            worker_path = worker_info['path']
            module_name = os.path.basename(worker_path).replace('.py', '')

            try:
                spec = importlib.util.spec_from_file_location(module_name, worker_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                worker_class = getattr(module, worker_name)
                worker_instance = worker_class(llm=self.llm, system_prompt=system_prompts)

                worker_node = worker_instance.create_node()
                workflow.add_node(worker_name, worker_node)
            except (ImportError, AttributeError, FileNotFoundError) as e:
                logger.error(f"Error loading worker {worker_name}: {e}")
                continue

        # connect all workers to supervisor
        for worker in self.workers:
            workflow.add_edge(worker, 'supervisor')

        # connect supervisor to planner
        workflow.add_edge('planner', 'supervisor')

        # create a conditional map
        conditional_map = {worker: worker for worker in self.workers}
        conditional_map['FINISH'] = END

        # set conditional edges
        workflow.add_conditional_edges('supervisor', lambda x: x['next'], conditional_map)

        # set entrypoint
        workflow.set_entry_point('planner')

        return workflow.compile()

# define chatbot class
class Chatbot:
    user_prompt = ''
    services = {}

    def __init__(self) -> None:
        """
        creates the chatbot object
        """
        pass

    @classmethod
    def set_prompt(cls, prompt: str) -> None:
        """
        sets the user prompt

        Parameters
        ----------
        prompt : str
            the user prompt
        """
        cls.user_prompt = prompt

    @classmethod
    def get_prompt(cls) -> str:
        """
        gets the user prompt

        Returns
        -------
        str
            the user prompt
        """
        return cls.user_prompt

    @classmethod
    def set_services(cls, services: dict) -> None:
        """
        sets the services

        Parameters
        ----------
        services : dict
            the services
        """
        cls.services = services

    @classmethod
    def get_services(cls) -> dict:
        """
        gets the services

        Returns
        -------
        dict
            the services
        """
        return cls.services

    def initialize_services(self) -> None:
        """
        initializes the services
        """
        self.llm = llm
        self.graph = GraphBuilder(self.llm).build_graph()
        self.services = {
            'llm': self.llm,
            'graph': self.graph
        }
        logger.info('All services initialized')
        self.set_services(self.services)

    def format_user_prompt(self, prompt: str) -> str:
        """
        formats the user prompt

        Parameters
        ----------
        prompt : str
            the user prompt

        Returns
        -------
        str
            the formatted user prompt
        """
        return {'messages': [HumanMessage(content=prompt)]}

    async def make_async(self, func: object, *args, **kwargs) -> object:
        """
        makes a function asynchronous

        Parameters
        ----------
        func : object
            the function
        args : list
            the arguments
        kwargs : dict
            the keyword arguments

        Returns
        -------
        object
            the result
        """
        return await to_thread.run_sync(func, *args, **kwargs)

    def run_graph_result(self, prompt: str) -> dict:
        """
        runs the graph result

        Parameters
        ----------
        prompt : str
            the prompt

        Returns
        -------
        dict
            the result
        """
        logger.info('Starting graph execution')
        config = {'recursion_limit': 50}
        result = self.graph.invoke(self.format_user_prompt(prompt), config)
        logger.info('Graph execution finished')
        return result['messages'][-1].content

    async def run_chatbot(self, prompt: str) -> dict:
        """
        runs the chatbot

        Parameters
        ----------
        prompt : str
            the prompt

        Returns
        -------
        dict
            the result
        """
        logger.info('Starting chatbot')
        self.set_prompt(prompt)
        graph_res = self.make_async(self.run_graph_result, prompt)
        graph_res = await asyncio.gather(*[graph_res])
        logger.info('Chatbot finished')

        logger.info('Returning chatbot result')
        chatbot_final_prompt = """You are an AI agent expert in chemical engineering. You are the final step in a multiagent system whose purpose is to answer questions about physical and chemical properties of compounds.
        
        Your job is to provide the final answer to the user's question. Here is the final answer:

        User question: {question}
        Answer: {answer}

        Return a response to the user with the answer. Use only the provided information to generate the response.
        """

        final_prompt = ChatPromptTemplate.from_messages([
            ('system', chatbot_final_prompt),
        ])
        final_response = self.llm.invoke(final_prompt, {'question': prompt, 'answer': graph_res})

        return final_response


if __name__ == '__main__':
    chatbot = Chatbot()
    chatbot.initialize_services()
    prompt = "What is the vapor density of water?"
    asyncio.run(chatbot.run_chatbot(prompt))