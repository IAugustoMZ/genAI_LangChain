import os
import uuid
import warnings
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


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

# create a customized tool
class WaterVaporPressure(BaseTool):
    name = 'WaterVaporPressure'
    description = 'Calculate the water vapor pressure at a given temperature. The pressure is given in kPa.'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, **kwargs) -> float:
        """
        Calculate the water vapor pressure at a given temperature.

        Parameters
        ----------
        entrada : dict
            The input data.

        Returns
        -------
        float
            The water vapor pressure at a given temperature.
        """
        try:
            # get the temperature
            T = kwargs.get('temperature')
            if not T:
                T = kwargs.get('T')

            T += 273.15  # convert the temperature to Kelvin

            # constants of Riedel equation
            A = 73.649
            B = -7258.2
            C = -7.3037
            D = 4.1653 * 10 ** (-6)
            n = 2

            # calculate the water vapor pressure using the Riedel equation
            P = np.exp(A + B / T + C * np.log(T) + D * T ** n) / 1000
            return P
        except Exception as e:
            return f'Error: {str(e)}'
        
# create the agent tools
tools = [WaterVaporPressure()]

# bind the tools to the chatbot
# agent_executor = create_react_agent(chatbot, tools)

# create the memory
memory = MemorySaver()

# agent with memory
agent_executor_memory = create_react_agent(chatbot, tools, checkpointer=memory)

# test the agent
session_id = str(uuid.uuid4())
config = {'configurable': {'thread_id': session_id}}
msg_list = [
    'What is the water vapor pressure at 250 degrees Celsius?',
    'What is the water vapor pressure at 100 degrees Celsius?',
    'What is the water vapor pressure at 50 degrees Celsius?',
    'What is the water vapor pressure at 20 degrees Celsius?',
    'What is the water vapor pressure at 80 degrees Celsius?',
    'What is the water vapor pressure at 30 degrees Celsius?',
    'What was the first question and the response?',
]
for question in msg_list:
    print(f'\n[USER]: {question}')
    message = HumanMessage(content=question)

    # response = agent_executor.invoke({'messages': [message]})
    response = agent_executor_memory.invoke({'messages': [message]}, config=config)
    print('[CHATBOT]: ',response['messages'][-1].content)

