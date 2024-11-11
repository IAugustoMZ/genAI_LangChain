import os
import warnings
from dotenv import load_dotenv
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# define path to the .env file
env_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    '.env'
)
load_dotenv(dotenv_path=env_path, override=True)

# ignore the warnings
warnings.filterwarnings("ignore")

# create the chatbot
chatbot = ChatOpenAI(
    model='gpt-3.5-turbo-0125',
    temperature=0,
    max_retries=2,
)

class PhaseEquilibriumAnalysis(BaseModel):
    """
    Phase equilibrium analysis data, including the components, temperature, pressure, and type of diagram, necessary to build a phase equilibrium diagram.
    """
    component1: Optional[str] = Field(default=None, title='Component 1')
    component2: Optional[str] = Field(default=None, title='Component 2')
    temperature: Optional[float] = Field(default=None, title='Temperature')
    pressure: Optional[float] = Field(default=None, title='Pressure')
    temperature_unit: Optional[str] = Field(default=None, title='Temperature unit')
    pressure_unit: Optional[str] = Field(default=None, title='Pressure unit')
    type_diagram: Optional[str] = Field(default=None, title='Type of diagram and whether it is a Pxy or Txy diagram')

class AnalysisBatch(BaseModel):
    """
    Batch of phase equilibrium analysis data
    """
    analysis: List[PhaseEquilibriumAnalysis] = Field(title='List of phase equilibrium analysis data')


# prompt 
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are an expert extraction exert. Your job is to extract the necessary information from the user to build a phase equilibrium diagram. Only extraxt relevant information from the text. If you dont know the value of a parameter, leave it as None. The type of the diagram may not be explicit, but if the user provides only the temperature, it is Pxy. If only the pressure is provided, it is Txy.'),
        ('human', "{text}")
    ]
)

# create the chat chain
# chain = prompt | chatbot.with_structured_output(schema=PhaseEquilibriumAnalysis)
chain = prompt | chatbot.with_structured_output(schema=AnalysisBatch)

msgs = [
    'Txy diagram for ethanol and water at 1 atm',
    'Pxy diagram for ethanol and water at 100 C',
    'Phase equilibrium diagram for ethanol and water at 100 C',
    'Phase equilibrium diagram for ethanol and water at 1 atm and a phase equilibrium diagram for ethanol and water at 100 C',
    'Phase equilibrium diagram for ethanol and water at 1 atm, a phase equilibrium for ethanol and methanol at 100 C, and a phase equilibrium diagram for methanol and water at 200 C',
]

for msg in msgs:
    print(f'[USER]: {msg}')
    response = chain.invoke({'text': msg})
    print(f'[BOT]: {response}')
    print()