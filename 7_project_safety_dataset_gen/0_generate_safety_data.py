# %%
import os
import re
import warnings
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ( ChatPromptTemplate, SystemMessagePromptTemplate, 
                               HumanMessagePromptTemplate )

# define path to the .env file
env_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'env', '.env'
)

# load the environment variables
load_dotenv(dotenv_path=env_path, override=True)

# ignore the warnings
warnings.filterwarnings("ignore")

exmples_dict ={
    'low': [
        "Minor Acid Spill: Small HCl leak during transfer, promptly contained, no injuries reported.",
        "Minor Gas Leak: Valve malfunction released trace ammonia, ventilation activated, no exposure.",
        "Chemical Splash: Sodium hydroxide splash on operator's PPE, no skin contact, immediate cleanup.",
        "Equipment Damage: Pump seal failure, minor lubricant spill, area contained, no environmental impact.",
        "False Fire Alarm: Dust triggered fire alarm, no actual fire, system reset, normal operations resumed.",
        "Minor Chemical Spill: Small amount of acetone spilled, quickly cleaned up, no injuries.",
        "Electrical Fault: Minor short circuit in non-critical equipment, no injuries, repaired promptly.",
        "Tripped Breaker: Breaker tripped due to overload, power restored in 15 minutes, no injuries.",
        "Minor Slip Incident: Worker slipped on wet floor, no injuries, area marked and dried.",
        "Low-Level Noise Complaint: Temporary increase in noise from machinery, ear protection used, no injuries."
    ],
    'medium': [
        "Chemical Exposure: Operator inhaled vapors due to valve leak, required medical evaluation.",
        "Fire Incident: Small fire in storage area, extinguished quickly, minor equipment damage.",
        "Gas Release: Moderate chlorine leak, area evacuated, two employees treated for minor irritation.",
        "Spill Incident: 50 liters of sulfuric acid spilled, containment successful, minor environmental impact.",
        "Equipment Failure: Reactor malfunction caused pressure release, minor injuries, temporary shutdown.",
        "Electrical Fire: Short circuit caused small fire, extinguished, minor damage, no injuries.",
        "Structural Damage: Minor collapse of storage rack, no injuries, prompt repair required.",
        "Process Upset: Temperature spike in reactor, controlled shutdown, minor equipment damage.",
        "Ventilation Failure: HVAC system failure, temporary evacuation, no injuries, repairs completed.",
        "Minor Explosion: Small explosion in lab, minor burns to one worker, equipment damage."
    ],
    'high/critical': [
        "Explosion: Reactor explosion caused major injuries, significant structural damage.",
        "Toxic Release: Large ammonia leak, widespread exposure, multiple hospitalizations.",
        "Fire Incident: Major fire in production unit, extensive damage, plant shutdown.",
        "Chemical Spill: 1000 liters of hydrochloric acid spilled, severe environmental impact, emergency response.",
        "Structural Collapse: Tank collapse caused multiple casualties, extensive damage to facility.",
        "Major Gas Leak: High-pressure methane release, area evacuated, significant fire risk.",
        "Electrical Fire: Major electrical fire in control room, severe damage, prolonged outage.",
        "Process Failure: Catastrophic failure of distillation column, significant injuries, evacuation required.",
        "Environmental Disaster: Large-scale oil spill, extensive environmental damage, long-term cleanup needed.",
        "Critical Equipment Failure: Boiler explosion, critical injuries, major facility damage, emergency shutdown."
    ]
}

# %%
# create the llm model
llm1 = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
llm2 = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

# create the templates for the first llm model
first_llm_sys_template = """
You are a very expert safety engineer, with a lot of years of experience in several distinct chemical processes. Your experiences were built upon reading several incident and accident reports. Your job is to write a example of an accident / incident report, depending on the severity asked by the user. You may use the following examples:

EXAMPLES: {examples}

Your answer should be only the description of the report, which must not have more than 100 characters.
"""

first_llm_user_template = """
Based on the examples provided, write a {severity} incident report concerning an issue of {issue}.
"""

second_llm_sys_template = """
You are a very experienced safety engineer which is interested in developing a natural language processing classifier to classify incident reports. You have a lot of years of experience in several distinct chemical processes. Your experiences were built upon reading several incident and accident reports. Your job is to write variations of the same incident report, in order to generate a dataset for training the classifier.

Your answer should be only the description of the report, which must not have more than 100 characters per example. You should output only the list of examples, with any other text.
"""

second_llm_user_template = """
Based on the phrase the user provides, write 150 variations of the same incident report.

Phrase: {phrase}
"""

# create first llm messages
messages_1 = [
    SystemMessagePromptTemplate.from_template(
        template=first_llm_sys_template,
    ),
    HumanMessagePromptTemplate.from_template(
        template=first_llm_user_template,
    )
]

# create first llm messages
messages_2 = [
    SystemMessagePromptTemplate.from_template(
        template=second_llm_sys_template,
    ),
    HumanMessagePromptTemplate.from_template(
        template=second_llm_user_template,
    )
]

# create the LLMs prompts
first_llm_prompt = ChatPromptTemplate.from_messages(messages=messages_1)
second_llm_prompt = ChatPromptTemplate.from_messages(messages=messages_2)

# create the chains
first_chain = first_llm_prompt | llm1
second_chain = second_llm_prompt | llm2

# %%
# invoke the first chain
# first_output = first_chain.invoke({
#     'examples': exmples_dict['low'],
#     'severity': 'low',
#     'issue': 'minor acid spill'
# })

# print(first_output.content)

# # invoke the second chain
# second_output = second_chain.invoke({
#     'phrase': first_output.content
# })
# print(second_output.content)

# #%%
# # get list
# report_list = ast.literal_eval(second_output.content)
# print(report_list)

# %%
# define the function
def get_reports(row):
    """
    get the reports for the given severity and issue
    """
    severity = row['severities']
    issue = row['issues']

    # invoke the first chain
    first_output = first_chain.invoke({
        'examples': exmples_dict[severity],
        'severity': severity,
        'issue': issue
    })

    # invoke the second chain
    second_output = second_chain.invoke({
        'phrase': first_output.content
    })

    # create the lists
    report_list = second_output.content.split('\n')

    # get list
    report_list = [re.sub(r'^\d{1,3}\.\s*', '', k) for k in report_list]

    return report_list

# %%
issues = ['acid spill',  'gas leak', 'chemical splash', 'equipment damage', 'fire alarm', 'chemical spill', 'electrical fault', 'tripped breaker', 'slip incident', 'noise complaint', 'excessive heat', 'fire', 'explosion', 'toxic release', 'electrical fire', 'structural damage',
'confined space', 'fall', 'environmental spill', 'radioactive leak', 'cryogenic spill', 'biological spill', 'injury', 'inhaled gas', 'flooding', 'mechanical failure', 'pressure vessel', 'boiler', 'steam burning', 'transportation', 'load lifting', 'stuck valve', 'high pressure', 'high temperature', 'high temperature and high pressure', 'vaccuum', 'low temperature', 'reaction out of control', 'overfilling', 'instrument failure', 'short-circuit',
'thermal isolation failure', 'corrosion', 'eye contact', 'skin contact', 'ingestion', 'burning', 'structural collapse']

severities = ['low',  'medium', 'high/critical']

# combinations
combinations = [(a, b) for a in issues for b in severities]

# create the dataframe with the combinations
df = pd.DataFrame(combinations, columns=['issues', 'severities'])

#%%
# get the reports
df['reports'] = df.apply(get_reports, axis=1)

# %%
# save the dataset
df.to_csv('data/safety_dataset_3.csv', index=False)
# %%
