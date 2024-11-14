import os
import time
import random
import warnings
from enum import Enum
from typing import List
# from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# define path to the .env file
# env_path = os.path.join(
#     os.path.dirname(os.path.dirname(__file__)),
#     '.env'
# )
# load_dotenv(dotenv_path=env_path, override=True)

# ignore the warnings
warnings.filterwarnings("ignore")

# create the chatbot
chatbot = ChatOpenAI(
    model='gpt-3.5-turbo-0125',
    temperature=0,
    max_retries=2,
)

# create the prompt
tagging_prompt = ChatPromptTemplate.from_template(
    """You are a very experienced safety engineer which is interested in developing a natural language processing classifier to classify incident reports. You have a lot of years of experience in several distinct chemical processes. Your experiences were built upon reading several incident and accident reports. 
    
    Extract the desored information from the following passage. Only extract the properties mentioned in the 'IncidentReportClassification'

    Passage: {text} 
    """
)

class Severity(str, Enum):
    """
    Severity of the incident report.
    """
    low = "low"
    medium = "medium"
    high_critical = "high/critical"

class Issue(str, Enum):
    """
    Issue of the incident report.
    """
    acid_spill = "acid spill"
    gas_leak = "gas leak"
    chemical_splash = "chemical splash"
    equipment_damage = "equipment damage"
    fire_alarm = "fire alarm"
    chemical_spill = "chemical spill"
    electrical_fault = "electrical fault"
    tripped_breaker = "tripped breaker"
    slip_incident = "slip incident"
    noise_complaint = "noise complaint"
    excessive_heat = "excessive heat"
    fire = "fire"
    explosion = "explosion"
    toxic_release = "toxic release"
    electrical_fire = "electrical fire"
    structural_damage = "structural damage"
    confined_space = "confined space"
    fall = "fall"
    environmental_spill = "environmental spill"
    radioactive_leak = "radioactive leak"
    cryogenic_spill = "cryogenic spill"
    biological_spill = "biological spill"
    injury = "injury"
    inhaled_gas = "inhaled gas"
    flooding = "flooding"
    mechanical_failure = "mechanical failure"
    pressure_vessel = "pressure vessel"
    boiler = "boiler"
    steam_burning = "steam burning"
    transportation = "transportation"
    load_lifting = "load lifting"
    stuck_valve = "stuck valve"
    high_pressure = "high pressure"
    high_temperature = "high temperature"
    high_temperature_and_high_pressure = "high temperature and high pressure"
    vaccuum = "vaccuum"
    low_temperature = "low temperature"
    reaction_out_of_control = "reaction out of control"
    overfilling = "overfilling"
    instrument_failure = "instrument failure"
    short_circuit = "short-circuit"
    thermal_isolation_failure = "thermal isolation failure"
    corrosion = "corrosion"
    eye_contact = "eye contact"
    skin_contact = "skin contact"
    ingestion = "ingestion"
    burning = "burning"
    structural_collapse = "structural collapse"

# create the incident report classification schema
class IncidentReportClassification(BaseModel):
    """
    Incident report classification data, including the severity, issue, and description of the incident report.
    """
    severity: List[Severity] = Field(title="Severity of the incident report") 
    issue: List[Issue] = Field(title="Issue of the incident report.")

# create the chat chain
classificationChain = tagging_prompt | chatbot.with_structured_output(schema=IncidentReportClassification)

# define list of reports
listReports = [
    "Minor Acid Spill: Small HCl leak during transfer, promptly contained, no injuries reported.",
    "Minor Gas Leak: Valve malfunction released trace ammonia, ventilation activated, no exposure.",
    "Chemical Splash: Sodium hydroxide splash on operator's PPE, no skin contact, immediate cleanup.",
    "Equipment Damage: Pump seal failure, minor lubricant spill, area contained, no environmental impact.",
    "False Fire Alarm: Dust triggered fire alarm, no actual fire, system reset, normal operations resumed.",
    "Minor Chemical Spill: Small amount of acetone spilled, quickly cleaned up, no injuries.",
    "Electrical Fault: Minor short circuit in non-critical equipment, no injuries, repaired promptly.",
    "Tripped Breaker: Breaker tripped due to overload, power restored in 15 minutes, no injuries.",
    "Minor Slip Incident: Worker slipped on wet floor, no injuries, area marked and dried.",
    "Low-Level Noise Complaint: Temporary increase in noise from machinery, ear protection used, no injuries.",
    "Chemical Exposure: Operator inhaled vapors due to valve leak, required medical evaluation.",
    "Fire Incident: Small fire in storage area, extinguished quickly, minor equipment damage.",
    "Gas Release: Moderate chlorine leak, area evacuated, two employees treated for minor irritation.",
    "Spill Incident: 50 liters of sulfuric acid spilled, containment successful, minor environmental impact.",
    "Equipment Failure: Reactor malfunction caused pressure release, minor injuries, temporary shutdown.",
    "Electrical Fire: Short circuit caused small fire, extinguished, minor damage, no injuries.",
    "Structural Damage: Minor collapse of storage rack, no injuries, prompt repair required.",
    "Process Upset: Temperature spike in reactor, controlled shutdown, minor equipment damage.",
    "Ventilation Failure: HVAC system failure, temporary evacuation, no injuries, repairs completed.",
    "Minor Explosion: Small explosion in lab, minor burns to one worker, equipment damage.",
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

# shuffle the list of reports
random.shuffle(listReports)

# iterate over the list of reports
for report in listReports:
    print(f'[USER]: {report}')
    try:

        response = classificationChain.invoke({'text': report})

        # format the response
        severity = str(response.severity[0]).split('.')[1]

        # issues can be a list, so split by comma
        issues = ', '.join([str(issue).split('.')[1] for issue in response.issue])

        print(f'[SYSTEM]: Classification: {severity}, Issue: {issues}')
    except:
        print('[SYSTEM]: Unable to classify the incident report.')
    print('---'*200)
    time.sleep(2)
 