import logging
import numpy as np
from typing import Optional
from langchain.tools import BaseTool
from plan_and_execute_agent import Chatbot
from langchain.callbacks.manager import CallbackManagerForToolRun

# create a logger
logger = logging.getLogger(__name__)

class CalculatePropertyTool(BaseTool):
    name: str = "DefineTemperatureAgent"
    description: str = "Define Temperature based on user input"

    def __init__(self) -> None:
        """
        Define Temperature Agent
        """
        super().__init__()
        self.chatbot = Chatbot()
        
    def _run(self, input_text: str, callback_manager = None) -> float:
        """
        Run the Define Temperature Agent

        Parameters
        ----------
        input_text : str
            input text
        callback_manager : Optional[CallbackManagerForToolRun], optional
            callback manager, by default None

        Returns
        -------
        str
            output text
        """
        logger.info("Running Define Temperature Agent")
        return np.random.uniform(0, 500) + 273.15