import logging
from typing import Optional
from langchain.tools import BaseTool
from plan_and_execute_agent import Chatbot
from langchain.callbacks.manager import CallbackManagerForToolRun

# create a logger
logger = logging.getLogger(__name__)

class DefinePropertyTool(BaseTool):
    name: str = "define_property_tool"
    description: str = "Define Property based on user input"

    def __init__(self) -> None:
        """
        Define Property Tool
        """
        super().__init__()
        
    def _run(self, input_text: str, callback_manager = None) -> float:
        """
        Run the Define Property Tool

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
        chatbot = Chatbot()
        logger.info("Running Define Temperature Agent")
        return 'vapor_density'