import logging
from typing import Optional
from langchain.tools import BaseTool
from plan_and_execute_agent import Chatbot
from langchain.callbacks.manager import CallbackManagerForToolRun

# create a logger
logger = logging.getLogger(__name__)

class DefineCompoundTool(BaseTool):
    name: str = "define_compound_tool"
    description: str = "Define Compound based on user input"

    def __init__(self) -> None:
        """
        Define Compound Agent
        """
        super().__init__()
        
    def _run(self, input_text: str, callback_manager = None) -> str:
        """
        Run the Define Compound Agent

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
        logger.info("Running Define Compound Agent")
        user_prompt = chatbot.get_prompt()
        if 'ethanol' in user_prompt:
            return 'ethanol'
        else:
            return 'water'