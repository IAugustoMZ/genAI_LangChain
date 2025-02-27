from typing import List
from langchain_community.tools import BaseTool
from tools.define_property_tool import DefinePropertyTool
from tools.define_compound_tool import DefineCompoundTool
from langchain_community.agent_toolkits.base import BaseToolkit
from tools.define_temperature_tool import DefineTemperatureTool

class ChemicalPropertiesToolkit(BaseToolkit):
    tools: List[BaseTool] = []

    def __init__(self) -> None:
        """
        Chemical Properties Toolkit
        """
        super().__init__()
        self.tools = [
            DefinePropertyTool(),
            DefineCompoundTool(),
            DefineTemperatureTool()
        ]

    def get_tools(self) -> List[BaseTool]:
        """
        gets the tools

        Returns
        -------
        List[BaseTool]
            list of tools
        """
        return self.tools