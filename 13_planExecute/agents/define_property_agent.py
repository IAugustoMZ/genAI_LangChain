import functools
from langchain_openai import AzureChatOpenAI
from toolkit import ChemicalPropertiesToolkit
from plan_and_execute_agent import CreateAgent

class DefinePropertyAgent:
    def __init__(self, llm: AzureChatOpenAI, system_prompt: str) -> None:
        """
        Define Property Agent

        Parameters
        ----------
        llm : AzureChatOpenAI
            language model
        system_prompt : str
            system prompt
        """
        self.llm = llm
        self.system_prompt = system_prompt
        self.toolkit = ChemicalPropertiesToolkit()
        self.tools = self.toolkit.get_tools()

    def create_node(self) -> object:
        """
        Create a node

        Returns
        -------
        object
            node
        """
        agent = CreateAgent()
        agent_node = functools.partial(
            CreateAgent.run_agent_node,
            agent=agent.create_agent_executor(self.llm, self.tools, self.system_prompt),
            agent_name="DefinePropertyAgent"
        )
        return agent_node