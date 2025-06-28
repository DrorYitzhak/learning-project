from llm.llm_model import get_llm
from Ai_Agents.Agent_HVT.tools.agent_tools import TOOLS
from Ai_Agents.Agent_HVT.memory.memory_config import get_memory
from Ai_Agents.Agent_HVT.templates.main_prompt import full_combined_prompt
# from langchain.agents import create_tool_calling_agent
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate  # 🔹 זה החסר לך

# Load base components
llm = get_llm()
memory = get_memory()

def ask_agent(prompt: str, system_prompt: str = None) -> str:
    """Receives a textual question and returns the agent's response"""
    try:
        # Use provided system prompt if available
        prompt_template = full_combined_prompt if system_prompt is None else PromptTemplate.from_template(system_prompt)

        agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt_template)
        # agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            max_execution_time=None
        )

        response = agent_executor.invoke({"input": prompt})
        return response.get("output", "No response.")
    except Exception as e:
        return f"❌ Error: {str(e)}"
