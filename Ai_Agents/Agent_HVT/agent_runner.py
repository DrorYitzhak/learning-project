from Ai_Agents.Agent_HVT.llm.llm_model import get_llm
from Ai_Agents.Agent_HVT.tools.agent_tools import TOOLS
from Ai_Agents.Agent_HVT.memory.memory_config import get_memory
from Ai_Agents.Agent_HVT.templates.main_prompt import full_combined_prompt
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from matplotlib.figure import Figure
import matplotlib

# Load base components
llm = get_llm()
memory = get_memory()

def ask_agent(prompt: str, system_prompt: str = None):
    """
    Receives a textual question and returns the agent's response
    (either text, a pandas DataFrame, or a matplotlib Figure).
    """
    try:
        # Choose the correct prompt
        prompt_template = full_combined_prompt if system_prompt is None else PromptTemplate.from_template(system_prompt)

        # Create the agent with intermediate steps enabled
        agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            max_execution_time=None,
            return_intermediate_steps=True,
        )

        # Invoke the agent
        result = agent_executor.invoke({"input": prompt})

        # First: Check for Figure in intermediate steps
        for action_log, observation in result.get("intermediate_steps", []):
            # If returned dict with DataFrame or Figure
            if isinstance(observation, dict):
                # Return matplotlib Figure if present
                fig = observation.get("output")
                if isinstance(fig, matplotlib.figure.Figure):
                    return fig
                # Return DataFrame (for table preview)
                df = observation.get("data")
                if hasattr(df, 'head') and hasattr(df, 'columns'):
                    return df.head(5)  # show preview

        # Second: Check if output itself is Figure or DataFrame
        output = result.get("output", None)
        if isinstance(output, Figure):
            return output
        if hasattr(output, 'head') and hasattr(output, 'columns'):
            return output.head(5)

        # Sometimes agent returns a list of dicts as preview rows
        if isinstance(output, list) and all(isinstance(row, dict) for row in output):
            import pandas as pd
            df = pd.DataFrame(output)
            return df.head(5)

        # Otherwise, return the string/text output
        return output or "No response."

    except Exception as e:
        return f"❌ Error: {str(e)}"
