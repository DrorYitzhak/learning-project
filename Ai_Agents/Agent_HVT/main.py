from llm.llm_model import get_llm
from Ai_Agents.Agent_HVT.tools.agent_tools import TOOLS
from Ai_Agents.Agent_HVT.memory.memory_config import get_memory
from Ai_Agents.Agent_HVT.templates.main_prompt import full_combined_prompt
from langchain.agents import create_tool_calling_agent


from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor

def build_agent():
    llm = get_llm()
    memory = get_memory()

    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=full_combined_prompt)

    return AgentExecutor(agent=agent, tools=TOOLS, memory=memory, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    agent = build_agent()

    response = agent.invoke({
        "input": "כמה שורות נכשלו לפי עמודת Verdict_ATE בקובץ HVT_Production_Results_4432102105000898_2025-05-11--19-09-55.csv"
    })
    print("🔁 תשובת הסוכן:", response["output"])

    response = agent.invoke({
        "input": "מה היה ה־Result, Min_Limit_ATE, Max_Limit_ATE ובאיזה Test_Name עבור אותו DUT_SN?"
    })
    print("🔁 תשובת הסוכן:", response["output"])

    response = agent.invoke({
        "input": "תן לי סיכום של כל הנפילות"
    })
    print(response["output"])

