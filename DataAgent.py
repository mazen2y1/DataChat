import os
from dotenv import load_dotenv
from LLMProvider import get_llm, get_text
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()

def create_data_agent(dataframe):
    llm = get_llm()

    agent = create_pandas_dataframe_agent(
        llm,
        dataframe,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={
            "handle parsing_errors": True
        }
    )
    return agent

def ask_data(agent, question):
    response = agent.invoke({
        "input": question
    })
    return response["output"]