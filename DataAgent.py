import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()

def create_data_agent(dataframe):
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

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