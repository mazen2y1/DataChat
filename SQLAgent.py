import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
load_dotenv()

def create_database_agent(database_path):
    db = SQLDatabase.from_uri(
        f"sqlite:///{database_path}"
    )

    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

    agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    agent_type="tool-calling"
    )
    return agent

def ask_database(agent, question):
    try:
        response = agent.invoke({
            "input": question
        })
        output = response.get("output")
        if not output:
            return (
                "I could not find the requested"
                "information in the database"
            )
        return output

    except Exception as e:
        print(f"SQL Agent Error: {e}")
        return (
            "I could not find the requested"
            "information in the database"
        )