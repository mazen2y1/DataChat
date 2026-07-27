import os
from dotenv import load_dotenv
from LLMProvider import get_llm, get_text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
load_dotenv()

def create_database_agent(database_path):
    db = SQLDatabase.from_uri(
        f"sqlite:///{database_path}"
    )

    llm = get_llm()

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
        import traceback
        traceback.print_exc()
        raise