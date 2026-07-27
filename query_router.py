import os
import json
import logging
from dotenv import load_dotenv
from LLMProvider import get_llm
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = get_llm()

def route_question(question, available_sources):
    available_sources = [s.upper() for s in available_sources]
    llm = get_llm()
    prompt = f"""
You are a query router and question decomposer
for a Chat With Data application.

Available sources:
{", ".join(available_sources)}

Source definitions:

TABULAR:
Uploaded CSV or Excel files.

DOCUMENT:
Uploaded PDF or TXT documents.

DATABASE:
Uploaded SQL database.

Your job:

1. Determine which sources are required.
2. Create a separate sub-question for each required source.
3. Each sub-question MUST contain ONLY the part that
   can be answered by that specific source.

prefix =
You are an SQL assistant.

You MUST query the database before answering.
Never guess any value.
If no row is found, say that it was not found.
Always base your answer only on the SQL query result.

Example:

User Question:
Using the university document and database,
tell me the university research budget and
the total inventory value.

Output:
{{
    "DOCUMENT": "What is the university's annual research budget?",
    "DATABASE": "What is the total inventory value?"
}}

Another example:

User Question:
Using all three sources, give me the maximum
Data_value from the CSV, the research budget
from the document, and total inventory value
from the database.

Output:
{{
    "TABULAR": "What is the maximum Data_value in the CSV?",
    "DOCUMENT": "What is the university's annual research budget?",
    "DATABASE": "What is the total inventory value?"
}}

Rules:

- Only use sources listed in Available sources.
- Do not send document questions to DATABASE.
- Do not send database questions to DOCUMENT.
- Do not send CSV questions to DOCUMENT or DATABASE.
- Return ONLY valid JSON.
- Do not use markdown.
- Do not explain your answer.

User Question:
{question}

Output:
"""
    response = llm.invoke(prompt)
    raw_response = response.content.strip()
    raw_response = raw_response.replace(
        "```json",
        ""
    ).replace(
        "```",
        ""
    ).strip()
    logger.info(f"Router raw LLM output: {raw_response!r}")
    try:
        routes = json.loads(
            raw_response
        )

    except json.JSONDecodeError:
        logger.error(
            f"Router failed"
            f"Question: {question!r} | Raw output: {raw_response!r}"
        )
        return {}

    if not isinstance(routes, dict):
        logger.error(f"no dict JSON: {routes!r}")
        return {}

    valid_routes = {}
    for source, sub_question in routes.items():
        source = str(source).upper()
        if source in available_sources and isinstance(sub_question, str) and sub_question.strip():
            valid_routes[source] = sub_question.strip()
        else:
            logger.warning(
                f"invalid route: {source} -> {sub_question!r}"
            )

    if not valid_routes:
        logger.warning(
            f"No routes found , Question: {question!r} | "
            f"Available sources: {available_sources} | Parsed routes: {routes!r}"
        )
    return valid_routes

def combine_answers(
    question,
    answers
):
    llm = get_llm()
    results = ""
    for source, answer in answers.items():
        results += f"""
SOURCE: {source}

RESULT:
{answer}

"""


    prompt = f"""
You are the final answer generator for a
multi-source Chat With Data application.

Original User Question:

{question}


Results retrieved from the data sources:

{results}


Create ONE final answer that directly answers
the original question.

Rules:

- Use ONLY information provided in the results.
- Combine information from all sources.
- Do not invent information.
- Preserve important numbers exactly.
- If a calculation can be performed using numbers
  provided in the results, perform the calculation.
- Clearly explain comparisons when requested.
- Do not mention internal agents or routing.

Final Answer:
"""
    response = llm.invoke(prompt)
    return response.content