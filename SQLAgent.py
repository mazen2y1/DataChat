import sqlite3
from LLMProvider import get_llm, get_text

def create_database_agent(database_path):
    return database_path

def ask_database(database_path, question):
    llm = get_llm()
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )

        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            conn.close()
            return "The database contains no tables."

        schema = ""

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()

            schema += f"\nTable: {table}\n"

            for column in columns:
                schema += f"- {column[1]} ({column[2]})\n"

        prompt = f"""
You are an SQLite expert.

Database Schema:

{schema}

User Question:
{question}

Rules:
1. Return ONLY one SQLite SELECT query.
2. Do NOT use markdown.
3. Do NOT explain anything.
4. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE.
"""

        response = llm.invoke(prompt)

        sql = (
            get_text(response)
            .replace("```sql", "")
            .replace("```", "")
            .strip()
        )

        print("Generated SQL:")
        print(sql)

        forbidden = [
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "create",
            "truncate",
        ]

        if any(word in sql.lower() for word in forbidden):
            conn.close()
            return "Unsafe SQL query generated."

        cursor.execute(sql)
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return "No matching records were found."

        answer_prompt = f"""
The user asked:

{question}

SQL Result:

{rows}

Answer the user's question using ONLY the SQL result.

Keep the answer short and natural.
"""

        final_response = llm.invoke(answer_prompt)
        conn.close()
        return get_text(final_response)
    except Exception as e:
        conn.close()
        print(e)
        return f"Database Error: {e}"