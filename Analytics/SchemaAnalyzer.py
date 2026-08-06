import json
from langchain_core.prompts import ChatPromptTemplate
from LLMProvider import get_text

class SchemaAnalyzer:
    def __init__(self, llm):
        self.llm = llm

    def _normalize_schema(self, schema: dict):
        expected_keys = [
            "revenue",
            "profit",
            "quantity",
            "customer",
            "product",
            "category",
            "subcategory",
            "date",
            "store",
            "location",
            "region",
            "state",
            "city",
            "segment",
            "discount",
        ]

        normalized = {}

        for key in expected_keys:
            value = schema.get(key)

            if isinstance(value, list):
                value = value[0] if value else None

            if isinstance(value, str):
                value = value.strip()
                if value.lower() in (
                    "",
                    "null",
                    "none",
                    "n/a",
                    "unknown",
                ):
                    value = None

            normalized[key] = value
        return normalized

    def analyze(self, df):
        columns = "\n".join(df.columns.astype(str).tolist())
        dtypes = df.dtypes.astype(str).to_string()
        sample = df.head(5).to_markdown(index=False)

        prompt = ChatPromptTemplate.from_template(
            """
You are a Senior Business Intelligence Data Analyst.

Identify which dataframe column matches each business field.

Rules:

- Return ONLY valid JSON.
- Every value MUST be either:
    - the exact column name
    - or null
- NEVER return arrays.
- NEVER return explanations.
- NEVER invent column names.
- Copy column names exactly as written.

Columns:
{columns}

Data Types:
{dtypes}

Sample:
{sample}

Return JSON:

{{
    "revenue": null,
    "profit": null,
    "quantity": null,
    "customer": null,
    "product": null,
    "category": null,
    "subcategory": null,
    "date": null,
    "store": null,
    "location": null,
    "region": null,
    "state": null,
    "city": null,
    "segment": null,
    "discount": null
}}
"""
        )
        chain = prompt | self.llm

        response = chain.invoke(
            {
                "columns": columns,
                "dtypes": dtypes,
                "sample": sample,
            }
        )

        content = get_text(response).strip()
        if content.startswith("```"):
            content = (
                content.replace("```json", "")
                .replace("```", "")
                .strip()
            )

        try:
            schema = json.loads(content)

        except Exception:
            start = content.find("{")
            end = content.rfind("}")

            if start == -1 or end == -1:
                raise ValueError(
                    f"Invalid JSON :\n\n{content}"
                )
            schema = json.loads(content[start:end + 1])
        return self._normalize_schema(schema)