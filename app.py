import streamlit as st
import pandas as pd
import tempfile
from DataLoader import load_file
from DataAgent import create_data_agent, ask_data
from RAG import build_RAG, ask_RAG
from SQLAgent import create_database_agent, ask_database
import query_router

st.set_page_config(
    page_title="Chat With Data",
    page_icon="📊"
)
st.title("📊 Chat With Data")
st.write(
    "Upload your data and ask questions "
    "across multiple data sources."
)

uploaded_files = st.file_uploader(
    "Upload your files",
    type=[
        "csv",
        "xlsx",
        "xls",
        "pdf",
        "txt",
        "db",
        "sqlite",
        "sqlite3"
    ],
    accept_multiple_files=True
)

if uploaded_files:
    dataframes = []
    texts = []
    databases = []
    for file in uploaded_files:
        extension = file.name.split(".")[-1].lower()
        if extension in [
            "db",
            "sqlite",
            "sqlite3"
        ]:
            databases.append(file)
            with st.expander(
                f"{file.name}"
            ):
                st.write(
                    "SQLite Database ready"
                )
            continue
        try:
            data = load_file(file)
            if isinstance(
                data,
                pd.DataFrame
            ):
                dataframes.append(data)
                with st.expander(
                    f"{file.name}"
                ):
                    st.dataframe(
                        data.head()
                    )
            elif isinstance(
                data,
                str
            ):
                texts.append(data)
                with st.expander(
                    f"{file.name}"
                ):
                    st.text(
                        data[:1000]
                    )

        except Exception as e:
            st.error(
                f"Error loading "
                f"{file.name}: {e}"
            )
            st.exception(e)

    available_sources = []

    if dataframes:
        available_sources.append(
            "TABULAR"
        )

    if texts:
        available_sources.append(
            "DOCUMENT"
        )

    if databases:
        available_sources.append(
            "DATABASE"
        )

    data_agent = None
    vectorstore = None
    database_agent = None

    if dataframes:
        try:
            if len(dataframes) == 1:
                data_agent = create_data_agent(
                    dataframes[0]
                )
            else:
                data_agent = create_data_agent(
                    dataframes
                )

        except Exception as e:
            st.error(
                f"Error : {e}"
            )
            st.exception(e)

    if texts:
        try:
            with st.spinner(
                "Processing"
            ):
                vectorstore = build_RAG(
                    texts
                )

        except Exception as e:
            st.error(
                f"Error RAG: {e}"
            )
            st.exception(e)

    if databases:
        try:
            database_file = databases[0]
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".db"
            ) as temp_db:

                temp_db.write(
                    database_file.getvalue()
                )

                database_path = (
                    temp_db.name
                )

            database_agent = (
                create_database_agent(
                    database_path
                )
            )

        except Exception as e:
            st.error(
                f"Error creating "
                f"Database Agent: {e}"
            )
            st.exception(e)

    st.divider()
    st.subheader(
        "Ask Your Data"
    )

    question = st.text_input(
        "Ask a question about "
        "your uploaded data"
    )

    if question:

        try:
            with st.spinner(
                "Understanding your question..."
            ):
                routes = (
                    query_router.route_question(
                        question,
                        available_sources
                    )
                )
            if routes:
                st.caption(
                    "Sources selected: "
                    + ", ".join(
                        routes.keys()
                    )
                )

            else:
                st.caption(
                    "No valid source selected."
                )

            answers = {}

            if (
                "TABULAR" in routes
                and data_agent is not None
            ):

                tabular_question = (
                    routes["TABULAR"]
                )

                with st.spinner(
                    "Analyzing tabular data"
                ):

                    answers["TABULAR"] = (
                        ask_data(
                            data_agent,
                            tabular_question
                        )
                    )

            if (
                "DOCUMENT" in routes
                and vectorstore is not None
            ):

                document_question = (
                    routes["DOCUMENT"]
                )

                with st.spinner(
                    "Searching documents..."
                ):

                    answers["DOCUMENT"] = (
                        ask_RAG(
                            vectorstore,
                            document_question
                        )
                    )

            if (
                "DATABASE" in routes
                and database_agent is not None
            ):

                database_question = (
                    routes["DATABASE"]
                )

                with st.spinner(
                    "Querying"
                ):

                    answers["DATABASE"] = (
                        ask_database(
                            database_agent,
                            database_question
                        )
                    )

            if not answers:

                answer = (
                    "I could not determine "
                    "the correct data source "
                    "for this question."
                )

            elif len(answers) == 1:
                answer = list(
                    answers.values()
                )[0]

            else:
                with st.spinner(
                    "Combining results"
                ):
                    answer = (
                        query_router.combine_answers(
                            question,
                            answers
                        )
                    )

            st.subheader(
                "Answer"
            )

            st.write(
                answer
            )


        except Exception as e:

            st.error(
                f"Error processing "
                f"question: {e}"
            )

            st.exception(e)

else:
    st.info(
        "Upload at least one file to start"
    )