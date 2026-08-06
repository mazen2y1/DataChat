import streamlit as st
import pandas as pd
import tempfile
from DataLoader import load_file
from DataAgent import create_data_agent, ask_data
from RAG import build_RAG, ask_RAG
from SQLAgent import create_database_agent, ask_database
import query_router
from Analytics.overview import DatasetOverview
from Analytics.kpi import KPIEngine
from Analytics.SchemaAnalyzer import SchemaAnalyzer
from LLMProvider import get_llm
from Analytics.charts import ChartGenerator
from Analytics.insights import AIInsights
import streamlit.components.v1 as components
st.set_page_config(
         page_title="Data Chat",
        page_icon="📊",
        layout="wide"
    )
with open(".streamlit/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
tab_chat, tab_analytics = st.tabs([
    "💬 Data Chat",
    "📊 AI Business Analytics"
])
with tab_chat:
    st.markdown("""
    <div style="
    background: linear-gradient(90deg,#10A37F,#2563EB);
    padding:30px;
    border-radius:18px;
    color:white;
        text-align:center;
                    margin-bottom:25px;
                ">
                    <h1 style="margin-bottom:8px;">🤖 Data Chat</h1>
                    <p style="font-size:18px;">
                        Chat with CSV, Excel, PDF & SQL using Artificial Intelligence
                    </p>
                </div>
                """,unsafe_allow_html=True)
    st.title("Start Chatting Your Data Now")
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
        dataframe_files = []
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
                    dataframe_files.append(file.name)
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
                    f"Error"
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
                if len(dataframes)== 1:
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
                    f"Error"
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
                    "Understanding.."
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
                        "No valid source"
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
                        "Searching documents"
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
                    f"Error"
                    f"question: {e}"
                )

                st.exception(e)
    else:
        st.info(
            "Upload at least one file to start"
        )

with tab_analytics:
    st.markdown("""
        <div style="
            background: linear-gradient(90deg,#3B82F6,#10A37F);
            padding:25px;
            border-radius:18px;
            color:white;
            text-align:center;
            margin-bottom:20px;
        ">
            <h1>AI Business Analytics</h1>
            <p>Automatic KPIs • Smart Charts • AI Insights</p>
        </div>
        """, unsafe_allow_html=True)
    st.title("Analyze The Selected Data")
    progress = st.progress(0)
    status = st.empty()
    if not uploaded_files:
        st.warning("Upload your data")

    else:
        if dataframes:
            selected_dataset = st.selectbox(
                "Select Dataset",
                dataframe_files
            )
            status.success("Dataset Loaded")
            progress.progress(10)

        if st.button("📊Analyze Dataset"):
            if not dataframes:
                st.error("No data found (CSV or Excel)")
                st.stop()
            selected_index = dataframe_files.index(selected_dataset)
            df = dataframes[selected_index]
            analyzer = DatasetOverview(df)
            overview = analyzer.analyze()
            status.success("Analyzing")
            progress.progress(25)
            st.subheader("Dataset Overview")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Rows", overview["rows"])
                st.metric("Columns", overview["columns"])
                st.metric("Missing", overview["missing_values"])

            with col2:

                st.metric("Duplicates", overview["duplicate_rows"])
                st.metric("Numeric", overview["numeric_columns"])
                st.subheader("Column Summary")

            st.dataframe(
                overview["column_summary"],
                use_container_width=True,
                hide_index=True
            )
            llm = get_llm()
            schema = SchemaAnalyzer(llm).analyze(df)
            status.success("Schema Detected")
            progress.progress(45)
            #st.write(schema)
            kpi_engine = KPIEngine(df, schema)
            kpis = kpi_engine.analyze()
            status.success("✅ KPIs Generated")
            progress.progress(60)
            if kpis:
                st.subheader("Key Performance Indicators")
                cols = st.columns(4)
                for i, (name, value) in enumerate(kpis.items()):
                    cols[i % 4].metric(name, value)
            else:
                st.info("No KPIs detected for this dataset")
            status.info("Building Dashboard...")
            progress.progress(75)
            charts = ChartGenerator(df)
            dashboard = charts.generate_dashboard()
            status.success("Dashboard is Ready")
            progress.progress(85)
            items = [(title, fig) for title, fig in dashboard.items() if fig is not None]

            for i in range(0, len(items), 2):
                col1, col2 = st.columns(2)

                title1, fig1 = items[i]
                with col1:
                    st.subheader(title1.replace("_", " ").title())
                    st.plotly_chart(fig1, use_container_width=True)

                if i + 1 < len(items):
                    title2, fig2 = items[i + 1]
                    with col2:
                        st.subheader(title2.replace("_", " ").title())
                        st.plotly_chart(fig2, use_container_width=True)
            status.info("AI is generating insights...")
            progress.progress(94)
            insights = AIInsights(
                df=df,
                schema=schema,
                overview=overview,
                kpis=kpis
            ).analyze()
            st.divider()
            st.subheader("AI Insights")
            if insights["summary"]:
                st.markdown("Summary")
                for item in insights["summary"]:
                    st.info(item)

            if insights["positive"]:
                st.markdown("Positive Findings")
                for item in insights["positive"]:
                    st.success(item)

            if insights["warnings"]:
                st.markdown("Warnings")
                for item in insights["warnings"]:
                    st.warning(item)
            if insights["trends"]:
                st.markdown("Trends")
                for item in insights["trends"]:
                    st.info(item)
            if insights["interesting"]:
                st.markdown("Interesting Facts")
                for item in insights["interesting"]:
                    st.info(item)
            progress.progress(100)
            status.success("Analysis Finished")