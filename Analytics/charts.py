from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class ChartGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = list(
            self.df.select_dtypes(include="number").columns
        )
        self.categorical_cols = list(
            self.df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns
        )
        self.datetime_cols = list(
            self.df.select_dtypes(
                include=["datetime64[ns]", "datetime64"]
            ).columns
        )

    def has_numeric(self) -> bool:
        return len(self.numeric_cols) > 0

    def has_categorical(self) -> bool:
        return len(self.categorical_cols) > 0

    def has_datetime(self) -> bool:
        return len(self.datetime_cols) > 0

    def histogram(
        self,
        column: str,
        bins: int = 30,
    ) -> go.Figure:

        fig = px.histogram(
            self.df,
            x=column,
            nbins=bins,
            title=f"Distribution of {column}",
        )

        return fig

    def bar_chart(
        self,
        x: str,
        y: str,
    ) -> go.Figure:

        fig = px.bar(
            self.df,
            x=x,
            y=y,
            title=f"{y} by {x}",
        )

        return fig
    
    def line_chart(
        self,
        x: str,
        y: str,
    ) -> go.Figure:

        fig = px.line(
            self.df,
            x=x,
            y=y,
            markers=True,
            title=f"{y} over {x}",
        )
        return fig

    def pie_chart(
        self,
        names: str,
        values: str,
    ) -> go.Figure:
        fig = px.pie(
            self.df,
            names=names,
            values=values,
            hole=0.4,
            title=f"{values} by {names}",
        )

        return fig

    def scatter_plot(
        self,
        x: str,
        y: str,
        color: Optional[str] = None,
    ) -> go.Figure:

        fig = px.scatter(
            self.df,
            x=x,
            y=y,
            color=color,
            title=f"{x} vs {y}",
        )

        return fig

    def box_plot(
        self,
        column: str,
    ) -> go.Figure:

        fig = px.box(
            self.df,
            y=column,
            title=f"Box Plot - {column}",
        )
        return fig

    def correlation_heatmap(self) -> Optional[go.Figure]:
        if len(self.numeric_cols) < 2:
            return None
        corr = self.df[self.numeric_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap",
        )

        return fig

    def missing_values_chart(self) -> Optional[go.Figure]:
        missing = self.df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return None

        fig = px.bar(
            x=missing.index,
            y=missing.values,
            labels={
                "x": "Columns",
                "y": "Missing Values",
            },
            title="Missing Values",
        )

        return fig

    def top_categories(
        self,
        column: str,
        top_n: int = 10,
    ) -> go.Figure:

        counts = (
            self.df[column]
            .value_counts()
            .head(top_n)
        )

        fig = px.bar(
            x=counts.index,
            y=counts.values,
            labels={
                "x": column,
                "y": "Count",
            },
            title=f"Top {top_n} {column}",
        )

        return fig

    def generate_dashboard(self) -> Dict[str, go.Figure]:
        charts: Dict[str, go.Figure] = {}
        if self.numeric_cols:

            charts["distribution"] = self.histogram(
                self.numeric_cols[0]
            )

            charts["boxplot"] = self.box_plot(
                self.numeric_cols[0]
            )

        if (
            self.categorical_cols
            and self.numeric_cols
        ):

            grouped = (
                self.df
                .groupby(self.categorical_cols[0])[self.numeric_cols[0]]
                .sum()
                .reset_index()
            )

            fig = px.bar(
                grouped,
                x=self.categorical_cols[0],
                y=self.numeric_cols[0],
                title="Category Analysis",
            )
            charts["category"] = fig
        heatmap = self.correlation_heatmap()
        if heatmap is not None:
            charts["correlation"] = heatmap
        missing = self.missing_values_chart()
        if missing is not None:
            charts["missing"] = missing

        return charts