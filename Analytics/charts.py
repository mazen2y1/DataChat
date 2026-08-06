from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class _Rules:
    
    MAX_MISSING_RATIO = 0.95
    MAX_UNIQUE_RATIO_FOR_ID = 0.98
    MIN_ROWS_FOR_ID_CHECK = 20
    ID_NAME_HINTS = ("id", "uuid", "guid", "index", "_pk", "pk_")
    MIN_DATETIME_PARSE_SUCCESS = 0.90

    TOP_N_NUMERIC_FOR_SCATTER = 2
    MIN_NUMERIC_FOR_CORRELATION = 3
    MAX_CORRELATION_COLUMNS = 12

    MAX_CATEGORIES_FOR_PIE = 8
    MAX_CATEGORIES_FOR_TREEMAP = 20
    MIN_CATEGORIES_FOR_TREEMAP = 2
    TOP_N_CATEGORIES_BAR = 10
    MAX_CATEGORY_CARDINALITY = 50

    MIN_UNIQUE_FOR_CONTINUOUS = 10
    HISTOGRAM_MAX_BINS = 40

    MAX_CHARTS = 10
    MIN_CHARTS_TARGET = 6

    LARGE_DATASET_ROWS = 50_000
    SAMPLE_SIZE_FOR_LARGE_SCATTER = 20_000

@dataclass
class _ColumnProfile:

    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    boolean: List[str] = field(default_factory=list)
    missing_ratio: Dict[str, float] = field(default_factory=dict)

    @property
    def has_any_usable_column(self) -> bool:
        return bool(self.numeric or self.categorical or self.datetime or self.boolean)

class ChartGenerator:
    
    def __init__(self, df: pd.DataFrame, theme: str = "plotly") -> None:
        self._raw_df = df if df is not None else pd.DataFrame()
        self._theme = theme
        self._df: Optional[pd.DataFrame] = None
        self._profile: Optional[_ColumnProfile] = None

    def style_chart(self, fig):
        fig.update_layout(
        paper_bgcolor="#050B2B",
        plot_bgcolor="#050B2B",
        font_color="#F8FAFC",
        colorway=[
            "#3159B4",
            "#0E6A86",
            "#D8A95B",
            "#4F46E5",
            "#60A5FA",
            "#38BDF8"
            ]
                )
        return fig

    def generate_dashboard(self) -> Dict[str, go.Figure]:
        figures: Dict[str, go.Figure] = {}

        if self._raw_df is None or self._raw_df.empty:
            figures["empty"] = self._empty_state_figure(
                "No data available",
                "The provided dataset is empty.",
            )
            return figures

        try:
            self._df = self._prepare_dataframe(self._raw_df)
            self._profile = self._profile_columns(self._df)
        except Exception:
            logger.exception("Failed")
            figures["error"] = self._empty_state_figure(
                "Unable to analyze dataset",
                "The dataset could not be processed",
            )
            return figures

        if not self._profile.has_any_usable_column:
            figures["empty"] = self._empty_state_figure(
                "No usable columns",
                "missing data",
            )
            return figures

        builders = self._select_chart_builders()
        for key, builder in builders:
            if len(figures) >= _Rules.MAX_CHARTS:
                break
            try:
                fig = builder()
            except Exception:
                logger.exception("Chart builder '%s' failed", key)
                continue
            if fig is not None:
                figures[key] = self._finalize_layout(fig)

        if not figures:
            figures["empty"] = self._empty_state_figure(
                "No charts could be generated",
                "The dataset did not contain enough data"
            )
        return figures

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy(deep=False)

        for col in working.columns:
            if working[col].dtype == object:
                parsed = self._try_parse_datetime(working[col])
                if parsed is not None:
                    working[col] = parsed

        return working

    @staticmethod
    def _try_parse_datetime(series: pd.Series) -> Optional[pd.Series]:
        non_null = series.dropna()
        if non_null.empty:
            return None

        sample = non_null.astype(str).head(50)
        if sample.str.match(r"^\d+$").mean() > 0.8:
            return None

        try:
            parsed = pd.to_datetime(series, errors="coerce", utc=False)
        except (ValueError, TypeError):
            return None

        success_ratio = parsed.notna().sum() / max(non_null.shape[0], 1)
        if success_ratio >= _Rules.MIN_DATETIME_PARSE_SUCCESS:
            return parsed
        return None

    def _profile_columns(self, df: pd.DataFrame) -> _ColumnProfile:
        profile = _ColumnProfile()
        n_rows = len(df)

        for col in df.columns:
            series = df[col]
            missing_ratio = series.isna().mean() if n_rows else 1.0
            profile.missing_ratio[col] = float(missing_ratio)

            if missing_ratio > _Rules.MAX_MISSING_RATIO:
                continue
            if self._is_constant(series):
                continue
            is_datetime_like = pd.api.types.is_datetime64_any_dtype(series)
            skip_uniqueness_check = is_datetime_like or pd.api.types.is_float_dtype(series)
            if self._looks_like_identifier(col, series, n_rows, skip_uniqueness_check):
                continue

            if pd.api.types.is_bool_dtype(series):
                profile.boolean.append(col)
            elif pd.api.types.is_datetime64_any_dtype(series):
                profile.datetime.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                profile.numeric.append(col)
            else:
                nunique = series.nunique(dropna=True)
                if nunique <= _Rules.MAX_CATEGORY_CARDINALITY or n_rows == 0:
                    profile.categorical.append(col)

        return profile

    @staticmethod
    def _is_constant(series: pd.Series) -> bool:
        non_null = series.dropna()
        if non_null.empty:
            return True
        return non_null.nunique() <= 1

    @staticmethod
    def _looks_like_identifier(name: str, series: pd.Series, n_rows: int, skip_uniqueness_check: bool = False) -> bool:
        if n_rows == 0:
            return False

        lowered = str(name).strip().lower()
        if any(hint == lowered or lowered.endswith(hint) or lowered.startswith(hint)
               for hint in _Rules.ID_NAME_HINTS):
            return True

        if skip_uniqueness_check:
            return False

        if n_rows >= _Rules.MIN_ROWS_FOR_ID_CHECK:
            unique_ratio = series.nunique(dropna=True) / n_rows
            if unique_ratio >= _Rules.MAX_UNIQUE_RATIO_FOR_ID:
                return True

        return False

    def _rank_numeric_columns(self) -> List[str]:
        df, profile = self._df, self._profile
        scores: List[Tuple[str, float]] = []
        n_rows = len(df)

        for col in profile.numeric:
            series = df[col].dropna()
            if series.empty or n_rows == 0:
                continue

            missing_penalty = 1.0 - profile.missing_ratio.get(col, 0.0)

            std = series.std()
            mean_abs = series.abs().mean()
            normalized_variance = (std / mean_abs) if mean_abs not in (0, np.nan) and not np.isnan(mean_abs) else (std or 0.0)
            normalized_variance = 0.0 if pd.isna(normalized_variance) else float(normalized_variance)

            uniqueness = series.nunique() / max(len(series), 1)

            score = (0.4 * missing_penalty) + (0.4 * min(normalized_variance, 5.0) / 5.0) + (0.2 * uniqueness)
            scores.append((col, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [col for col, _ in scores]

    def _rank_categorical_columns(self) -> List[str]:
        df, profile = self._df, self._profile
        scores: List[Tuple[str, float]] = []
        n_rows = len(df)
        if n_rows == 0:
            return []

        for col in profile.categorical:
            series = df[col].dropna()
            if series.empty:
                continue

            nunique = series.nunique()
            if nunique < 2:
                continue

            missing_penalty = 1.0 - profile.missing_ratio.get(col, 0.0)

            ideal = 6.0
            cardinality_score = 1.0 / (1.0 + abs(nunique - ideal) / ideal)

            freq = series.value_counts(normalize=True)
            top_share = float(freq.iloc[0]) if not freq.empty else 1.0
            balance_score = 1.0 - top_share

            score = (0.4 * missing_penalty) + (0.35 * cardinality_score) + (0.25 * balance_score)
            scores.append((col, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [col for col, _ in scores]

    def _select_chart_builders(self) -> List[Tuple[str, "callable"]]:
        profile = self._profile
        ranked_numeric = self._rank_numeric_columns()
        ranked_categorical = self._rank_categorical_columns()

        builders: List[Tuple[str, "callable"]] = []

        continuous_numeric = self._continuous_numeric_columns(ranked_numeric)
        if continuous_numeric:
            builders.append(("distribution", lambda: self._build_histogram(continuous_numeric[0])))

        if continuous_numeric:
            builders.append(("boxplot", lambda: self._build_boxplot(continuous_numeric[0], ranked_categorical)))

        if ranked_categorical:
            builders.append(("top_categories", lambda: self._build_top_categories(ranked_categorical[0])))

        if ranked_categorical and ranked_numeric:
            builders.append(("category", lambda: self._build_category_bar(ranked_categorical[0], ranked_numeric[0])))

        if ranked_categorical:
            best_cat = ranked_categorical[0]
            nunique = self._df[best_cat].nunique(dropna=True)
            if nunique <= _Rules.MAX_CATEGORIES_FOR_PIE:
                builders.append(("pie", lambda: self._build_pie(best_cat)))

        if len(ranked_numeric) >= _Rules.TOP_N_NUMERIC_FOR_SCATTER:
            builders.append((
                "scatter",
                lambda: self._build_scatter(ranked_numeric[0], ranked_numeric[1], ranked_categorical),
            ))

        if len(profile.numeric) >= _Rules.MIN_NUMERIC_FOR_CORRELATION:
            builders.append(("correlation", lambda: self._build_correlation(ranked_numeric)))

        if profile.datetime:
            best_datetime = profile.datetime[0]
            builders.append(("trend", lambda: self._build_trend(best_datetime, ranked_numeric)))

        if any(ratio > 0 for ratio in profile.missing_ratio.values()):
            builders.append(("missing", lambda: self._build_missing_chart()))

        if ranked_categorical and ranked_numeric:
            best_cat = ranked_categorical[0]
            nunique = self._df[best_cat].nunique(dropna=True)
            if _Rules.MIN_CATEGORIES_FOR_TREEMAP <= nunique <= _Rules.MAX_CATEGORIES_FOR_TREEMAP:
                builders.append(("treemap", lambda: self._build_treemap(best_cat, ranked_numeric[0])))

        return builders

    def _continuous_numeric_columns(self, ranked_numeric: Sequence[str]) -> List[str]:
        df = self._df
        continuous = []
        for col in ranked_numeric:
            if df[col].dropna().nunique() >= _Rules.MIN_UNIQUE_FOR_CONTINUOUS:
                continuous.append(col)
        return continuous

    def _build_histogram(self, column: str) -> go.Figure:
        df = self._df
        fig = px.histogram(
            df,
            x=column,
            nbins=min(_Rules.HISTOGRAM_MAX_BINS, max(10, int(df[column].dropna().nunique() ** 0.5) * 3)),
            title=f"Distribution of {self._label(column)}",
            template=self._theme,
        )
        fig.update_layout(
            xaxis_title=self._label(column),
            yaxis_title="Count",
            bargap=0.05,
        )
        fig.update_traces(hovertemplate=f"{self._label(column)}: %{{x}}<br>Count: %{{y}}<extra></extra>")
        return fig

    def _build_boxplot(self, numeric_col: str, ranked_categorical: Sequence[str]) -> go.Figure:
        df = self._df
        group_col = ranked_categorical[0] if ranked_categorical and self._df[ranked_categorical[0]].nunique() <= 12 else None

        if group_col:
            fig = px.box(
                df,
                x=group_col,
                y=numeric_col,
                title=f"{self._label(numeric_col)} Distribution by {self._label(group_col)}",
                template=self._theme,
            )
            fig.update_layout(xaxis_title=self._label(group_col), yaxis_title=self._label(numeric_col))
        else:
            fig = px.box(
                df,
                y=numeric_col,
                title=f"{self._label(numeric_col)} - Outlier Overview",
                template=self._theme,
            )
            fig.update_layout(yaxis_title=self._label(numeric_col))
        return fig

    def _build_top_categories(self, column: str) -> go.Figure:
        counts = (
            self._df[column]
            .dropna()
            .value_counts()
            .head(_Rules.TOP_N_CATEGORIES_BAR)
            .sort_values(ascending=True)
        )
        fig = px.bar(
            x=counts.values,
            y=counts.index.astype(str),
            orientation="h",
            title=f"Top {self._label(column)} Categories",
            template=self._theme,
            labels={"x": "Count", "y": self._label(column)},
        )
        fig.update_traces(hovertemplate=f"{self._label(column)}: %{{y}}<br>Count: %{{x}}<extra></extra>")
        return fig

    def _build_category_bar(self, category_col: str, numeric_col: str) -> go.Figure:
        df = self._df
        grouped = (
            df.groupby(category_col, dropna=True)[numeric_col]
            .mean()
            .dropna()
            .sort_values(ascending=False)
            .head(_Rules.TOP_N_CATEGORIES_BAR)
        )
        fig = px.bar(
            x=grouped.index.astype(str),
            y=grouped.values,
            title=f"Average {self._label(numeric_col)} by {self._label(category_col)}",
            template=self._theme,
            labels={"x": self._label(category_col), "y": f"Avg {self._label(numeric_col)}"},
        )
        fig.update_traces(hovertemplate=f"{self._label(category_col)}: %{{x}}<br>Avg {self._label(numeric_col)}: %{{y}}<extra></extra>")
        return fig

    def _build_pie(self, column: str) -> go.Figure:
        counts = self._df[column].dropna().value_counts()
        fig = px.pie(
            names=counts.index.astype(str),
            values=counts.values,
            title=f"Share by {self._label(column)}",
            template=self._theme,
            hole=0.35,
        )
        fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>")
        return fig

    def _build_scatter(self, x_col: str, y_col: str, ranked_categorical: Sequence[str]) -> go.Figure:
        df = self._df
        color_col = None
        if ranked_categorical:
            candidate = ranked_categorical[0]
            if df[candidate].nunique(dropna=True) <= 10:
                color_col = candidate

        plot_df = df
        if len(df) > _Rules.LARGE_DATASET_ROWS:
            plot_df = df.sample(n=_Rules.SAMPLE_SIZE_FOR_LARGE_SCATTER, random_state=42)

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"{self._label(x_col)} vs {self._label(y_col)}",
            template=self._theme,
            opacity=0.75,
        )
        fig.update_layout(xaxis_title=self._label(x_col), yaxis_title=self._label(y_col))
        return fig

    def _build_correlation(self, ranked_numeric: Sequence[str]) -> Optional[go.Figure]:
        cols = list(ranked_numeric[: _Rules.MAX_CORRELATION_COLUMNS])
        if len(cols) < _Rules.MIN_NUMERIC_FOR_CORRELATION:
            return None

        corr = self._df[cols].corr(numeric_only=True)
        if corr.isna().all().all():
            return None

        labels = [self._label(c) for c in corr.columns]
        fig = px.imshow(
            corr,
            x=labels,
            y=labels,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Between Numeric Columns",
            template=self._theme,
            text_auto=".2f",
        )
        fig.update_layout(coloraxis_colorbar=dict(title="corr"))
        return fig

    def _build_trend(self, datetime_col: str, ranked_numeric: Sequence[str]) -> Optional[go.Figure]:
        df = self._df
        valid = df[[datetime_col]].dropna()
        if valid.empty:
            return None

        span_days = (valid[datetime_col].max() - valid[datetime_col].min()).days
        freq = self._pick_resample_frequency(span_days)

        if ranked_numeric:
            value_col = ranked_numeric[0]
            indexed = (
                df[[datetime_col, value_col]]
                .dropna(subset=[datetime_col])
                .set_index(datetime_col)[value_col]
            )
            series = self._safe_resample(indexed, freq, "mean")
            y_label = f"Avg {self._label(value_col)}"
        else:
            indexed = (
                df[[datetime_col]]
                .dropna(subset=[datetime_col])
                .set_index(datetime_col)
                .assign(_count=1)["_count"]
            )
            series = self._safe_resample(indexed, freq, "sum")
            y_label = "Record Count"

        series = series.dropna()
        if series.empty:
            return None

        fig = px.line(
            x=series.index,
            y=series.values,
            title=f"Trend Over Time ({self._label(datetime_col)})",
            template=self._theme,
            markers=len(series) <= 60,
            labels={"x": self._label(datetime_col), "y": y_label},
        )
        fig.update_layout(xaxis_title=self._label(datetime_col), yaxis_title=y_label)
        return fig

    @staticmethod
    def _safe_resample(series: pd.Series, freq: str, how: str) -> pd.Series:
        legacy_aliases = {"ME": "M", "YE": "Y"}
        try:
            resampler = series.resample(freq)
        except ValueError:
            resampler = series.resample(legacy_aliases.get(freq, freq))
        return getattr(resampler, how)()

    @staticmethod
    def _pick_resample_frequency(span_days: int) -> str:
        if span_days <= 2:
            return "h"
        if span_days <= 90:
            return "D"
        if span_days <= 730:
            return "W"
        if span_days <= 365 * 8:
            return "ME"
        return "YE"

    def _build_missing_chart(self) -> Optional[go.Figure]:
        profile = self._profile
        missing = {col: ratio for col, ratio in profile.missing_ratio.items() if ratio > 0}
        if not missing:
            return None

        series = pd.Series(missing).sort_values(ascending=True) * 100.0
        fig = px.bar(
            x=series.values,
            y=[self._label(c) for c in series.index],
            orientation="h",
            title="Missing Values by Column",
            template=self._theme,
            labels={"x": "Missing (%)", "y": "Column"},
        )
        fig.update_traces(hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>")
        fig.update_layout(xaxis_range=[0, 100])
        return fig

    def _build_treemap(self, category_col: str, numeric_col: str) -> Optional[go.Figure]:
        df = self._df
        grouped = (
            df.groupby(category_col, dropna=True)[numeric_col]
            .sum()
            .dropna()
        )
        grouped = grouped[grouped > 0]
        if grouped.empty:
            return None

        fig = px.treemap(
            names=grouped.index.astype(str),
            parents=[""] * len(grouped),
            values=grouped.values,
            title=f"{self._label(numeric_col)} Composition by {self._label(category_col)}",
            template=self._theme,
        )
        fig.update_traces(hovertemplate="%{label}<br>%{value}<extra></extra>")
        return fig

    @staticmethod
    def _label(column_name: str) -> str:
        return str(column_name).replace("_", " ").replace("-", " ").strip().title()

    def _finalize_layout(self, fig: go.Figure) -> go.Figure:
        fig = self.style_chart(fig)
        fig.update_layout(
            template=self._theme,
            margin=dict(l=40, r=30, t=60, b=40),
            title_font_size=18,
            hoverlabel=dict(font_size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def _empty_state_figure(self, title: str, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(
            title=title,
            template=self._theme,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig