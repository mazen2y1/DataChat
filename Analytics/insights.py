import pandas as pd

class AIInsights:
    def __init__(self, df, schema, overview, kpis):
        self.df = df
        self.schema = schema or {}
        self.overview = overview or {}
        self.kpis = kpis or {}

    def analyze(self):
        insights = {
            "summary": [],
            "positive": [],
            "warnings": [],
            "trends": [],
            "interesting": []
        }

        self._dataset_summary(insights)
        self._dataset_quality(insights)
        self._business_insights(insights)
        self._trend_analysis(insights)
        self._category_analysis(insights)
        self._region_analysis(insights)
        self._interesting_facts(insights)

        return insights

    def _dataset_summary(self, insights):
        insights["summary"].append(
            f"Dataset contains {len(self.df):,} rows and {len(self.df.columns)} columns"
        )

        insights["summary"].append(
            f"{len(self.kpis)} KPIs detected automatically"
        )

    def _dataset_quality(self, insights):
        missing = self.overview.get("missing_values", 0)

        if missing == 0:
            insights["positive"].append(
                "No missing values detected"
            )
        else:
            insights["warnings"].append(
                f"{missing:,} missing values detected"
            )

        duplicates = self.overview.get("duplicate_rows", 0)
        if duplicates:
            insights["warnings"].append(
                f"{duplicates:,} duplicate rows detected"
            )

    def _business_insights(self, insights):
        revenue = self.kpis.get("Total Revenue")

        if revenue is not None:
            insights["positive"].append(
                f"Total revenue reached {revenue:,.2f}"
            )

        profit = self.kpis.get("Total Profit")
        if profit is not None:
            insights["positive"].append(
                f"Total profit reached {profit:,.2f}"
            )

            if profit < 0:
                insights["warnings"].append(
                    "Overall business is operating at a loss"
                )

        revenue_col = self.schema.get("revenue")
        profit_col = self.schema.get("profit")

        if revenue_col and revenue_col in self.df.columns:

            insights["interesting"].append(
                f"Average order value is {self.df[revenue_col].mean():,.2f}"
            )

            insights["interesting"].append(
                f"Highest single sale is {self.df[revenue_col].max():,.2f}"
            )

        if profit_col and profit_col in self.df.columns:
            negative_orders = (self.df[profit_col] < 0).sum()
            if negative_orders > 0:
                insights["warnings"].append(
                    f"{negative_orders} orders generated negative profit"
                )

    def _trend_analysis(self, insights):

        date_col = self.schema.get("date")
        revenue_col = self.schema.get("revenue")

        if not date_col:
            return

        if not revenue_col:
            return

        if date_col not in self.df.columns:
            return

        if revenue_col not in self.df.columns:
            return

        try:

            df = self.df.copy()

            df[date_col] = pd.to_datetime(
                df[date_col],
                errors="coerce"
            )

            df = df.dropna(subset=[date_col])

            trend = (
                df.groupby(date_col)[revenue_col]
                .sum()
                .sort_index()
            )

            if len(trend) < 2:
                return

            if trend.iloc[-1] > trend.iloc[0]:
                insights["trends"].append(
                    "Revenue shows an upward trend over time"
                )

            elif trend.iloc[-1] < trend.iloc[0]:
                insights["trends"].append(
                    "Revenue shows a downward trend over time"
                )

            else:
                insights["trends"].append(
                    "Revenue remained relatively stable"
                )

        except Exception:
            pass

    def _category_analysis(self, insights):
        revenue_col = self.schema.get("revenue")
        category_col = self.schema.get("category")

        if not revenue_col:
            return

        if not category_col:
            return

        if revenue_col not in self.df.columns:
            return

        if category_col not in self.df.columns:
            return

        try:
            grouped = (
                self.df.groupby(category_col)[revenue_col]
                .sum()
                .sort_values(ascending=False)
            )

            if len(grouped) == 0:
                return

            insights["trends"].append(
                f"{grouped.index[0]} the top revenue category"
            )

            insights["interesting"].append(
                f"{grouped.index[-1]} the lowest revenue"
            )

        except Exception:
            pass

    def _region_analysis(self, insights):

        revenue_col = self.schema.get("revenue")

        region_col = (
            self.schema.get("region")
            or self.schema.get("location")
            or self.schema.get("state")
        )
        if not revenue_col:
            return

        if not region_col:
            return

        if revenue_col not in self.df.columns:
            return
        if region_col not in self.df.columns:
            return

        try:
            grouped = (
                self.df.groupby(region_col)[revenue_col]
                .sum()
                .sort_values(ascending=False)
            )

            if len(grouped):
                insights["trends"].append(
                    f"{grouped.index[0]} is the highest revenue region"
                )

        except Exception:
            pass

    def _interesting_facts(self, insights):
        revenue_col = self.schema.get("revenue")

        if revenue_col and revenue_col in self.df.columns:
            q1 = self.df[revenue_col].quantile(.25)
            q3 = self.df[revenue_col].quantile(.75)

            iqr = q3 - q1

            outliers = (
                self.df[revenue_col]
                > q3 + 1.5 * iqr
            ).sum()

            if outliers:

                insights["warnings"].append(
                    f"{outliers} unusually high sales detected"
                )

        discount_col = self.schema.get("discount")
        if (
            discount_col
            and discount_col in self.df.columns
        ):
            avg_discount = self.df[discount_col].mean()
            if avg_discount > 0.30:
                insights["warnings"].append(
                    "Average discount is high"
                )