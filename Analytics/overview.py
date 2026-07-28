import pandas as pd

class DatasetOverview:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze(self):
        df = self.df
        overview = {
            "rows": len(df),
            "columns": len(df.columns),

            "numeric_columns":
                len(df.select_dtypes(include="number").columns),

            "categorical_columns":
                len(df.select_dtypes(include="object").columns),

            "datetime_columns":
                len(df.select_dtypes(include="datetime").columns),

            "missing_values":
                int(df.isnull().sum().sum()),

            "duplicate_rows":
                int(df.duplicated().sum()),

            "memory_usage":
                round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            
            "column_summary": self.column_summary()
        }

        return overview

    def column_summary(self):
        df = self.df
        summary = pd.DataFrame({

            "Column": df.columns,

            "Type": df.dtypes.astype(str),

            "Missing": df.isnull().sum().values,

            "Unique": df.nunique().values
        })
        return summary
