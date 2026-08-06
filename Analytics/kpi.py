import pandas as pd
class KPIEngine:
    def __init__(self, df, schema):
        self.df = df
        self.schema = schema

    def analyze(self):
        kpis = {}
        revenue = self.schema.get("revenue")
        profit = self.schema.get("profit")
        quantity = self.schema.get("quantity")
        customer = self.schema.get("customer")
        store = self.schema.get("store")
        date = self.schema.get("date")

        def normalize(col):
            if isinstance(col, list):
                return col[0] if col else None
            return col

        revenue = normalize(revenue)
        profit = normalize(profit)
        quantity = normalize(quantity)
        customer = normalize(customer)
        store = normalize(store)
        date = normalize(date)
        
        if isinstance(revenue, list):
            revenue = revenue[0] if revenue else None
        if revenue in self.df.columns:
            kpis["Total Revenue"] = round(self.df[revenue].sum(), 2)
            kpis["Average Revenue"] = round(self.df[revenue].mean(), 2)
            print("Revenue Column:", revenue)
            print(self.df[revenue].describe())
            kpis["Max Revenue"] = round(self.df[revenue].max(), 2)
            kpis["Min Revenue"] = round(self.df[revenue].min(), 2)

        if profit in self.df.columns:
            kpis["Total Profit"] = round(self.df[profit].sum(), 2)

        if quantity in self.df.columns:
            kpis["Total Quantity"] = int(self.df[quantity].sum())

        if customer in self.df.columns:
            kpis["Customers"] = self.df[customer].nunique()

        if store in self.df.columns:
            kpis["Stores"] = self.df[store].nunique()

        if date in self.df.columns:
            dates = pd.to_datetime(
                self.df[date],
                dayfirst=True,
                errors="coerce"
            )
            dates = dates.dropna()

            if not dates.empty:
                kpis["Period"] = (
                    f"{dates.min().strftime('%d-%m-%Y')} → "
                    f"{dates.max().strftime('%d-%m-%Y')}"
                )

        return kpis