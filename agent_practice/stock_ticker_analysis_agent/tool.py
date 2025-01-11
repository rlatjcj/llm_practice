import pandas as pd
import yfinance as yf
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools.tavily_search import TavilySearchResults
import plotly.graph_objects as go
from datetime import datetime, timedelta


class StockTickerAnalysisTool:
    """Tool for analyzing stock tickers."""

    def __init__(self) -> None:
        """Initialize the tool with necessary components."""
        self.tavily_tool = TavilySearchResults(max_results=5)

    def analyze_stock_ticker(self, ticker: str) -> str:
        """Analyze a stock ticker and return a summary of the stock's performance and financial data."""

        def format_number(number):
            if number is None or pd.isna(number):
                return "N/A"
            return f"{number:,.0f}"

        def format_financial_summary(financials):
            summary = {}
            for date, data in financials.items():
                date_str = date.strftime("%Y-%m-%d")
                summary[date_str] = {
                    "총수익": format_number(data.get("TotalRevenue")),
                    "영업이익": format_number(data.get("OperatingIncome")),
                    "순이익": format_number(data.get("NetIncome")),
                    "EBITDA": format_number(data.get("EBITDA")),
                    "EPS(희석)": (
                        f"${data.get('DilutedEPS'):.2f}"
                        if pd.notna(data.get("DilutedEPS"))
                        else "N/A"
                    ),
                }
            return summary

        ticker = yf.Ticker(ticker)
        historical_prices = ticker.history(period="5d", interval="1d")

        last_5_days_close = historical_prices["Close"].tail(5)
        last_5_days_close_dict = {
            date.strftime("%Y-%m-%d"): price
            for date, price in last_5_days_close.items()
        }

        # Retrieve annual and quarterly financial statement data
        annual_financials = ticker.get_financials()
        quarterly_financials = ticker.get_financials(freq="quarterly")

        return str(
            {
                "최근 5일간 종가": last_5_days_close_dict,
                "연간 재무제표 요약": format_financial_summary(annual_financials),
                "분기별 재무제표 요약": format_financial_summary(quarterly_financials),
            }
        )

    def create_stock_chart(self, ticker: str, days: int = 30) -> str:
        """Create a stock chart using yfinance and plotly."""
        # Get stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.download(ticker, start=start_date, end=end_date)

        # Create candlestick chart
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=stock.index,
                    open=stock["Open"],
                    high=stock["High"],
                    low=stock["Low"],
                    close=stock["Close"],
                )
            ]
        )

        # Update layout
        fig.update_layout(
            title=f"{ticker} Stock Price",
            yaxis_title="Stock Price (USD)",
            xaxis_title="Date",
        )

        # Convert to JSON for storage
        chart_data = fig.to_json()

        return f"![Chart]\n```json\n{chart_data}\n```"

    @property
    def python_repl_tool(self):
        """Get Python REPL tool with stock charting capabilities."""
        return PythonREPLTool(
            globals={
                "yf": yf,
                "go": go,
                "create_stock_chart": self.create_stock_chart,
                "pd": pd,
                "datetime": datetime,
                "timedelta": timedelta,
            }
        )
