import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
import plotly.graph_objects as go
import numpy as np
import streamlit as st

# === 全域參數 ===
START = "2024-06-01"
END   = pd.Timestamp.today().strftime("%Y-%m-%d")
CONSECUTIVE = 2

# === 技術指標 & 訊號產生 ===
def generate_signals(df):
    close = df["Close"].squeeze()
    ema23 = close.ewm(span=23, adjust=False).mean()
    df["EMA23"] = ema23
    df["Signal"] = (close > ema23).astype(int)
    df["ConsecutiveBuy"] = df["Signal"].rolling(CONSECUTIVE).sum()
    df["FinalSignal"] = (df["ConsecutiveBuy"] == CONSECUTIVE).astype(int)
    return df

# === 策略 ===
class EMA23Strategy(Strategy):
    n1 = 23
    stop_loss = 0.01
    take_profit = 0.03  

    def init(self):
        price = self.data.Close
        self.ema23 = self.I(
            lambda x: pd.Series(x).ewm(span=self.n1, adjust=False).mean(),
            price
        )

    def next(self):
        price = self.data.Close[-1]
        ema = self.ema23[-1]
        if price > ema and not self.position:
            self.buy(
                sl=price * (1 - self.stop_loss),
                tp=price * (1 + self.take_profit)
            )
        elif price < ema and self.position.is_long:
            self.position.close()

# === 畫圖函式 ===
def plot_signals(df, trades, stock, best_sl):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="收盤價"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA23"], mode="lines", name="EMA23"))

    buy_trades = trades[trades["Size"] > 0]
    fig.add_trace(go.Scatter(
        x=buy_trades["EntryTime"], y=buy_trades["EntryPrice"],
        mode="markers", marker=dict(color="green", symbol="arrow-up", size=12),
        name="買進"
    ))

    sell_trades = trades[trades["Size"] < 0]
    fig.add_trace(go.Scatter(
        x=sell_trades["ExitTime"], y=sell_trades["ExitPrice"],
        mode="markers", marker=dict(color="red", symbol="arrow-down", size=12),
        name="賣出"
    ))

    fig.update_layout(
        title=f"{stock} 最佳停損: {best_sl:.2%}",
        xaxis_title="日期",
        yaxis_title="價格"
    )
    return fig

# === 讀取台股代碼與中文名稱 ===
twse_df = pd.read_csv("twse_list.csv", dtype=str)
code2name = dict(zip(twse_df["code"], twse_df["name"]))

# === Streamlit 介面 ===
st.title("📈 EMA23 最佳停損/停利搜尋器")

search_space = np.arange(0.01, 0.051, 0.005)
take_profit_space = np.arange(0.02, 0.061, 0.01)
best_params = []
results = []

stock_input = st.text_area(
    "請輸入股票編號（逗號或空白分隔，例如 2330.TW,2317.TW,2303.TW）："
)
run = st.button("開始回測")

if run and stock_input.strip():
    STOCKS = [s.strip() for s in stock_input.replace(",", " ").split() if s.strip()]
    for stock in STOCKS:
        stock_code = stock.split(".")[0]  # 例如 2330.TW -> 2330
        stock_name = code2name.get(stock_code, "")
        display_name = f"{stock} {stock_name}" if stock_name else stock

        df = yf.download(stock, start=START, end=END)
        if df.empty or len(df) < 30:
            st.warning(f"{stock} 沒有足夠資料")
            continue

        # 處理欄位格式
        if isinstance(df.columns, pd.MultiIndex):
            # 只取第一層欄位名稱
            df.columns = df.columns.get_level_values(0)
        # 檢查是否有正確欄位
        expected_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not expected_cols.issubset(set(df.columns)):
            # 嘗試自動對應欄位
            col_map = {}
            for i, name in enumerate(["Open", "High", "Low", "Close", "Volume"]):
                if i < len(df.columns):
                    col_map[df.columns[i]] = name
            df = df.rename(columns=col_map)
        if not expected_cols.issubset(set(df.columns)):
            st.warning(f"{stock} 缺少必要欄位，跳過")
            continue

        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.set_index("Date")
        df = generate_signals(df)

        best_sl, best_tp, best_return = None, None, -float("inf")
        best_stats = None

        # === 搜尋最佳停損/停利 ===
        for sl in search_space:
            for tp in take_profit_space:
                EMA23Strategy.stop_loss = sl
                EMA23Strategy.take_profit = tp
                bt = Backtest(
                    df, EMA23Strategy,
                    cash=1_000_000,
                    commission=0.001,
                    exclusive_orders=True,
                    finalize_trades=True
                )
                stats = bt.run()
                final_return = stats["Return [%]"]

                if final_return > best_return:
                    best_return = final_return
                    best_sl, best_tp = sl, tp
                    best_stats = stats

        # === 顯示結果 ===
        st.success(
            f"⭐ {display_name} 最佳停損參數：{best_sl:.3%}，最佳停利參數：{best_tp:.3%}，報酬率：{best_return:.2f}%"
        )

        trades = best_stats._trades
        trades["股票"] = display_name
        results.append(trades)

        # 顯示最後一筆買進建議
        last_buy = trades[trades["Size"] > 0].iloc[-1] if not trades[trades["Size"] > 0].empty else None
        if last_buy is not None:
            buy_price = last_buy["EntryPrice"]
            sl_price = buy_price * (1 - best_sl)
            tp_price = buy_price * (1 + best_tp)
            buy_info = (
                f"- 買進日期：{last_buy['EntryTime'].date()}  \n"
                f"- 買進價：{buy_price:.2f}  \n"
                f"- 停損價：{sl_price:.2f}  \n"
                f"- 停利價：{tp_price:.2f}"
            )
        else:
            buy_info = "近期無買進訊號，請留意均線突破再操作。"

        best_params.append(
            {
                "股票": display_name,
                "最佳停損": best_sl,
                "最佳停利": best_tp,
                "報酬率(%)": best_return,
                "勝率(%)": best_stats.get("Win Rate [%]", None),
                "平均持有天數": best_stats.get("Avg. Trade Duration", None),
                "建議買點": buy_info
            }
        )

        fig = plot_signals(df.reset_index(), trades, display_name, best_sl)
        st.plotly_chart(fig, use_container_width=True, key=stock)

    # 總覽
    st.subheader("📊 最佳停損/停利參數總覽")
    df_summary = pd.DataFrame(best_params)
    st.dataframe(
        df_summary[["股票", "最佳停損", "最佳停利", "報酬率(%)", "勝率(%)", "平均持有天數"]],
        use_container_width=True
    )

    st.markdown("### 各股票最新建議買點")
    for idx, row in df_summary.iterrows():
        avg_days = row['平均持有天數']
        if pd.notnull(avg_days):
            if hasattr(avg_days, "days"):
                avg_days = avg_days.days + avg_days.seconds / 86400
            else:
                avg_days = float(avg_days)
            avg_days_str = f"{avg_days:.2f} 天"
        else:
            avg_days_str = "無"
        st.info(
            f"【{row['股票']} 最新建議買點】\n"
            f"{row['建議買點']}\n"
            f"- 勝率：{row['勝率(%)']:.2f}%\n"
            f"- 平均持有天數：{avg_days_str}"
        )

st.markdown("""
**賣出條件詳細說明：**
- 當持有部位時，若收盤價低於 EMA23，系統會立即賣出（平倉）。
- 若股價下跌至停損參數（如 -5%），自動賣出以控制損失。
- 若股價上漲至停利參數（如 +6%），自動賣出以鎖定獲利。
""")
