import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
import numpy as np
import streamlit as st

START = "2023-01-01"
END   = "2023-12-31"

st.title("主流股自動篩選 + 多重條件組合最佳停損/停利搜尋")

# 自動抓取主流股（近一月成交量排行前20名）
def get_hot_stocks():
    stocks = ["2330.TW", "2317.TW", "2303.TW", "2412.TW", "2454.TW", "2882.TW", "2881.TW", "2603.TW", "2308.TW", "1301.TW", "1303.TW", "2002.TW", "2886.TW", "2885.TW", "2891.TW", "2884.TW", "2880.TW", "2883.TW", "2887.TW", "2888.TW"]
    vol_dict = {}
    for stock in stocks:
        df = yf.download(stock, start="2023-08-01", end="2023-09-01")
        if df is not None and not df.empty and "Volume" in df.columns:
            mean_vol = df["Volume"].mean()
            if isinstance(mean_vol, pd.Series):
                mean_vol = mean_vol.iloc[0]
            vol_dict[stock] = float(mean_vol)
    # 只排序有資料的股票
    hot_stocks = sorted(vol_dict, key=vol_dict.get, reverse=True)[:20]
    return hot_stocks

hot_stocks = get_hot_stocks()
st.write(f"自動篩選主流股（近一月成交量排行前20名）：{hot_stocks}")

stock_input = st.text_area("可自行輸入股票編號（用逗號或空白分隔），留空則用主流股自動篩選")
run = st.button("開始搜尋最佳參數")

# 搜尋範圍（ATR 倍數）
sl_space = [1.0, 1.5, 2.0]     # 停損 = ATR * N
tp_space = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
fixed_tp_space = [0.05, 0.08, 0.10]  # 固定停利5%、8%、10%

def calc_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def generate_signals(df):
    df["EMA23"] = df["Close"].ewm(span=23, adjust=False).mean()
    df["EMA60"] = df["Close"].ewm(span=60, adjust=False).mean()
    df["多頭排列"] = (df["EMA23"] > df["EMA60"]).astype(int)
    df["VOL5"] = df["Volume"].rolling(5).mean()
    df["爆量"] = (df["Volume"] > 2 * df["VOL5"]).astype(int)
    df["漲幅"] = (df["Close"] / df["Close"].shift(1) - 1) * 100
    df["漲幅大於2"] = (df["漲幅"] >= 2).astype(int)
    return df

class MultiConditionStrategy(Strategy):
    stop_mult = 1.0
    tp_mult = 2.0
    fixed_tp = 0.05

    def init(self):
        # 初始化指標
        self.ema23 = self.data.EMA23
        self.ema60 = self.data.EMA60
        self.atr = self.data.ATR
        self.vol5 = self.data.VOL5
        self.boom = self.data.爆量
        self.up2 = self.data.漲幅大於2

    def next(self):
        price = self.data.Close[-1]
        ema23 = self.ema23[-1]
        ema60 = self.ema60[-1]
        atr_val = self.atr[-1]
        boom = self.boom[-1]
        up2 = self.up2[-1]
        # 多重條件組合：均線多頭且（爆量或漲幅≥2%）
        if price > ema23 and ema23 > ema60 and (boom or up2) and not self.position and atr_val > 0:
            self.buy(
                sl=price - atr_val * self.stop_mult,
                tp=max(price + atr_val * self.tp_mult, price * (1 + self.fixed_tp))
            )
        # 可加其他出場條件

def run_best_param_search(stock):
    df = yf.download(stock, start=START, end=END)
    if df.empty or len(df) < 30:
        return None
    df = df.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    # 檢查欄位是否齊全
    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(df.columns)):
        return None
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df["ATR"] = calc_atr(df, period=14)
    df = generate_signals(df)

    best_sl, best_tp, best_return, best_stats = None, None, -float("inf"), None
    best_fixed_tp = None
    best_score = -float("inf")

    for sl in sl_space:
        for tp in tp_space:
            for fixed_tp in fixed_tp_space:
                MultiConditionStrategy.stop_mult = sl
                MultiConditionStrategy.tp_mult = tp
                MultiConditionStrategy.fixed_tp = fixed_tp
                bt = Backtest(df, MultiConditionStrategy, cash=1_000_000, commission=0.001,
                              exclusive_orders=True, finalize_trades=True)
                stats = bt.run()
                final_return = stats["Return [%]"]
                score = stats["Return [%]"] / abs(stats["Max. Drawdown [%]"]) if stats["Max. Drawdown [%]"] != 0 else 0
                if score > best_score:
                    best_score = score
                    best_return = final_return
                    best_sl = sl
                    best_tp = tp
                    best_fixed_tp = fixed_tp
                    best_stats = stats

    return {
        "股票": stock,
        "最佳停損倍數(ATR)": best_sl,
        "最佳停利倍數(ATR)": best_tp,
        "最佳固定停利(%)": f"{best_fixed_tp*100:.0f}",
        "報酬率(%)": f"{best_return:.2f}",
        "勝率(%)": f"{best_stats.get('Win Rate [%]', '-'):.2f}" if best_stats is not None else "-",
        "最大回撤(%)": f"{best_stats.get('Max. Drawdown [%]', '-'):.2f}" if best_stats is not None else "-",
        "交易次數": best_stats.get("# Trades", "-") if best_stats is not None else "-",
        "平均持有天數": best_stats.get("Avg. Holding Time", "-") if best_stats is not None else "-"
    }

if run:
    if stock_input.strip():
        STOCKS = sorted(set([s.strip().replace("'", "") for s in stock_input.replace(",", " ").split() if s.strip()]))
    else:
        STOCKS = hot_stocks
    st.write(f"本次搜尋股票：{STOCKS}")

    best_params = []
    progress = st.progress(0)
    for idx, stock in enumerate(STOCKS):
        st.write(f"📥 下載 {stock} 股價資料中...")
        result = run_best_param_search(stock)
        if result:
            best_params.append(result)
        else:
            st.warning(f"{stock} 沒有資料或回測失敗")
        progress.progress((idx + 1) / len(STOCKS))

    st.subheader("最佳停損/停利參數總覽")
    st.dataframe(pd.DataFrame(best_params), use_container_width=True)
