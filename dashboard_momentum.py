import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import sys
sys.path.append(r"C:\Users\khm39\vscode\MFEprojects")
from LMI_regime_momentum_strategy import (
    DataFetcher, RegimeClassifier, MomentumCalculator,
    WeightAllocator, PortfolioSimulator, DV01Calculator
)

st.set_page_config(page_title="LMI Dashboard", layout="wide")
st.title("LMI Regime Momentum Dashboard")
st.caption(f"Last updated: {datetime.today().strftime('%Y-%m-%d %H:%M KST')}")
STOCK_TICKERS = ["XLK","XLF","XLV","XLY","XLP","XLI","XLE","XLB","XLU","XLC"]
BOND_TICKERS  = ["TLT","IEF","SHY","TIP","LQD","HYG"]
HOLDINGS_FILE = "portfolio_holdings.csv"

# ══════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("포트폴리오 관리")
    st.caption("리밸런싱 후 수량/평균단가를 업데이트하세요")

    try:
        holdings_df = pd.read_csv(HOLDINGS_FILE)
    except FileNotFoundError:
        holdings_df = pd.DataFrame({
            "ticker":       ["XLK","XLF","XLV","XLY","XLP","XLI","XLE","XLB","XLU","XLC",
                             "TLT","IEF","SHY","TIP","LQD","HYG"],
            "shares":       [0.0] * 16,
            "avg_cost_usd": [0.0] * 16,
            "asset_class":  ["stock"]*10 + ["bond"]*6
        })

    st.markdown("**보유 수량 & 평균단가 입력**")
    edited_df = st.data_editor(
        holdings_df,
        column_config={
            "ticker":       st.column_config.TextColumn("Ticker", disabled=True),
            "shares":       st.column_config.NumberColumn("수량", min_value=0.0, step=0.01, format="%.2f"),
            "avg_cost_usd": st.column_config.NumberColumn("평균단가(USD)", min_value=0.0, step=0.01, format="%.2f"),
            "asset_class":  st.column_config.SelectboxColumn("구분", options=["stock","bond"], disabled=True),
        },
        hide_index=True,
        use_container_width=True,
    )

    col_save, _ = st.columns(2)
    if col_save.button("💾 저장", use_container_width=True):
        edited_df.to_csv(HOLDINGS_FILE, index=False)
        st.success("저장 완료!")
        st.rerun()

    st.divider()

    # ── 환율 입력 ──────────────────────────────
    st.markdown("**환율 설정**")
    krw_rate = st.number_input(
        "USD/KRW 환율",
        min_value=900.0, max_value=2000.0,
        value=1350.0, step=1.0, format="%.1f"
    )

    st.divider()

    # ── 리밸런싱 날짜 ──────────────────────────
    rebal_date = st.date_input(
        "📅 리밸런싱 기준 날짜",
        value=datetime.today()
    )


# ══════════════════════════════════════════════
# 실시간 가격
# ══════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_live_prices(tickers: tuple) -> dict:
    result = {}
    for t in tickers:
        try:
            df = yf.download(t, period="5d", interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                result[t] = round(float(df["Close"].dropna().iloc[-1]), 2)
            else:
                result[t] = None
        except Exception:
            result[t] = None
    return result

live_prices = get_live_prices(tuple(STOCK_TICKERS + BOND_TICKERS))
live_prices = {k: v for k, v in live_prices.items() if v is not None and v == v}


# ══════════════════════════════════════════════
# 전략 계산
# ══════════════════════════════════════════════
@st.cache_data(ttl=1800)
def run_strategy():
    START_DATE = "1992-01-01"
    END_DATE   = datetime.today().strftime("%Y-%m-%d")
    TEST_START = "2008-01-01"
    SPLIT_DATE = "2020-01-01"

    fetcher   = DataFetcher(START_DATE, END_DATE)
    cli_df, vix_monthly, lagging = fetcher.fetch_cli_vix()
    clivix_df = RegimeClassifier(cli_df, vix_monthly).classify()

    moment_start = pd.to_datetime(TEST_START) + pd.DateOffset(months=12)
    stock_data   = fetcher.fetch_etf_data(STOCK_TICKERS, TEST_START, END_DATE)
    bond_data    = fetcher.fetch_etf_data(BOND_TICKERS,  TEST_START, END_DATE)

    stock_momentum = MomentumCalculator.compute_momentum_score(stock_data, lagging).resample("MS").last().loc[moment_start:]
    bond_momentum  = MomentumCalculator.compute_momentum_score(bond_data,  lagging).resample("MS").last().loc[moment_start:]
    stock_momentum[stock_momentum < 0] = 0
    bond_momentum[bond_momentum   < 0] = 0

    stock_weights = WeightAllocator.get_weights(stock_momentum)
    bond_weights  = WeightAllocator.get_weights(bond_momentum)
    stock_weights["stock_weight_change"] = stock_weights.diff().abs().sum(axis=1)
    bond_weights["bond_weight_change"]   = bond_weights.diff().abs().sum(axis=1)

    port = PortfolioSimulator(
        STOCK_TICKERS, BOND_TICKERS, stock_data, bond_data,
        stock_weights, bond_weights, clivix_df, transaction_cost=0.001
    ).simulate()

    tbill_df = fetcher.fetch_tbill(port.index)
    port = port.drop(columns=["risk_free_rate"], errors="ignore").merge(tbill_df, left_index=True, right_index=True)

    in_sample_df    = port.loc[:SPLIT_DATE]
    optimal_weights = PortfolioSimulator.optimize_by_regime(in_sample_df)
    port            = port.join(optimal_weights.set_index("regime"), on="regime")

    TC = 0.001
    port["strategy_return"] = (
        port["stock_ret"] * port["stock_weight"]
        + port["bond_ret"] * port["bond_weight"]
        - port["stock_weight_change"] * port["stock_weight"] * TC
        - port["bond_weight_change"]  * port["bond_weight"]  * TC
    )
    port["cumulative_return"] = (1 + port["strategy_return"]).cumprod() - 1

    bench_px = fetcher.fetch_etf_data(["SPY","AGG"], moment_start.strftime("%Y-%m-%d"), END_DATE)
    if isinstance(bench_px.columns, pd.MultiIndex):
        bench_px = bench_px["Adj Close"] if "Adj Close" in bench_px.columns.get_level_values(0) else bench_px["Close"]
    bench_px.columns  = [str(c) for c in bench_px.columns]
    bench_mret        = bench_px[["SPY","AGG"]].dropna().resample("MS").last().pct_change().dropna()
    benchmark         = (1 + 0.60*bench_mret["SPY"] + 0.40*bench_mret["AGG"]).cumprod() - 1
    benchmark         = benchmark.reindex(port.index).ffill()

    current_regime = port.loc[port.index[-1], "regime"]

    return {
        "port": port,
        "benchmark": benchmark,
        "optimal_weights": optimal_weights,
        "stock_weights": stock_weights,
        "bond_weights": bond_weights,
        "current_regime": current_regime,
    }


# ══════════════════════════════════════════════
# DV01
# ══════════════════════════════════════════════
@st.cache_data(ttl=86400)
def get_dv01(bond_tickers: tuple):
    try:
        dv01calc  = DV01Calculator(list(bond_tickers))
        durations = dv01calc.get_modified_durations()
        prices_dv = dv01calc.get_latest_prices()
        return dv01calc.compute_dv01(durations, prices_dv)
    except Exception:
        return pd.DataFrame(index=list(bond_tickers), columns=["Modified Duration", "Price", "DV01"])


# ══════════════════════════════════════════════
# 포트폴리오 평가
# ══════════════════════════════════════════════
def compute_portfolio(holdings, prices):
    df = holdings.copy()
    df["current_price_usd"]  = df["ticker"].map(prices)
    df["market_value_usd"]   = df["shares"] * df["current_price_usd"].fillna(0)
    df["cost_basis_usd"]     = df["shares"] * df["avg_cost_usd"]
    df["unrealized_pnl_usd"] = df["market_value_usd"] - df["cost_basis_usd"]
    df["unrealized_pnl_pct"] = np.where(
        df["cost_basis_usd"] > 0,
        df["unrealized_pnl_usd"] / df["cost_basis_usd"] * 100, 0.0
    )
    total = df["market_value_usd"].sum()
    df["actual_weight_pct"] = np.where(total > 0, df["market_value_usd"] / total * 100, 0.0)
    return df.set_index("ticker")


# ══════════════════════════════════════════════
# 리밸런싱 날짜 기준 레짐 & 비중 조회
# ══════════════════════════════════════════════
def get_regime_at_date(port_df, target_date):
    """선택 날짜 이전 가장 최근 월의 레짐 반환"""
    ts = pd.Timestamp(target_date)
    past = port_df[port_df.index <= ts]
    if past.empty:
        return "N/A"
    return past.iloc[-1]["regime"]

def get_weights_at_date(stock_weights, bond_weights, target_date):
    """선택 날짜 이전 가장 최근 월의 목표비중 반환"""
    ts = pd.Timestamp(target_date)
    sw = stock_weights.drop(columns=["stock_weight_change"])
    bw = bond_weights.drop(columns=["bond_weight_change"])
    sw_past = sw[sw.index <= ts]
    bw_past = bw[bw.index <= ts]
    sw_row = sw_past.iloc[-1] * 100 if not sw_past.empty else pd.Series(dtype=float)
    bw_row = bw_past.iloc[-1] * 100 if not bw_past.empty else pd.Series(dtype=float)
    return pd.concat([sw_row, bw_row]).rename("목표비중(%)")


# ══════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════
try:
    strat = run_strategy()
except Exception as e:
    st.error(f"전략 계산 오류: {e}")
    st.stop()

dv01_result = get_dv01(tuple(BOND_TICKERS))
port        = strat["port"]
bm          = strat["benchmark"]

portfolio  = compute_portfolio(edited_df, live_prices)
total_usd  = portfolio["market_value_usd"].sum()
total_krw  = total_usd * krw_rate
total_pnl  = portfolio["unrealized_pnl_usd"].sum()
total_pnl_krw = total_pnl * krw_rate

# 리밸런싱 날짜 기준 레짐 & 비중
rebal_regime  = get_regime_at_date(port, rebal_date)
rebal_weights = get_weights_at_date(strat["stock_weights"], strat["bond_weights"], rebal_date)


# ══════════════════════════════════════════════
# 탭
# ══════════════════════════════════════════════
tab1, tab2 = st.tabs(["운용현황", "벤치마크 트래킹"])

# ── TAB 1 ──────────────────────────────────────
with tab1:
    # 상단 메트릭 (현재 기준)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재 레짐", strat["current_regime"])
    c2.metric("총 평가금액 (USD)", f"${total_usd:,.0f}")
    c2.caption(f"₩{total_krw:,.0f}  (환율 {krw_rate:,.1f})")
    c3.metric("미실현 손익 (USD)", f"${total_pnl:+,.0f}")
    c3.caption(f"₩{total_pnl_krw:+,.0f}")
    c4.metric("리밸런싱 기준일", str(rebal_date))
    c4.caption(f"당시 레짐: {rebal_regime}")

    st.caption(f"가격 기준: 실시간 (5분 캐시)  |  환율: {krw_rate:,.1f} KRW/USD")
    st.divider()

    # 포트폴리오 상세 (USD + KRW 병기)
    st.subheader("현재 포트폴리오 상세")
    disp = portfolio[[
        "asset_class","shares","avg_cost_usd","current_price_usd",
        "market_value_usd","unrealized_pnl_usd","unrealized_pnl_pct","actual_weight_pct"
    ]].copy()
    disp["market_value_krw"]   = disp["market_value_usd"]   * krw_rate
    disp["unrealized_pnl_krw"] = disp["unrealized_pnl_usd"] * krw_rate
    disp.columns = ["구분","수량","평균단가","현재가",
                    "평가금액(USD)","미실현손익(USD)","손익률(%)",
                    "실제비중(%)","평가금액(KRW)","미실현손익(KRW)"]
    disp = disp[disp["수량"] > 0]
    st.dataframe(
        disp.style.format({
            "수량":          "{:.2f}",
            "평균단가":      "${:.2f}",
            "현재가":        "${:.2f}",
            "평가금액(USD)": "${:,.0f}",
            "미실현손익(USD)":"${:+,.0f}",
            "손익률(%)":     "{:+.2f}%",
            "실제비중(%)":   "{:.1f}%",
            "평가금액(KRW)": "₩{:,.0f}",
            "미실현손익(KRW)":"₩{:+,.0f}",
        }),
        use_container_width=True, height=450
    )
    st.divider()

    # 비중 비교 (리밸런싱 날짜 기준 목표비중)
    st.subheader(f"실제 비중 vs 목표 비중 ({rebal_date} 기준, 레짐: {rebal_regime})")
    actual_w   = portfolio["actual_weight_pct"].rename("실제비중(%)")
    compare_df = pd.concat([actual_w, rebal_weights], axis=1).fillna(0)
    compare_df["괴리(%)"] = (compare_df["실제비중(%)"] - compare_df["목표비중(%)"]).round(2)
    compare_df = compare_df[compare_df[["실제비중(%)","목표비중(%)"]].sum(axis=1) > 0]

    fig_w = go.Figure()
    fig_w.add_bar(x=compare_df.index, y=compare_df["실제비중(%)"], name="실제비중", marker_color="#3498db")
    fig_w.add_bar(x=compare_df.index, y=compare_df["목표비중(%)"],  name="목표비중", marker_color="#e67e22", opacity=0.7)
    fig_w.update_layout(barmode="group", height=350, yaxis_title="비중 (%)", hovermode="x unified")
    st.plotly_chart(fig_w, use_container_width=True)
    st.dataframe(compare_df.style.format("{:.2f}%"), use_container_width=True)


# ── TAB 2 ──────────────────────────────────────
with tab2:
    st.subheader("누적수익률: 전략 vs SPY/AGG 60/40")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port.index, y=port["cumulative_return"]*100,
        name="LMI 전략", line=dict(color="#1f77b4", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=bm.index, y=bm*100,
        name="SPY/AGG 60/40", line=dict(color="#ff7f0e", width=2)
    ))
    fig.update_layout(
        yaxis_title="누적수익률 (%)", height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    def perf_row(series, label):
        years = len(series) / 12
        cagr  = (1 + series).prod() ** (1/years) - 1
        vol   = series.std() * np.sqrt(12)
        sr    = (series.mean() * 12) / vol
        mdd   = ((1+series).cumprod() / (1+series).cumprod().cummax() - 1).min()
        return {"": label, "CAGR": f"{cagr*100:.2f}%", "연변동성": f"{vol*100:.2f}%",
                "Sharpe": f"{sr:.2f}", "MDD": f"{mdd*100:.2f}%"}

    bench_ret = bm.diff().dropna()
    rows = [
        perf_row(port["strategy_return"].dropna(), "LMI 전략 (전체)"),
        perf_row(bench_ret.dropna(), "SPY/AGG 60/40 (전체)"),
    ]
    st.dataframe(pd.DataFrame(rows).set_index(""), use_container_width=True)