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

st.set_page_config(page_title="LMI Dashboard", page_icon="📊", layout="wide")
st.title("📊 LMI Regime Momentum Dashboard")
st.caption(f"Last updated: {datetime.today().strftime('%Y-%m-%d %H:%M KST')}")
STOCK_TICKERS = ["XLK","XLF","XLV","XLY","XLP","XLI","XLE","XLB","XLU","XLC"]
BOND_TICKERS  = ["TLT","IEF","SHY","TIP","LQD","HYG"]

HOLDINGS_FILE = "portfolio_holdings.csv"

# ══════════════════════════════════════════════
# 포트폴리오 Holdings 로드/편집 (사이드바)
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("📂 포트폴리오 관리")
    st.caption("리밸런싱 후 수량/평균단가를 업데이트하세요")

    # CSV 파일 있으면 로드, 없으면 빈 템플릿
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

    # 편집 가능한 테이블
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

    col_save, col_reset = st.columns(2)
    if col_save.button("💾 저장", use_container_width=True):
        edited_df.to_csv(HOLDINGS_FILE, index=False)
        st.success("저장 완료!")
        st.rerun()

    # 마지막 리밸런싱 날짜 기록
    st.divider()
    rebal_date = st.date_input(
        "📅 최근 리밸런싱 날짜",
        value=datetime.today()
    )


# ══════════════════════════════════════════════
# 실시간 가격 조회 (yfinance, 5분 캐시)
# ══════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_live_prices(tickers: list[str]) -> dict:
    data = yf.download(tickers, period="1d", interval="1m", progress=False)
    if isinstance(data, pd.DataFrame) and "Close" in data:
        latest = data["Close"].iloc[-1]
        return {t: round(float(latest.get(t, np.nan)), 2) for t in tickers}
    return {t: None for t in tickers}

live_prices = get_live_prices(tuple(STOCK_TICKERS + BOND_TICKERS))
# nan 제거 또는 0으로 대체
live_prices = {k: v for k, v in live_prices.items() if v == v}  # nan 제거


# ══════════════════════════════════════════════
# 전략 계산 (30분 캐시)
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

    oos_start       = (pd.to_datetime(SPLIT_DATE) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
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
        "split_date": SPLIT_DATE,
        "oos_start": oos_start,
    }

# ══════════════════════════════════════════════
# DV01 (하루 1번 캐시)
# ══════════════════════════════════════════════
@st.cache_data(ttl=86400)
def get_dv01(bond_tickers: tuple):
    try:
        dv01calc  = DV01Calculator(list(bond_tickers))
        durations = dv01calc.get_modified_durations()
        prices_dv = dv01calc.get_latest_prices()
        return dv01calc.compute_dv01(durations, prices_dv)
    except Exception:
        return pd.DataFrame(
            index=list(bond_tickers),
            columns=["Modified Duration", "Price", "DV01"]
        )

# ══════════════════════════════════════════════
# 포트폴리오 평가 계산
# ══════════════════════════════════════════════
def compute_portfolio(holdings, prices):
    df = holdings.copy()
    df["current_price_usd"]  = df["ticker"].map(prices)
    df["market_value_usd"]   = df["shares"] * df["current_price_usd"].fillna(0)  # ← fillna 추가
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
# 탭
# ══════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🏠 운용현황", "📈 벤치마크 트래킹", "🧩 ETF 유니버스"])

# ── TAB 1 ──────────────────────────────────────
with tab1:
    regime = strat["current_regime"]
    c1, c2, c3, c4 = st.columns(4)   # 5→4로 줄이기
    c1.metric("현재 레짐", regime)
    c2.metric("총 평가금액 (USD)", f"${total_usd:,.0f}")
    c3.metric("미실현 손익 (USD)", f"${total_pnl:+,.0f}")
    c4.metric("리밸런싱", str(rebal_date))
    st.caption(f"📅 최근 리밸런싱: {rebal_date}  |  가격 기준: 실시간 (5분 캐시)")
    st.divider()

    st.subheader("📋 현재 포트폴리오 상세")
    disp = portfolio[[
        "asset_class","shares","avg_cost_usd","current_price_usd",
        "market_value_usd",
        "unrealized_pnl_usd","unrealized_pnl_pct","actual_weight_pct"
    ]].copy()
    disp.columns = ["구분","수량","평균단가","현재가","평가금액(USD)",
                    "미실현손익(USD)","손익률(%)","실제비중(%)"]
    disp = disp[disp["수량"] > 0]
    st.dataframe(
        disp.style.format({
            "수량":"{:.2f}", "평균단가":"${:.2f}", "현재가":"${:.2f}",
            "평가금액(USD)":"${:,.0f}", 
            "미실현손익(USD)":"${:+,.0f}", "손익률(%)":"{:+.2f}%", "실제비중(%)":"{:.1f}%"
        }),
        use_container_width=True, height=450
    )
    st.divider()

    st.subheader("🎯 실제 비중 vs 전략 목표 비중")
    target_sw  = strat["stock_weights"].drop(columns=["stock_weight_change"]).iloc[-1] * 100
    target_bw  = strat["bond_weights"].drop(columns=["bond_weight_change"]).iloc[-1] * 100
    target_all = pd.concat([target_sw, target_bw]).rename("목표비중(%)")
    actual_w   = portfolio["actual_weight_pct"].rename("실제비중(%)")
    compare_df = pd.concat([actual_w, target_all], axis=1).fillna(0)
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
    st.subheader("📈 누적수익률: 전략 vs SPY/AGG 60/40")
    in_s  = port.loc[:strat["split_date"], "cumulative_return"]
    out_s = port.loc[strat["oos_start"]:,  "cumulative_return"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=in_s.index,  y=in_s*100,  name="전략 (In-Sample)",     line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=out_s.index, y=out_s*100, name="전략 (Out-of-Sample)", line=dict(color="#1f77b4", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=bm.index,    y=bm*100,    name="SPY/AGG 60/40",        line=dict(color="#ff7f0e", width=2)))
    fig.add_vline(x=pd.Timestamp(strat["split_date"]).timestamp() * 1000,
    line_dash="dot", line_color="gray", annotation_text="OOS 시작")
    fig.update_layout(yaxis_title="누적수익률 (%)", height=500, hovermode="x unified",
                      legend=dict(orientation="h", y=1.02))
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
        perf_row(port.loc[:strat["split_date"], "strategy_return"].dropna(), "전략 In-Sample"),
        perf_row(port.loc[strat["oos_start"]:,  "strategy_return"].dropna(), "전략 Out-of-Sample"),
        perf_row(bench_ret.loc[:strat["split_date"]].dropna(), "60/40 In-Sample"),
        perf_row(bench_ret.loc[strat["oos_start"]:].dropna(),  "60/40 Out-of-Sample"),
    ]
    st.dataframe(pd.DataFrame(rows).set_index(""), use_container_width=True)

# ── TAB 3 ──────────────────────────────────────
with tab3:
    st.subheader("🧩 투자 유니버스 ETF 현황")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 📦 주식 ETF")
        sw_latest = strat["stock_weights"].drop(columns=["stock_weight_change"]).iloc[-1]
        rows = [{"Ticker": t, "현재가": live_prices.get(t), "목표비중(%)": round(sw_latest.get(t,0)*100, 2)}
                for t in STOCK_TICKERS]
        st.dataframe(
            pd.DataFrame(rows).set_index("Ticker")
              .style.format({"현재가": "${:.2f}", "목표비중(%)": "{:.2f}%"}),
            use_container_width=True
        )

    with c2:
        st.markdown("#### 🏦 채권 ETF + DV01")
        bw_latest = strat["bond_weights"].drop(columns=["bond_weight_change"]).iloc[-1]
        rows = []
        for t in BOND_TICKERS:
            rows.append({
                "Ticker":         t,
                "현재가":          live_prices.get(t),
                "목표비중(%)":     round(bw_latest.get(t, 0)*100, 2),
                "Mod. Duration":  dv01_result.loc[t, "Modified Duration"] if t in dv01_result.index else None,
                "DV01":           dv01_result.loc[t, "DV01"] if t in dv01_result.index else None,
            })
        st.dataframe(
            pd.DataFrame(rows).set_index("Ticker")
              .style.format({"현재가":"${:.2f}", "목표비중(%)":"{:.2f}%",
                             "Mod. Duration":"{:.2f}", "DV01":"{:.4f}"}),
            use_container_width=True
        )

        