"""
app.py — MiF AI Investment Application  (Streamlit frontend)

Palette: Boardroom Blue Steel
  #0F1B2B  deep navy bg
  #1F3B5C  surface / panels
  #4F6F8F  mid blue / borders
  #A7B4C8  muted text
  #F4F6FA  primary text

Run with:
    python -m streamlit run app.py
"""

import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents import DataAgent, StrategyAgent, RiskAgent, ExecutionAgent
from database import (
    init_db, migrate_db, reset_db, get_sim_state, get_positions,
    get_transactions, get_snapshots, save_snapshot, advance_date,
    get_target_alerts,
)
from portfolio_manager import (
    fetch_prices, get_market_snapshot, get_portfolio_valuation,
    compute_kpis, format_pct, format_dollar, STARTING_CAPITAL,
)

COMPETITION_START = "2026-03-13"
COMPETITION_END   = "2026-03-27"
BENCHMARK         = "SPY"
ASSETS = [
    "NVDA","AMD","ARM","AVGO","MRVL",
    "AAPL","MSFT","GOOGL","META","AMZN",
    "JPM","GS","V",
    "LLY","UNH",
    "XOM","CVX",
    "LMT","RTX",
    "PLTR","NET",
    "SPY","QQQ","GLD","TLT",
]

# ── Blue Steel Palette ────────────────────────────────────────────────────────
_BG      = "#0F1B2B"
_SURFACE = "#1F3B5C"
_CARD2   = "#2A4A6E"
_BORDER  = "#4F6F8F"
_FG      = "#F4F6FA"
_MUTED   = "#A7B4C8"
_ACCENT  = "#7BAFD4"
_GREEN   = "#5BC4A0"
_RED     = "#E07070"
_AMBER   = "#C8A84B"

st.set_page_config(page_title="Group B Portfolio", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{ background:{_BG}; color:{_FG}; }}
  [data-testid="stSidebar"]          {{ background:{_SURFACE}; border-right:1px solid {_BORDER}; }}
  [data-testid="stSidebar"] *        {{ color:{_FG} !important; }}
  [data-testid="metric-container"]   {{
    background:{_SURFACE}; border:1px solid {_BORDER};
    border-top:3px solid {_ACCENT}; border-radius:8px; padding:16px !important;
  }}
  [data-testid="stMetricLabel"]      {{ color:{_MUTED} !important; font-size:11px !important;
                                         letter-spacing:0.07em; text-transform:uppercase; font-weight:600 !important; }}
  [data-testid="stMetricValue"]      {{ color:{_FG} !important; font-size:26px !important; font-weight:700 !important; }}
  [data-testid="stTabs"] button      {{ background:{_SURFACE}; border:1px solid {_BORDER};
                                         color:{_MUTED}; border-radius:6px 6px 0 0; padding:8px 18px; font-weight:600; }}
  [data-testid="stTabs"] button[aria-selected="true"] {{ background:{_CARD2}; color:{_ACCENT}; border-bottom:2px solid {_ACCENT}; }}
  .stButton > button                 {{ border-radius:6px; font-weight:600; border:none; }}
  .stButton > button[kind="primary"] {{ background:{_ACCENT}; color:{_BG}; }}
  .stButton > button[kind="primary"]:hover {{ background:{_FG}; color:{_BG}; }}
  .stButton > button[kind="secondary"] {{ background:{_SURFACE}; color:{_FG}; border:1px solid {_BORDER}; }}
  .stButton > button[kind="secondary"]:hover {{ background:{_CARD2}; border-color:{_ACCENT}; }}
  .reasoning-box {{ background:{_SURFACE}; border:1px solid {_BORDER};
      border-left:4px solid {_ACCENT}; border-radius:0 8px 8px 0;
      padding:16px; font-family:'Courier New',monospace; font-size:12px;
      line-height:1.7; color:{_FG}; max-height:420px; overflow-y:auto; white-space:pre-wrap; }}
  .sim-card {{ background:{_CARD2}; border:1px solid {_ACCENT}; border-radius:8px; padding:14px 16px; margin-bottom:10px; }}
  h1,h2,h3 {{ color:{_FG} !important; }}
  hr        {{ border-color:{_BORDER}; }}
</style>
""", unsafe_allow_html=True)

for _k, _v in [("data_summary",""),("strategy_text",""),("risk_report",""),
               ("exec_report",""),("proposed_trades",[]),("approved_trades",[]),
               ("enriched_snap",{})]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _layout(height=340, yprefix="", ysuffix="", yfmt=""):
    return dict(
        plot_bgcolor=_BG, paper_bgcolor=_BG, font_color=_FG,
        height=height, margin=dict(l=0, r=10, t=10, b=0),
        hovermode="x unified",
        xaxis=dict(gridcolor=_CARD2, zeroline=False,
                   tickformat="%b %d", type="date", dtick="D1",
                   tickcolor=_MUTED, tickfont=dict(color=_MUTED, size=11)),
        yaxis=dict(gridcolor=_CARD2, zeroline=False,
                   tickprefix=yprefix, ticksuffix=ysuffix, tickformat=yfmt,
                   automargin=True, tickcolor=_MUTED, tickfont=dict(color=_MUTED, size=11)),
        legend=dict(bgcolor=_SURFACE, bordercolor=_BORDER, borderwidth=1,
                    font=dict(color=_MUTED, size=11)),
    )


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar(sim):
    with st.sidebar:
        st.markdown("# 🏆 Group B Portfolio")
        st.markdown("*MiF Deep Learning Competition · IE Business School*")
        st.divider()
        st.markdown(f"**Competition:** `{COMPETITION_START}` → `{COMPETITION_END}`")
        st.divider()
        st.markdown("### 🗓️ Simulation Date")
        if sim:
            st.markdown(
                f'<div style="background:{_ACCENT};color:{_BG};padding:10px 14px;'
                f'border-radius:6px;font-size:20px;font-weight:700;text-align:center;">'
                f'{sim["current_date"]}</div>', unsafe_allow_html=True)
            days_left = max(0, 11 - sim.get("day_number", 0))
            st.caption(f"Day {sim.get('day_number',0)} · {days_left} days left")
            if sim["current_date"] < date.today().isoformat():
                st.warning("⚠️ Sim date behind today. Hit **Adv. Day** to sync.", icon="📅")
        else:
            st.warning("No simulation active")
        st.divider()
        st.markdown("### ⚙️ Controls")
        if st.button("📅 Adv. Day", use_container_width=True, type="primary"):
            if sim:
                advance_date(sim["current_date"]); st.rerun()
        st.divider()

        # Start New Simulation — prominent card
        st.markdown(f"""
        <div class="sim-card">
          <div style="color:{_ACCENT};font-weight:700;font-size:13px;margin-bottom:5px;">🔄 START NEW SIMULATION</div>
          <div style="color:{_MUTED};font-size:11px;line-height:1.5;">
            Resets all trades, positions and cash to $1,000,000.<br>
            Use only at the start of a fresh run.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if "confirm_reset" not in st.session_state:
            st.session_state["confirm_reset"] = False
        if not st.session_state["confirm_reset"]:
            if st.button("🔄 Start New Simulation", use_container_width=True, type="secondary"):
                st.session_state["confirm_reset"] = True; st.rerun()
        else:
            st.warning("⚠️ This will erase ALL trades and positions.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes", use_container_width=True, type="primary"):
                    reset_db(STARTING_CAPITAL, COMPETITION_START)
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    for f in ["prices.csv","screener_results.csv"]:
                        if os.path.exists(f): os.remove(f)
                    st.success("Simulation reset!"); st.rerun()
            with c2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state["confirm_reset"] = False; st.rerun()
        st.divider()
        st.markdown("### 🛡️ Risk Limits")
        st.markdown("- Max position: **40%** (25% if ATR >3%)\n- Max trades/day: **2**\n- TC: **10 bps**")
        st.divider()



# ── TRADING FLOOR ─────────────────────────────────────────────────────────────
def render_trading_floor(sim, market_snap, prices_df=None, scores_df=None):
    price_history = {}
    if prices_df is not None:
        for ticker in prices_df.columns:
            s = prices_df[ticker].dropna()
            if len(s) >= 5: price_history[ticker] = s.tolist()

    st.markdown("## 🏛️ Trading Floor")
    st.markdown("Run the **four-agent pipeline** in order. Strategy Agent uses **three entry frameworks**: "
                "momentum plays (RSI 40–65), oversold bounces (RSI <35 + lower BB), and sector rotation.")
    st.divider()

    st.markdown("### 🔍 Stock Screener")
    st.caption("α = 0.40×momentum + 0.25×gap + 0.15×trend + 0.10×RSI + 0.10×volume.  Top 12 passed to Claude.")
    if scores_df is not None and not scores_df.empty:
        top20 = scores_df.head(20).copy(); top20.index = range(1, len(top20)+1)
        top20["Price"]       = top20["price"].apply(lambda x: f"${x:.2f}")
        top20["5d Return"]   = top20["ret_5d"].apply(lambda x: f"{x*100:+.1f}%" if x is not None else "—")
        top20["20d Return"]  = top20["ret_20d"].apply(lambda x: f"{x*100:+.1f}%" if x is not None else "—")
        top20["Ann. Vol"]    = top20["vol_20d"].apply(lambda x: f"{x*100:.0f}%" if x is not None else "—")
        top20["Alpha Score"] = top20["score"].apply(lambda x: f"{x*100:+.2f}%")
        st.dataframe(top20[["symbol","Price","5d Return","20d Return","Ann. Vol","Alpha Score"]].rename(
            columns={"symbol":"Ticker"}), use_container_width=True, hide_index=False)
        st.caption(f"Top 20 of {len(scores_df):,} stocks screened.")
    else:
        st.info("Screener results appear here after running `python run_daily.py`.")
    st.divider()

    st.markdown("### 1️⃣ Data Agent — Technical Indicators")
    st.caption("Computes RSI-14, SMA-5/20, Bollinger Bands, ATR-14.")
    if st.button("📊 Run Data Agent", type="secondary"):
        with st.spinner("Fetching prices & computing indicators..."):
            summary, enriched = DataAgent().run(market_snap, sim, price_history)
            st.session_state.data_summary  = summary
            st.session_state.enriched_snap = enriched
    if st.session_state.data_summary: st.markdown(st.session_state.data_summary)
    st.divider()

    st.markdown("### 2️⃣ Strategy Agent — AI Reasoning")
    st.caption("Claude searches the web then applies Momentum · Oversold Bounce · Sector Rotation.")
    if st.button("🤖 Run Strategy Agent", type="primary", disabled=not st.session_state.data_summary):
        with st.spinner("🔍 Searching web & reasoning… (20–40s)"):
            try:
                enriched = st.session_state.get("enriched_snap", market_snap)
                trades, text = StrategyAgent().run(enriched, sim, st.session_state.data_summary, ASSETS)
                st.session_state.strategy_text   = text
                st.session_state.proposed_trades = trades
                st.session_state.risk_report     = ""
                st.session_state.approved_trades = []
            except RuntimeError as e: st.error(str(e))
    if st.session_state.strategy_text:
        st.markdown("**🧠 Agent Reasoning:**")
        st.markdown(f'<div class="reasoning-box">{st.session_state.strategy_text.replace("<","&lt;").replace(">","&gt;")}</div>',
                    unsafe_allow_html=True)
        n = len(st.session_state.proposed_trades)
        (st.success if n else st.warning)(f"{'✅ '+str(n)+' trade(s) proposed' if n else 'No trades — holding recommended'}")
    st.divider()

    st.markdown("### 3️⃣ Risk Agent — Constraint Validation")
    st.caption("Max 40% per position (25% if ATR >3%) · max 2 trades/day · cash ≥ $0")
    if st.button("🛡️ Run Risk Agent",
                 disabled=not (st.session_state.proposed_trades or st.session_state.strategy_text)):
        approved, report = RiskAgent().run(st.session_state.proposed_trades, sim,
                                           st.session_state.get("enriched_snap", market_snap))
        st.session_state.approved_trades = approved
        st.session_state.risk_report     = report
    if st.session_state.risk_report: st.markdown(st.session_state.risk_report)
    st.divider()

    st.markdown("### 4️⃣ Execution Agent — Commit to Database")
    st.caption("Writes approved trades to SQLite. Debits/credits cash, logs every trade.")
    col_exec, col_hold = st.columns([2,1])
    with col_exec:
        if st.button("✅ Execute Approved Trades", type="primary",
                     disabled=not st.session_state.approved_trades):
            with st.spinner("Committing trades..."):
                n_exec, report = ExecutionAgent().run(st.session_state.approved_trades, sim, market_snap)
                prices = {t: m["price"] for t,m in market_snap.items() if m["price"]}
                total, holdings, cash = get_portfolio_valuation(prices)
                sim2 = get_sim_state()
                save_snapshot(sim["current_date"], cash, holdings, total, sim2["day_number"],
                              n_exec, st.session_state.strategy_text[:1000],
                              f"Day {sim2['day_number']}: {n_exec} trade(s)")
                st.session_state.exec_report     = report
                st.session_state.proposed_trades = []
                st.session_state.approved_trades = []
                st.session_state.risk_report     = ""
            st.success(f"✅ {n_exec} trade(s) executed!"); st.rerun()
    with col_hold:
        if st.button("⏭️ Hold Positions", type="secondary"):
            prices = {t: m["price"] for t,m in market_snap.items() if m["price"]}
            total, holdings, cash = get_portfolio_valuation(prices)
            sim2 = get_sim_state()
            save_snapshot(sim["current_date"], cash, holdings, total,
                          sim2["day_number"], 0, "", "Held — no trades")
            st.info("Holding. Snapshot saved."); st.rerun()
    if st.session_state.exec_report: st.markdown(st.session_state.exec_report)


# ── PORTFOLIO ─────────────────────────────────────────────────────────────────
def render_portfolio(sim, market_snap):
    st.markdown("## 💼 Portfolio Holdings")
    for a in get_target_alerts(market_snap):
        if a["type"] == "TARGET_HIT":
            st.success(f"🎯 **{a['symbol']} hit target!** ${a['price']:.2f} ≥ ${a['level']:.2f} — P&L: +{a['pnl_pct']:.1f}%")
        else:
            st.error(f"🛑 **{a['symbol']} stop-loss!** ${a['price']:.2f} ≤ ${a['level']:.2f} — P&L: {a['pnl_pct']:.1f}%")

    positions   = get_positions()
    total_value = sim["total_value"]
    if not positions:
        st.info("No open positions. Deploy capital via the Trading Floor.")
    else:
        rows = []
        for p in positions:
            sym   = p["symbol"]; sh = p["shares"]; avg_c = p["avg_cost"]
            price = market_snap.get(sym,{}).get("price") or avg_c
            mkt   = sh*price; cost = sh*avg_c
            pnl_a = mkt-cost; pnl_p = pnl_a/cost if cost else 0
            w     = mkt/total_value if total_value else 0
            tp    = p.get("target_price"); sl = p.get("stop_loss")
            rows.append({"Ticker":sym,"Shares":f"{sh:.2f}","Avg Cost":f"${avg_c:.2f}",
                         "Last Price":f"${price:.2f}","Mkt Value":format_dollar(mkt),
                         "P&L ($)":f"${pnl_a:+,.0f}","P&L (%)":format_pct(pnl_p),
                         "Weight":format_pct(w,sign=False),
                         "🎯 Target":f"${tp:.2f}" if tp else "—",
                         "🛑 Stop":f"${sl:.2f}" if sl else "—",
                         "1d Ret":format_pct(market_snap.get(sym,{}).get("ret_1d"))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider(); st.markdown("### 🛡️ Risk Summary")
    cash        = sim["cash_balance"]
    cash_pct    = cash/total_value if total_value else 0
    txns        = get_transactions(50)
    trades_today= sum(1 for t in txns if t["date"]==sim["current_date"])
    c1,c2,c3    = st.columns(3)
    with c1: st.metric("Cash Weight", format_pct(cash_pct, sign=False))
    with c2:
        if positions:
            max_w = max((p["shares"]*(market_snap.get(p["symbol"],{}).get("price") or p["avg_cost"]))/total_value for p in positions)
            st.metric("Largest Position", format_pct(max_w, sign=False),
                      delta="✅ OK" if max_w<=0.40 else "⚠️ Over limit",
                      delta_color="normal" if max_w<=0.40 else "inverse")
        else: st.metric("Largest Position","—")
    with c3:
        st.metric("Trades Today", f"{trades_today} / 2",
                  delta="✅ OK" if trades_today<=2 else "⚠️ At limit",
                  delta_color="normal" if trades_today<=2 else "inverse")

    SECTOR_MAP = {"AAPL":"Tech","MSFT":"Tech","NVDA":"Tech","GOOGL":"Tech","META":"Tech","AMZN":"Tech",
                  "JPM":"Finance","GS":"Finance","BAC":"Finance","SPY":"Index","QQQ":"Index","IWM":"Index",
                  "GLD":"Commodities","SLV":"Commodities","USO":"Commodities","UCO":"Commodities",
                  "BNO":"Commodities","DBO":"Commodities","TLT":"Bonds","IEF":"Bonds",
                  "XLE":"Energy","XOM":"Energy","CVX":"Energy","PUBM":"Tech"}
    if positions:
        st_totals: dict = {}
        for p in positions:
            price = market_snap.get(p["symbol"],{}).get("price") or p["avg_cost"]
            mkt   = p["shares"]*price; w = mkt/total_value if total_value else 0
            st_totals[SECTOR_MAP.get(p["symbol"],"Other")] = st_totals.get(SECTOR_MAP.get(p["symbol"],"Other"),0)+w
        if st_totals:
            parts = " &nbsp;|&nbsp; ".join(f"**{s}** {w*100:.1f}%" for s,w in sorted(st_totals.items(),key=lambda x:-x[1]))
            st.markdown(f"**Sector Exposure:** &nbsp; {parts}", unsafe_allow_html=True)

    st.divider(); st.markdown("### 📋 Recent Transactions")
    if txns:
        rows = [{"Date":t["date"],"Action":"🟢 BUY" if t["action"]=="BUY" else "🔴 SELL",
                 "Ticker":t["symbol"],"Shares":f"{t['shares']:.2f}","Price":f"${t['price']:.2f}",
                 "Value":format_dollar(t["value"]),"TC":f"${t.get('tc_cost',0):.2f}",
                 "Reason":(t.get("reason") or "")} for t in txns]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else: st.info("No transactions yet.")

    st.divider(); st.markdown("### ⬇️ Export Portfolio")
    if positions:
        dl_rows = []
        for p in positions:
            sym   = p["symbol"]
            price = market_snap.get(sym,{}).get("price") or p["avg_cost"]
            mkt   = p["shares"]*price
            pnl_a = mkt - p["shares"]*p["avg_cost"]
            pnl_p = pnl_a/(p["shares"]*p["avg_cost"]) if p["avg_cost"] else 0
            w     = mkt/total_value if total_value else 0
            dl_rows.append({"Date":sim["current_date"],"Symbol":sym,"Shares":round(p["shares"],4),
                            "Avg_Cost":round(p["avg_cost"],4),"Last_Price":round(price,4),
                            "Market_Value":round(mkt,2),"PnL_Dollar":round(pnl_a,2),
                            "PnL_Pct":round(pnl_p*100,3),"Weight_Pct":round(w*100,3),
                            "Target_Price":p.get("target_price") or "","Stop_Loss":p.get("stop_loss") or "",
                            "Cash":round(sim["cash_balance"],2),"Total_Value":round(total_value,2)})
        csv_bytes = pd.DataFrame(dl_rows).to_csv(index=False).encode("utf-8-sig")
        st.download_button(label=f"⬇️ Download Portfolio CSV  ({sim['current_date']})",
                           data=csv_bytes, file_name=f"portfolio_{sim['current_date']}.csv",
                           mime="text/csv", use_container_width=True)
    else: st.info("No positions to export yet.")


# ── PERFORMANCE ───────────────────────────────────────────────────────────────
def render_performance(sim, prices_df):
    st.markdown("## 📈 Performance Dashboard")
    snapshots = get_snapshots(); kpis = compute_kpis(snapshots)
    pnl_dollar = (sim["total_value"]-STARTING_CAPITAL) if sim else 0
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("📈 Total Return",
                        format_pct(kpis["total_ret"]) if kpis["total_ret"] is not None else "—",
                        delta=f"{format_dollar(pnl_dollar)} P&L" if pnl_dollar else None,
                        delta_color="normal" if pnl_dollar>=0 else "inverse")
    with k2: st.metric("📉 Max Drawdown", format_pct(kpis["max_dd"]) if kpis["max_dd"] is not None else "—")
    with k3: st.metric("🏆 Win Rate",
                        f"{kpis['win_rate']*100:.0f}%" if kpis["win_rate"] else "—",
                        delta=f"{len(snapshots)} trading days" if snapshots else None)
    with k4: st.metric("📊 Ann. Volatility", format_pct(kpis["vol_ann"],sign=False) if kpis["vol_ann"] else "—")

    if not snapshots:
        st.info("No data yet — run agents and execute trades to populate charts."); return

    st.divider()
    ctrl1,ctrl2,_ = st.columns([2,2,4])
    with ctrl1: time_range = st.selectbox("🕐 Time Range",["All","1W","2W"],index=0)
    with ctrl2:
        bm_options = [t for t in ["SPY","QQQ","IWM","GLD","TLT"] if t in prices_df.columns] or ["SPY"]
        selected_bm= st.selectbox("📊 Benchmark",["None"]+bm_options,index=1 if bm_options else 0)

    df_s = pd.DataFrame(snapshots).copy()
    df_s["date"]    = pd.to_datetime(df_s["date"].astype(str).str[:10]).dt.date
    df_s            = df_s.sort_values("date").reset_index(drop=True)
    df_s["date_ts"] = pd.to_datetime(df_s["date"])
    df_s["ret_pct"] = (df_s["total_equity"]/STARTING_CAPITAL-1)*100
    if time_range != "All" and len(df_s) > 0:
        cutoff = df_s["date_ts"].max()-pd.Timedelta(days={"1W":5,"2W":10}.get(time_range,999))
        df_s   = df_s[df_s["date_ts"]>=cutoff].reset_index(drop=True)
        if not df_s.empty:
            df_s["ret_pct"] = (df_s["total_equity"]/df_s["total_equity"].iloc[0]-1)*100
    BM = selected_bm if selected_bm!="None" else None
    st.divider()

    # Chart 1
    st.markdown("### 📊 Portfolio Value")
    ev   = df_s["total_equity"]
    pad  = max((ev.max()-ev.min())*0.3,500)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_s["date_ts"],y=ev,mode="lines+markers",name="Portfolio",
        line=dict(color=_ACCENT,width=2.5),marker=dict(size=7,color=_ACCENT),
        fill="tozeroy",fillcolor="rgba(123,175,212,0.10)",
        hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra></extra>"))
    if BM and BM in prices_df.columns:
        bm_raw = prices_df[BM].copy(); bm_raw.index = pd.to_datetime(bm_raw.index.astype(str).str[:10])
        bm = bm_raw.reindex(df_s["date_ts"]).ffill()
        if not bm.dropna().empty:
            bm_n = bm/bm.dropna().iloc[0]*STARTING_CAPITAL
            fig1.add_trace(go.Scatter(x=bm_n.index,y=bm_n.values,mode="lines",name=f"{BM} (norm.)",
                line=dict(color=_MUTED,width=1.5,dash="dash"),
                hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra></extra>"))
    fig1.add_hline(y=STARTING_CAPITAL,line_dash="dot",line_color=_AMBER,
                   annotation_text="Start $1M",annotation_position="bottom right",annotation_font_color=_AMBER)
    l1 = _layout(height=320,yprefix="$",yfmt=",.0f"); l1["yaxis"]["range"]=[ev.min()-pad,ev.max()+pad]
    fig1.update_layout(**l1); st.plotly_chart(fig1,use_container_width=True)

    # Chart 2
    st.markdown(f"### 📉 Cumulative Return vs {BM or 'Benchmark'} (%)")
    last_ret = df_s["ret_pct"].iloc[-1] if len(df_s) else 0
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_s["date_ts"],y=df_s["ret_pct"],mode="lines+markers",name="Portfolio",
        line=dict(color=_GREEN if last_ret>=0 else _RED,width=2.5),marker=dict(size=6),
        hovertemplate="<b>%{x|%b %d}</b><br>%{y:+.3f}%<extra></extra>"))
    if BM and BM in prices_df.columns:
        bm2 = prices_df[BM].copy(); bm2.index = pd.to_datetime(bm2.index.astype(str).str[:10])
        bm2 = bm2.reindex(df_s["date_ts"]).ffill().dropna()
        if not bm2.empty:
            bm2_ret = (bm2/bm2.iloc[0]-1)*100
            fig2.add_trace(go.Scatter(x=bm2_ret.index,y=bm2_ret.values,mode="lines",name=f"{BM}",
                line=dict(color=_MUTED,width=1.5,dash="dash"),
                hovertemplate="<b>%{x|%b %d}</b><br>%{y:+.3f}%<extra></extra>"))
    fig2.add_hline(y=0,line_dash="dot",line_color=_BORDER)
    fig2.update_layout(**_layout(height=250,ysuffix="%",yfmt="+.2f")); st.plotly_chart(fig2,use_container_width=True)

    # Chart 3
    if len(df_s)>1:
        st.markdown("### 📊 Daily P&L ($)")
        df_s["daily_pnl"] = df_s["total_equity"].diff().fillna(0)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=df_s["date_ts"],y=df_s["daily_pnl"],
            marker_color=[_GREEN if v>=0 else _RED for v in df_s["daily_pnl"]],
            hovertemplate="<b>%{x|%b %d}</b><br>%{y:+$,.0f}<extra></extra>"))
        fig3.add_hline(y=0,line_color=_MUTED,line_width=1)
        fig3.update_layout(**_layout(height=210,yprefix="$",yfmt="+,.0f")); st.plotly_chart(fig3,use_container_width=True)

    # Chart 4
    st.markdown("### 💼 Cash vs Holdings Over Time")
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=df_s["date_ts"],y=df_s["holdings_value"],name="Holdings",marker_color=_ACCENT,
        hovertemplate="<b>%{x|%b %d}</b><br>Holdings: $%{y:,.0f}<extra></extra>"))
    fig4.add_trace(go.Bar(x=df_s["date_ts"],y=df_s["cash"],name="Cash",marker_color=_BORDER,
        hovertemplate="<b>%{x|%b %d}</b><br>Cash: $%{y:,.0f}<extra></extra>"))
    l4=_layout(height=210,yprefix="$",yfmt=",.0f"); l4["barmode"]="stack"
    fig4.update_layout(**l4); st.plotly_chart(fig4,use_container_width=True)

    st.divider(); st.markdown("### 📋 Daily Snapshots")
    df_t = df_s[["date","cash","holdings_value","total_equity","ret_pct","trades_count","notes"]].copy()
    df_t.columns = ["Date","Cash ($)","Holdings ($)","Total Equity ($)","Return (%)","Trades","Notes"]
    for c in ["Cash ($)","Holdings ($)","Total Equity ($)"]: df_t[c]=df_t[c].apply(lambda x:f"${float(x):,.0f}")
    df_t["Return (%)"]=df_t["Return (%)"].apply(lambda x:f"{float(x):+.3f}%")
    st.dataframe(df_t.sort_values("Date",ascending=False),use_container_width=True,hide_index=True)


# ── OVERVIEW ──────────────────────────────────────────────────────────────────
def render_overview(sim, prices_df):
    st.markdown("## 🔭 Strategy Overview")
    st.markdown("Cross-day analytics — rolling risk-adjusted ratios, drawdown, "
                "capital deployment, and live valuation multiples for current positions.")

    snapshots = get_snapshots()
    if not snapshots or len(snapshots) < 2:
        st.info("Need at least 2 trading days of data to populate this tab."); return

    df_s = pd.DataFrame(snapshots).copy()
    df_s["date"]      = pd.to_datetime(df_s["date"].astype(str).str[:10]).dt.date
    df_s              = df_s.sort_values("date").reset_index(drop=True)
    df_s["date_ts"]   = pd.to_datetime(df_s["date"])
    df_s["ret_pct"]   = (df_s["total_equity"]/STARTING_CAPITAL-1)*100
    df_s["daily_ret"] = df_s["total_equity"].pct_change().fillna(0)
    n = len(df_s); rets = df_s["daily_ret"].values; ANN = 252

    def _sharpe(r, w=5):
        out=[None]*len(r)
        for i in range(w-1,len(r)):
            ww=r[i-w+1:i+1]; mu=np.mean(ww); sd=np.std(ww,ddof=1)
            out[i]=(mu/sd*np.sqrt(ANN)) if sd>1e-10 else 0.0
        return out

    def _sortino(r, w=5):
        out=[None]*len(r)
        for i in range(w-1,len(r)):
            ww=r[i-w+1:i+1]; mu=np.mean(ww)
            neg=np.array([x for x in ww if x<0]); dd=np.std(neg,ddof=1) if len(neg)>1 else 1e-9
            out[i]=(mu/dd*np.sqrt(ANN)) if dd>1e-10 else 0.0
        return out

    def _calmar(equity, w=5):
        out=[None]*len(equity)
        for i in range(w-1,len(equity)):
            ww=equity[i-w+1:i+1]; ret=(ww[-1]/ww[0]-1)*ANN/w
            rm=np.maximum.accumulate(ww); dd=np.min((np.array(ww)-rm)/rm)
            out[i]=(ret/abs(dd)) if abs(dd)>1e-10 else 0.0
        return out

    df_s["sharpe_5d"]    = _sharpe(rets)
    df_s["sortino_5d"]   = _sortino(rets)
    df_s["calmar_5d"]    = _calmar(df_s["total_equity"].values)
    roll_max             = df_s["total_equity"].cummax()
    df_s["drawdown"]     = (df_s["total_equity"]-roll_max)/roll_max*100
    df_s["deployed_pct"] = df_s["holdings_value"]/df_s["total_equity"]*100
    df_s["daily_vol"]    = df_s["daily_ret"].rolling(5).std().fillna(0)*np.sqrt(ANN)*100

    def _last(series):
        vals=[v for v in series.dropna() if v is not None]; return vals[-1] if vals else None

    last_sharpe  = _last(df_s["sharpe_5d"])
    last_sortino = _last(df_s["sortino_5d"])
    last_calmar  = _last(df_s["calmar_5d"])
    last_dd      = df_s["drawdown"].min()
    win_days     = int((df_s["daily_ret"]>0).sum())
    total_ret    = df_s["ret_pct"].iloc[-1]

    st.divider()
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.metric("📈 Total Return",       f"{total_ret:+.2f}%")
    with k2: st.metric("⚡ Sharpe (5d)",         f"{last_sharpe:.2f}"  if last_sharpe  else "—")
    with k3: st.metric("🎯 Sortino (5d)",        f"{last_sortino:.2f}" if last_sortino else "—")
    with k4: st.metric("🏔️ Calmar (5d)",         f"{last_calmar:.2f}"  if last_calmar  else "—")
    with k5: st.metric("📉 Max Drawdown",        f"{last_dd:.2f}%")
    with k6: st.metric("🏆 Win Days",            f"{win_days} / {n}")
    st.divider()

    # Row 1: Sharpe + Sortino
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ⚡ Rolling Sharpe Ratio (5-day)")
        st.caption("Annualised.  > 1 = good  ·  > 2 = excellent  ·  < 0 = underwater")
        df_sr = df_s[df_s["sharpe_5d"].notna()]
        fig_sr = go.Figure()
        fig_sr.add_hrect(y0=1,y1=50,  fillcolor="rgba(91,196,160,0.06)",line_width=0)
        fig_sr.add_hrect(y0=-50,y1=0, fillcolor="rgba(224,112,112,0.06)",line_width=0)
        fig_sr.add_trace(go.Scatter(x=df_sr["date_ts"],y=df_sr["sharpe_5d"],mode="lines+markers",
            line=dict(color=_ACCENT,width=2.5),marker=dict(size=8,color=_ACCENT),
            hovertemplate="<b>%{x|%b %d}</b><br>Sharpe: %{y:.2f}<extra></extra>"))
        fig_sr.add_hline(y=1,line_dash="dot",line_color=_GREEN,
                         annotation_text="SR=1",annotation_font_color=_GREEN)
        fig_sr.add_hline(y=0,line_color=_RED,line_width=1)
        fig_sr.update_layout(**_layout(height=260)); st.plotly_chart(fig_sr,use_container_width=True)

    with col_b:
        st.markdown("#### 🎯 Rolling Sortino Ratio (5-day)")
        st.caption("Penalises only downside deviation — higher is better")
        df_so = df_s[df_s["sortino_5d"].notna()]
        fig_so = go.Figure()
        fig_so.add_trace(go.Scatter(x=df_so["date_ts"],y=df_so["sortino_5d"],mode="lines+markers",
            line=dict(color=_MUTED,width=2.5),marker=dict(size=8,color=_MUTED),
            hovertemplate="<b>%{x|%b %d}</b><br>Sortino: %{y:.2f}<extra></extra>"))
        fig_so.add_hline(y=1,line_dash="dot",line_color=_GREEN,
                         annotation_text="SR=1",annotation_font_color=_GREEN)
        fig_so.add_hline(y=0,line_color=_RED,line_width=1)
        fig_so.update_layout(**_layout(height=260)); st.plotly_chart(fig_so,use_container_width=True)

    # Row 2: Drawdown + Deployment
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### 📉 Drawdown from Peak (%)")
        st.caption("0% = at all-time high.  Deeper = larger loss from peak.")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=df_s["date_ts"],y=df_s["drawdown"],mode="lines+markers",
            line=dict(color=_RED,width=2.5),marker=dict(size=7,color=_RED),
            fill="tozeroy",fillcolor="rgba(224,112,112,0.12)",
            hovertemplate="<b>%{x|%b %d}</b><br>DD: %{y:.2f}%<extra></extra>"))
        fig_dd.add_hline(y=0,line_color=_MUTED,line_width=1)
        fig_dd.update_layout(**_layout(height=260,ysuffix="%",yfmt=".2f")); st.plotly_chart(fig_dd,use_container_width=True)

    with col_d:
        st.markdown("#### 💰 Capital Deployment (%)")
        st.caption("Invested portion of total equity each day")
        fig_dep = go.Figure()
        fig_dep.add_trace(go.Bar(x=df_s["date_ts"],y=df_s["deployed_pct"],
            marker_color=_ACCENT,
            hovertemplate="<b>%{x|%b %d}</b><br>Deployed: %{y:.1f}%<extra></extra>"))
        fig_dep.add_hline(y=70,line_dash="dot",line_color=_GREEN,
                          annotation_text="Target 70%",annotation_font_color=_GREEN,annotation_position="right")
        fig_dep.update_layout(**_layout(height=260,ysuffix="%",yfmt=".0f")); st.plotly_chart(fig_dep,use_container_width=True)

    # Row 3: Daily returns + Rolling vol
    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("#### 📊 Daily Return Distribution (%)")
        st.caption("Green = profit · Red = loss · Gold = average")
        daily_rets_pct = df_s["daily_ret"]*100
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(x=df_s["date_ts"],y=daily_rets_pct,
            marker_color=[_GREEN if v>=0 else _RED for v in daily_rets_pct],
            hovertemplate="<b>%{x|%b %d}</b><br>%{y:+.3f}%<extra></extra>"))
        mu = daily_rets_pct.mean()
        fig_dist.add_hline(y=0,line_color=_MUTED,line_width=1)
        fig_dist.add_hline(y=mu,line_dash="dot",line_color=_AMBER,
                           annotation_text=f"Avg {mu:+.2f}%",annotation_font_color=_AMBER,annotation_position="right")
        fig_dist.update_layout(**_layout(height=260,ysuffix="%",yfmt="+.2f")); st.plotly_chart(fig_dist,use_container_width=True)

    with col_f:
        st.markdown("#### 📈 Rolling Annualised Volatility (5-day)")
        st.caption("Higher = more volatile strategy")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df_s["date_ts"],y=df_s["daily_vol"],mode="lines+markers",
            line=dict(color=_AMBER,width=2.5),marker=dict(size=7,color=_AMBER),
            fill="tozeroy",fillcolor="rgba(200,168,75,0.08)",
            hovertemplate="<b>%{x|%b %d}</b><br>Vol: %{y:.1f}%<extra></extra>"))
        fig_vol.update_layout(**_layout(height=260,ysuffix="%",yfmt=".1f")); st.plotly_chart(fig_vol,use_container_width=True)

    # Row 4: Calmar + Trades
    col_g, col_h = st.columns(2)
    with col_g:
        st.markdown("#### 🏔️ Rolling Calmar Ratio (5-day)")
        st.caption("Ann. return ÷ max drawdown.  Higher = better return per unit of drawdown risk.")
        df_cal = df_s[df_s["calmar_5d"].notna()]
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=df_cal["date_ts"],y=df_cal["calmar_5d"],mode="lines+markers",
            line=dict(color=_GREEN,width=2.5),marker=dict(size=7,color=_GREEN),
            hovertemplate="<b>%{x|%b %d}</b><br>Calmar: %{y:.2f}<extra></extra>"))
        fig_cal.add_hline(y=0,line_color=_RED,line_width=1)
        fig_cal.update_layout(**_layout(height=260)); st.plotly_chart(fig_cal,use_container_width=True)

    with col_h:
        st.markdown("#### 🔄 Trades Executed per Day")
        st.caption("Max 2 per day allowed by risk rules")
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Bar(x=df_s["date_ts"],y=df_s["trades_count"],
            marker_color=_SURFACE, marker_line_color=_ACCENT,marker_line_width=1.5,
            hovertemplate="<b>%{x|%b %d}</b><br>Trades: %{y}<extra></extra>"))
        fig_tr.add_hline(y=2,line_dash="dot",line_color=_AMBER,
                         annotation_text="Max 2",annotation_font_color=_AMBER,annotation_position="right")
        fig_tr.update_layout(**_layout(height=260,yfmt=".0f")); st.plotly_chart(fig_tr,use_container_width=True)

    # Valuation ratios
    st.divider()
    st.markdown("### 📐 Current Position Valuation Ratios (P/E, P/B, EV/EBITDA, Beta)")
    st.caption("Live data from Yahoo Finance — may lag for ETFs and commodity funds.")
    positions = get_positions()
    if not positions:
        st.info("No open positions.")
    else:
        import yfinance as yf
        total_value = sim["total_value"]
        ratio_rows  = []
        with st.spinner("Fetching valuation data…"):
            for p in positions:
                sym = p["symbol"]
                try:
                    info  = yf.Ticker(sym).info
                    price = info.get("regularMarketPrice") or p["avg_cost"]
                    mkt   = p["shares"]*price; w = mkt/total_value if total_value else 0
                    ratio_rows.append({
                        "Ticker":       sym,
                        "Weight":       f"{w*100:.1f}%",
                        "P/E (Trail.)": f"{info['trailingPE']:.1f}"       if info.get("trailingPE")           else "—",
                        "P/E (Fwd)":    f"{info['forwardPE']:.1f}"        if info.get("forwardPE")            else "—",
                        "P/B":          f"{info['priceToBook']:.2f}"      if info.get("priceToBook")          else "—",
                        "EV/EBITDA":    f"{info['enterpriseToEbitda']:.1f}" if info.get("enterpriseToEbitda") else "—",
                        "Beta":         f"{info['beta']:.2f}"             if info.get("beta")                 else "—",
                        "Div Yield":    f"{info['dividendYield']*100:.2f}%" if info.get("dividendYield")      else "—",
                    })
                except Exception:
                    ratio_rows.append({"Ticker":sym,"Weight":"—","P/E (Trail.)":"—",
                                       "P/E (Fwd)":"—","P/B":"—","EV/EBITDA":"—","Beta":"—","Div Yield":"—"})
        if ratio_rows:
            st.dataframe(pd.DataFrame(ratio_rows),use_container_width=True,hide_index=True)
            # Weight bar
            try:
                df_w = pd.DataFrame(ratio_rows)
                df_w["w_val"] = pd.to_numeric(df_w["Weight"].str.replace("%",""),errors="coerce").fillna(0)
                fig_w = go.Figure()
                fig_w.add_trace(go.Bar(x=df_w["Ticker"],y=df_w["w_val"],marker_color=_ACCENT,
                    marker_line_color=_BORDER,marker_line_width=1,
                    hovertemplate="<b>%{x}</b><br>Weight: %{y:.1f}%<extra></extra>"))
                fig_w.add_hline(y=40,line_dash="dot",line_color=_RED,
                                annotation_text="Max 40%",annotation_font_color=_RED,annotation_position="right")
                lw = _layout(height=220,ysuffix="%",yfmt=".1f")
                lw["xaxis"]["type"]="category"; lw["xaxis"]["tickformat"]=None
                fig_w.update_layout(**lw); st.plotly_chart(fig_w,use_container_width=True)
            except Exception:
                pass

    # Rolling metrics table
    st.divider(); st.markdown("### 📋 Daily Rolling Metrics Table")
    def _fmt(v, fmt):
        try:
            return fmt.format(float(v)) if v is not None and not (isinstance(v,float) and np.isnan(v)) else "—"
        except Exception: return "—"

    df_disp = df_s[["date","total_equity","ret_pct","daily_ret",
                    "sharpe_5d","sortino_5d","calmar_5d","drawdown","deployed_pct","trades_count"]].copy()
    df_disp.columns = ["Date","Total Equity","Return (%)","Daily Ret.",
                        "Sharpe","Sortino","Calmar","Drawdown (%)","Deployed (%)","Trades"]
    df_disp["Total Equity"] = df_disp["Total Equity"].apply(lambda x: f"${float(x):,.0f}")
    df_disp["Return (%)"]   = df_disp["Return (%)"].apply(lambda x: f"{float(x):+.2f}%")
    df_disp["Daily Ret."]   = df_disp["Daily Ret."].apply(lambda x: f"{float(x)*100:+.3f}%")
    for col in ["Sharpe","Sortino","Calmar"]:
        df_disp[col] = df_disp[col].apply(lambda x: _fmt(x,"{:.2f}"))
    df_disp["Drawdown (%)"] = df_disp["Drawdown (%)"].apply(lambda x: f"{float(x):.2f}%")
    df_disp["Deployed (%)"] = df_disp["Deployed (%)"].apply(lambda x: f"{float(x):.1f}%")
    st.dataframe(df_disp.sort_values("Date",ascending=False),use_container_width=True,hide_index=True)


# ── BACKTEST ──────────────────────────────────────────────────────────────────
# ── Shared backtest data loader ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _enrich_log(log_df, trades_df=None):
    """Clean and add all derived columns to a raw log DataFrame.
    Safe to call on data from any source (disk or uploader)."""
    log_df = log_df.copy()
    log_df.columns  = [c.lstrip("\ufeff").strip() for c in log_df.columns]
    log_df["date"]  = pd.to_datetime(log_df["date"].astype(str).str[:10])
    log_df          = log_df.sort_values("date").reset_index(drop=True)
    for col in ["daily_ret", "cum_ret", "portfolio", "cash", "holdings"]:
        log_df[col] = pd.to_numeric(log_df[col], errors="coerce").fillna(0)

    ANN                    = 252
    raw_rets               = log_df["portfolio"].pct_change().fillna(0).values
    log_df["raw_ret"]      = raw_rets
    roll_max               = log_df["portfolio"].cummax()
    log_df["drawdown"]     = (log_df["portfolio"] - roll_max) / roll_max * 100
    log_df["deployed_pct"] = log_df["holdings"] / log_df["portfolio"].replace(0, np.nan) * 100

    W = 3
    sharpe_out = [None] * len(raw_rets)
    sortino_out= [None] * len(raw_rets)
    calmar_out = [None] * len(raw_rets)
    vol_out    = [None] * len(raw_rets)
    for i in range(W - 1, len(raw_rets)):
        w  = raw_rets[i - W + 1 : i + 1]
        mu = np.mean(w); sd = np.std(w, ddof=1)
        sharpe_out[i]  = mu / sd * np.sqrt(ANN)  if sd  > 1e-10 else 0.0
        neg = w[w < 0]; dd2 = np.std(neg, ddof=1) if len(neg) > 1 else 1e-9
        sortino_out[i] = mu / dd2 * np.sqrt(ANN) if dd2 > 1e-10 else 0.0
        vol_out[i]     = sd * np.sqrt(ANN) * 100
        eq_w  = log_df["portfolio"].values[i - W + 1 : i + 1]
        ret_w = (eq_w[-1] / eq_w[0] - 1) * ANN / W
        rm_w  = np.maximum.accumulate(eq_w)
        dd_w  = np.min((eq_w - rm_w) / rm_w)
        calmar_out[i] = ret_w / abs(dd_w) if abs(dd_w) > 1e-10 else 0.0

    log_df["sharpe"]  = sharpe_out
    log_df["sortino"] = sortino_out
    log_df["calmar"]  = calmar_out
    log_df["vol_ann"] = vol_out

    if trades_df is not None:
        trades_df = trades_df.copy()
        trades_df.columns = [c.lstrip("\ufeff").strip() for c in trades_df.columns]

    return log_df, trades_df


@st.cache_data(show_spinner=False)
def _load_backtest_data():
    """Load backtest CSVs from disk (cached). Returns (log_df, trades_df) or (None, None)."""
    log_df = trades_df = None
    if os.path.exists("backtest_daily_log.csv"):
        log_df = pd.read_csv("backtest_daily_log.csv")
    if os.path.exists("backtest_trades.csv"):
        trades_df = pd.read_csv("backtest_trades.csv")
    if log_df is None:
        return None, None
    return _enrich_log(log_df, trades_df)


def _bt_uploader(key_prefix="bt"):
    """Show file uploaders; enriches and returns (log_df, trades_df) from uploaded files."""
    st.info("Copy `backtest_daily_log.csv` and `backtest_trades.csv` into your project folder, "
            "or upload them below.")
    col_u1, col_u2 = st.columns(2)
    raw_log = raw_tr = None
    with col_u1:
        up_log = st.file_uploader("📂 backtest_daily_log.csv", type="csv", key=f"{key_prefix}_log")
        if up_log: raw_log = pd.read_csv(up_log)
    with col_u2:
        up_tr = st.file_uploader("📂 backtest_trades.csv", type="csv", key=f"{key_prefix}_trades")
        if up_tr: raw_tr = pd.read_csv(up_tr)
    if raw_log is None:
        return None, None
    return _enrich_log(raw_log, raw_tr)


# ── TAB: Backtest Results (tables + raw charts) ───────────────────────────────
def render_backtest():
    st.markdown("## 📊 Backtest Results")
    st.markdown(
        "Pre-competition walk-forward backtest — **Feb 27 → Mar 12, 2026** (10 trading days).  "
        "Momentum alpha scoring with forced daily turnover, 10-bps transaction costs."
    )

    log_df, trades_df = _load_backtest_data()
    if log_df is None:
        log_df, trades_df = _bt_uploader(key_prefix="bt5")
    if log_df is None:
        st.warning("No backtest data loaded yet."); return

    dd_ser    = log_df["drawdown"]
    ANN       = 252
    rets      = log_df["raw_ret"].values
    max_dd    = dd_ser.min()
    win_days  = int((log_df["daily_ret"] > 0).sum())
    total_ret = log_df["cum_ret"].iloc[-1]
    final_val = log_df["portfolio"].iloc[-1]
    mu = np.mean(rets[1:]); sd = np.std(rets[1:], ddof=1)
    sharpe   = mu/sd*np.sqrt(ANN) if sd > 1e-10 else 0
    neg      = rets[rets < 0]; sortino_dd = np.std(neg, ddof=1) if len(neg) > 1 else 1e-9
    sortino  = mu/sortino_dd*np.sqrt(ANN) if sortino_dd > 1e-10 else 0
    total_trades  = int(log_df["n_trades"].sum()) if "n_trades" in log_df.columns else 0
    deployed_avg  = log_df["deployed_pct"].mean()

    # KPI strip
    st.divider()
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.metric("💰 Final Value",   f"${final_val:,.0f}", delta=f"+${final_val-1_000_000:,.0f}")
    with k2: st.metric("📈 Total Return",  f"{total_ret:+.2f}%")
    with k3: st.metric("⚡ Ann. Sharpe",    f"{sharpe:.2f}")
    with k4: st.metric("🎯 Ann. Sortino",  f"{sortino:.2f}")
    with k5: st.metric("📉 Max Drawdown",  f"{max_dd:.2f}%")
    with k6: st.metric("🏆 Win Days",      f"{win_days} / {len(log_df)}")
    st.divider()

    # Row 1: equity curve + daily return bars
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📈 Portfolio Value Over Time")
        st.caption(f"Start $1,000,000 → Final ${final_val:,.0f}  ·  Avg deployed {deployed_avg:.1f}%")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=log_df["date"], y=log_df["portfolio"], mode="lines+markers",
            line=dict(color=_ACCENT, width=2.5), marker=dict(size=8, color=_ACCENT),
            fill="tozeroy", fillcolor="rgba(123,175,212,0.10)",
            hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra></extra>"))
        fig_eq.add_hline(y=1_000_000, line_dash="dot", line_color=_AMBER,
                         annotation_text="Start $1M", annotation_font_color=_AMBER,
                         annotation_position="bottom right")
        l = _layout(height=280, yprefix="$", yfmt=",.0f")
        pad = max((log_df["portfolio"].max()-log_df["portfolio"].min())*0.2, 2000)
        l["yaxis"]["range"] = [log_df["portfolio"].min()-pad, log_df["portfolio"].max()+pad]
        fig_eq.update_layout(**l); st.plotly_chart(fig_eq, use_container_width=True)

    with col_b:
        st.markdown("#### 📊 Daily Return (%)")
        st.caption("Day-over-day % change · Gold = average")
        dr = log_df["daily_ret"]
        fig_dr = go.Figure()
        fig_dr.add_trace(go.Bar(x=log_df["date"], y=dr,
            marker_color=[_GREEN if v >= 0 else _RED for v in dr],
            hovertemplate="<b>%{x|%b %d}</b><br>%{y:+.3f}%<extra></extra>"))
        mu_dr = dr[dr != 0].mean()
        fig_dr.add_hline(y=0, line_color=_MUTED, line_width=1)
        fig_dr.add_hline(y=mu_dr, line_dash="dot", line_color=_AMBER,
                         annotation_text=f"Avg {mu_dr:+.2f}%",
                         annotation_font_color=_AMBER, annotation_position="right")
        fig_dr.update_layout(**_layout(height=280, ysuffix="%", yfmt="+.2f"))
        st.plotly_chart(fig_dr, use_container_width=True)

    # Row 2: drawdown + cash vs holdings
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### 📉 Drawdown from Peak (%)")
        st.caption(f"Max drawdown: {max_dd:.2f}%")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=log_df["date"], y=dd_ser, mode="lines+markers",
            line=dict(color=_RED, width=2.5), marker=dict(size=7, color=_RED),
            fill="tozeroy", fillcolor="rgba(224,112,112,0.12)",
            hovertemplate="<b>%{x|%b %d}</b><br>DD: %{y:.2f}%<extra></extra>"))
        fig_dd.add_hline(y=0, line_color=_MUTED, line_width=1)
        fig_dd.update_layout(**_layout(height=240, ysuffix="%", yfmt=".2f"))
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_d:
        st.markdown("#### 💼 Cash vs Holdings Over Time")
        st.caption(f"Total trades: {total_trades}  ·  Avg deployed: {deployed_avg:.1f}%")
        fig_ch = go.Figure()
        fig_ch.add_trace(go.Bar(x=log_df["date"], y=log_df["holdings"], name="Holdings",
            marker_color=_ACCENT,
            hovertemplate="<b>%{x|%b %d}</b><br>Holdings: $%{y:,.0f}<extra></extra>"))
        fig_ch.add_trace(go.Bar(x=log_df["date"], y=log_df["cash"], name="Cash",
            marker_color=_BORDER,
            hovertemplate="<b>%{x|%b %d}</b><br>Cash: $%{y:,.0f}<extra></extra>"))
        l_ch = _layout(height=240, yprefix="$", yfmt=",.0f"); l_ch["barmode"] = "stack"
        fig_ch.update_layout(**l_ch); st.plotly_chart(fig_ch, use_container_width=True)

    # Row 3: cumulative return + positions
    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("#### 📈 Cumulative Return (%)")
        st.caption(f"Final: {total_ret:+.2f}%")
        fig_cr = go.Figure()
        fig_cr.add_trace(go.Scatter(
            x=log_df["date"], y=log_df["cum_ret"], mode="lines+markers",
            line=dict(color=_GREEN, width=2.5), marker=dict(size=8, color=_GREEN),
            fill="tozeroy", fillcolor="rgba(91,196,160,0.10)",
            hovertemplate="<b>%{x|%b %d}</b><br>%{y:+.2f}%<extra></extra>"))
        fig_cr.add_hline(y=0, line_dash="dot", line_color=_BORDER)
        fig_cr.update_layout(**_layout(height=240, ysuffix="%", yfmt="+.2f"))
        st.plotly_chart(fig_cr, use_container_width=True)

    with col_f:
        st.markdown("#### 🔢 Positions Held per Day")
        st.caption("Number of open positions each day")
        n_pos = (log_df["positions"].apply(lambda x: len(str(x).split("|")) if pd.notna(x) else 0)
                 if "positions" in log_df.columns else pd.Series([0]*len(log_df)))
        fig_pos = go.Figure()
        fig_pos.add_trace(go.Bar(x=log_df["date"], y=n_pos,
            marker_color=_SURFACE, marker_line_color=_ACCENT, marker_line_width=1.5,
            hovertemplate="<b>%{x|%b %d}</b><br>Positions: %{y}<extra></extra>"))
        fig_pos.update_layout(**_layout(height=240, yfmt=".0f"))
        st.plotly_chart(fig_pos, use_container_width=True)

    # Daily snapshot table
    st.divider()
    st.markdown("### 📋 Daily Backtest Snapshot")
    disp = log_df.copy()
    disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
    df_out = pd.DataFrame({
        "Date":            disp["date"],
        "Portfolio Value": disp["portfolio"].apply(lambda x: f"${x:,.0f}"),
        "Cash":            disp["cash"].apply(lambda x: f"${x:,.0f}"),
        "Holdings":        disp["holdings"].apply(lambda x: f"${x:,.0f}"),
        "Daily Ret.":      disp["daily_ret"].apply(lambda x: f"{x:+.3f}%"),
        "Cum. Ret.":       disp["cum_ret"].apply(lambda x: f"{x:+.3f}%"),
        "Drawdown":        dd_ser.apply(lambda x: f"{x:.2f}%"),
        **( {"Trades": disp["n_trades"]} if "n_trades" in disp.columns else {} ),
    })
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    # Trades table
    if trades_df is not None:
        st.divider()
        st.markdown("### 🔄 All Backtest Trades")
        ctrl1, ctrl2 = st.columns(2)
        with ctrl1:
            action_filter = st.multiselect("Filter by Action",
                options=sorted(trades_df["action"].unique().tolist()),
                default=sorted(trades_df["action"].unique().tolist()), key="bt_action_filter")
        with ctrl2:
            sym_opts = sorted(trades_df["symbol"].unique().tolist())
            sym_filter = st.multiselect("Filter by Ticker", options=sym_opts,
                                        default=sym_opts, key="bt_sym_filter")
        filtered = trades_df[
            trades_df["action"].isin(action_filter) &
            trades_df["symbol"].isin(sym_filter)
        ].copy()
        if "price"   in filtered.columns: filtered["price"]   = filtered["price"].apply(lambda x: f"${float(x):.2f}")
        if "value"   in filtered.columns: filtered["value"]   = filtered["value"].apply(lambda x: f"${float(x):,.0f}")
        if "shares"  in filtered.columns: filtered["shares"]  = filtered["shares"].apply(lambda x: f"{float(x):.2f}")
        if "pnl_pct" in filtered.columns:
            filtered["pnl_pct"] = filtered["pnl_pct"].apply(
                lambda x: f"{float(x):+.2f}%" if pd.notna(x) and str(x).strip() not in ("","nan") else "—")
        rename_t = {"date":"Date","action":"Action","symbol":"Ticker","price":"Price",
                    "shares":"Shares","value":"Value","pnl_pct":"P&L %","reason":"Reason"}
        ft_ren = filtered.rename(columns=rename_t)
        st.dataframe(ft_ren[[c for c in rename_t.values() if c in ft_ren.columns]],
                     use_container_width=True, hide_index=True)
        st.caption(f"{len(filtered)} trades shown · {len(trades_df)} total")
        st.download_button("⬇️ Download Backtest Trades CSV",
                           data=trades_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="backtest_trades.csv", mime="text/csv",
                           use_container_width=True)


# ── TAB: Backtest Performance (rolling ratios & risk analytics) ───────────────
def render_backtest_performance():
    st.markdown("## 📈 Backtest Performance Analytics")
    st.markdown(
        "Rolling risk-adjusted ratios, capital deployment, volatility and trade analytics "
        "computed on the pre-competition backtest (Feb 27 → Mar 12, 2026, 3-day rolling window)."
    )

    log_df, trades_df = _load_backtest_data()
    if log_df is None:
        log_df, _ = _bt_uploader(key_prefix="bt6")
    if log_df is None:
        st.warning("No backtest data loaded yet."); return

    ANN       = 252
    dd_ser    = log_df["drawdown"]
    rets      = log_df["raw_ret"].values
    max_dd    = dd_ser.min()
    win_days  = int((log_df["daily_ret"] > 0).sum())
    total_ret = log_df["cum_ret"].iloc[-1]
    final_val = log_df["portfolio"].iloc[-1]
    mu = np.mean(rets[1:]); sd = np.std(rets[1:], ddof=1)
    sharpe   = mu/sd*np.sqrt(ANN) if sd > 1e-10 else 0
    neg      = rets[rets < 0]; sdd = np.std(neg, ddof=1) if len(neg) > 1 else 1e-9
    sortino  = mu/sdd*np.sqrt(ANN) if sdd > 1e-10 else 0
    last_calmar = [v for v in log_df["calmar"] if v is not None]
    calmar_val  = last_calmar[-1] if last_calmar else 0
    deployed_avg = log_df["deployed_pct"].mean()

    # ── KPI strip ─────────────────────────────────────────────────────────────
    st.divider()
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.metric("📈 Total Return",   f"{total_ret:+.2f}%")
    with k2: st.metric("⚡ Ann. Sharpe",     f"{sharpe:.2f}")
    with k3: st.metric("🎯 Ann. Sortino",   f"{sortino:.2f}")
    with k4: st.metric("🏔️ Calmar",          f"{calmar_val:.2f}")
    with k5: st.metric("📉 Max Drawdown",   f"{max_dd:.2f}%")
    with k6: st.metric("💰 Avg Deployed",   f"{deployed_avg:.1f}%")
    st.divider()

    # ── Row 1: Rolling Sharpe + Rolling Sortino ────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ⚡ Rolling Sharpe Ratio (3-day)")
        st.caption("Annualised.  > 1 = good  ·  > 2 = excellent  ·  < 0 = underwater")
        df_sr = log_df[log_df["sharpe"].notna()].copy()
        fig_sr = go.Figure()
        fig_sr.add_hrect(y0=1,  y1=50,  fillcolor="rgba(91,196,160,0.06)",  line_width=0)
        fig_sr.add_hrect(y0=-50,y1=0,   fillcolor="rgba(224,112,112,0.06)", line_width=0)
        fig_sr.add_trace(go.Scatter(
            x=df_sr["date"], y=df_sr["sharpe"], mode="lines+markers",
            line=dict(color=_ACCENT, width=2.5), marker=dict(size=9, color=_ACCENT),
            hovertemplate="<b>%{x|%b %d}</b><br>Sharpe: %{y:.2f}<extra></extra>"))
        fig_sr.add_hline(y=1, line_dash="dot", line_color=_GREEN,
                         annotation_text="SR = 1", annotation_font_color=_GREEN)
        fig_sr.add_hline(y=0, line_color=_RED, line_width=1)
        fig_sr.update_layout(**_layout(height=260))
        st.plotly_chart(fig_sr, use_container_width=True)

    with col_b:
        st.markdown("#### 🎯 Rolling Sortino Ratio (3-day)")
        st.caption("Penalises only downside deviation — higher is better")
        df_so = log_df[log_df["sortino"].notna()].copy()
        fig_so = go.Figure()
        fig_so.add_hrect(y0=1,  y1=50,  fillcolor="rgba(91,196,160,0.06)",  line_width=0)
        fig_so.add_hrect(y0=-50,y1=0,   fillcolor="rgba(224,112,112,0.06)", line_width=0)
        fig_so.add_trace(go.Scatter(
            x=df_so["date"], y=df_so["sortino"], mode="lines+markers",
            line=dict(color=_MUTED, width=2.5), marker=dict(size=9, color=_MUTED),
            hovertemplate="<b>%{x|%b %d}</b><br>Sortino: %{y:.2f}<extra></extra>"))
        fig_so.add_hline(y=1, line_dash="dot", line_color=_GREEN,
                         annotation_text="SR = 1", annotation_font_color=_GREEN)
        fig_so.add_hline(y=0, line_color=_RED, line_width=1)
        fig_so.update_layout(**_layout(height=260))
        st.plotly_chart(fig_so, use_container_width=True)

    # ── Row 2: Drawdown + Capital deployment ──────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### 📉 Drawdown from Peak (%)")
        st.caption(f"Max drawdown over backtest period: **{max_dd:.2f}%**")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=log_df["date"], y=dd_ser, mode="lines+markers",
            line=dict(color=_RED, width=2.5), marker=dict(size=8, color=_RED),
            fill="tozeroy", fillcolor="rgba(224,112,112,0.12)",
            hovertemplate="<b>%{x|%b %d}</b><br>DD: %{y:.2f}%<extra></extra>"))
        fig_dd.add_hline(y=0, line_color=_MUTED, line_width=1)
        fig_dd.update_layout(**_layout(height=260, ysuffix="%", yfmt=".2f"))
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_d:
        st.markdown("#### 💰 Capital Deployment (%)")
        st.caption(f"Invested fraction of portfolio each day · Avg {deployed_avg:.1f}%")
        fig_dep = go.Figure()
        fig_dep.add_trace(go.Bar(
            x=log_df["date"], y=log_df["deployed_pct"], name="Deployed %",
            marker_color=_ACCENT,
            hovertemplate="<b>%{x|%b %d}</b><br>Deployed: %{y:.1f}%<extra></extra>"))
        fig_dep.add_hline(y=70, line_dash="dot", line_color=_GREEN,
                          annotation_text="Target 70%", annotation_font_color=_GREEN,
                          annotation_position="right")
        fig_dep.update_layout(**_layout(height=260, ysuffix="%", yfmt=".0f"))
        st.plotly_chart(fig_dep, use_container_width=True)

    # ── Row 3: Rolling Calmar + Rolling Volatility ────────────────────────────
    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("#### 🏔️ Rolling Calmar Ratio (3-day)")
        st.caption("Ann. return ÷ max drawdown.  Higher = better return per unit of risk.")
        df_cal = log_df[log_df["calmar"].notna()].copy()
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=df_cal["date"], y=df_cal["calmar"], mode="lines+markers",
            line=dict(color=_GREEN, width=2.5), marker=dict(size=9, color=_GREEN),
            hovertemplate="<b>%{x|%b %d}</b><br>Calmar: %{y:.2f}<extra></extra>"))
        fig_cal.add_hline(y=0, line_color=_RED, line_width=1)
        fig_cal.update_layout(**_layout(height=260))
        st.plotly_chart(fig_cal, use_container_width=True)

    with col_f:
        st.markdown("#### 📊 Rolling Annualised Volatility (3-day, %)")
        st.caption("Higher = more volatile — ideal range for momentum is 15–35%")
        df_vol = log_df[log_df["vol_ann"].notna()].copy()
        fig_vol = go.Figure()
        fig_vol.add_hrect(y0=15, y1=35, fillcolor="rgba(91,196,160,0.06)", line_width=0)
        fig_vol.add_trace(go.Scatter(
            x=df_vol["date"], y=df_vol["vol_ann"], mode="lines+markers",
            line=dict(color=_AMBER, width=2.5), marker=dict(size=9, color=_AMBER),
            fill="tozeroy", fillcolor="rgba(200,168,75,0.08)",
            hovertemplate="<b>%{x|%b %d}</b><br>Vol: %{y:.1f}%<extra></extra>"))
        fig_vol.update_layout(**_layout(height=260, ysuffix="%", yfmt=".1f"))
        st.plotly_chart(fig_vol, use_container_width=True)

    # ── Row 4: Return distribution + Sharpe vs Sortino overlay ───────────────
    col_g, col_h = st.columns(2)
    with col_g:
        st.markdown("#### 🎲 Daily Return Distribution")
        st.caption("Each bar = one trading day  ·  Gold line = mean return")
        dr_pct = log_df["daily_ret"]
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(x=log_df["date"], y=dr_pct,
            marker_color=[_GREEN if v >= 0 else _RED for v in dr_pct],
            hovertemplate="<b>%{x|%b %d}</b><br>%{y:+.3f}%<extra></extra>"))
        mu_d = dr_pct[dr_pct != 0].mean()
        fig_dist.add_hline(y=0,    line_color=_MUTED, line_width=1)
        fig_dist.add_hline(y=mu_d, line_dash="dot", line_color=_AMBER,
                           annotation_text=f"Avg {mu_d:+.2f}%",
                           annotation_font_color=_AMBER, annotation_position="right")
        fig_dist.update_layout(**_layout(height=260, ysuffix="%", yfmt="+.2f"))
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_h:
        st.markdown("#### ⚡🎯 Sharpe vs Sortino — Side-by-Side")
        st.caption("Both annualised · Where Sortino > Sharpe the downside risk is lower than total vol")
        df_both = log_df[log_df["sharpe"].notna() & log_df["sortino"].notna()].copy()
        fig_both = go.Figure()
        fig_both.add_trace(go.Scatter(
            x=df_both["date"], y=df_both["sharpe"], mode="lines+markers", name="Sharpe",
            line=dict(color=_ACCENT, width=2.5), marker=dict(size=8, color=_ACCENT),
            hovertemplate="<b>%{x|%b %d}</b><br>Sharpe: %{y:.2f}<extra></extra>"))
        fig_both.add_trace(go.Scatter(
            x=df_both["date"], y=df_both["sortino"], mode="lines+markers", name="Sortino",
            line=dict(color=_MUTED, width=2.5, dash="dash"), marker=dict(size=8, color=_MUTED),
            hovertemplate="<b>%{x|%b %d}</b><br>Sortino: %{y:.2f}<extra></extra>"))
        fig_both.add_hline(y=1, line_dash="dot", line_color=_GREEN,
                           annotation_text="Ratio = 1", annotation_font_color=_GREEN)
        fig_both.add_hline(y=0, line_color=_RED, line_width=1)
        fig_both.update_layout(**_layout(height=260))
        st.plotly_chart(fig_both, use_container_width=True)

    # ── Rolling metrics table ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Rolling Metrics by Day")
    st.caption("3-day rolling window · — = insufficient data for that row")

    def _fmt(v, fmt):
        try:
            return fmt.format(float(v)) if v is not None and not np.isnan(float(v)) else "—"
        except Exception: return "—"

    tbl = log_df.copy()
    tbl["date"] = tbl["date"].dt.strftime("%Y-%m-%d")
    df_tbl = pd.DataFrame({
        "Date":           tbl["date"],
        "Portfolio":      tbl["portfolio"].apply(lambda x: f"${x:,.0f}"),
        "Daily Ret.":     tbl["daily_ret"].apply(lambda x: f"{x:+.3f}%"),
        "Cum. Ret.":      tbl["cum_ret"].apply(lambda x: f"{x:+.3f}%"),
        "Sharpe (3d)":    tbl["sharpe"].apply(lambda x: _fmt(x, "{:.2f}")),
        "Sortino (3d)":   tbl["sortino"].apply(lambda x: _fmt(x, "{:.2f}")),
        "Calmar (3d)":    tbl["calmar"].apply(lambda x: _fmt(x, "{:.2f}")),
        "Ann. Vol (3d)":  tbl["vol_ann"].apply(lambda x: _fmt(x, "{:.1f}%")),
        "Drawdown":       dd_ser.apply(lambda x: f"{x:.2f}%"),
        "Deployed":       tbl["deployed_pct"].apply(lambda x: f"{x:.1f}%"),
    })
    st.dataframe(df_tbl, use_container_width=True, hide_index=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    init_db(); migrate_db()
    sim = get_sim_state()
    render_sidebar(sim)

    st.markdown(f"""
    <div style="padding-bottom:4px;">
      <span style="font-size:30px;font-weight:800;color:{_FG};">📈 Group B</span>
      <span style="font-size:30px;font-weight:300;color:{_ACCENT};"> — AI Investment Portfolio</span>
    </div>
    <div style="color:{_MUTED};font-size:13px;letter-spacing:0.03em;margin-bottom:4px;">
      Paper Trading Simulator &nbsp;·&nbsp; MiF Deep Learning Competition &nbsp;·&nbsp; March 2026
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if sim is None:
        st.warning("No simulation found — initialising…")
        reset_db(STARTING_CAPITAL, COMPETITION_START); st.rerun(); return

    delta_pct = (sim["total_value"]-STARTING_CAPITAL)/STARTING_CAPITAL
    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("💰 Total Portfolio Value", format_dollar(sim["total_value"]),
                        delta=format_pct(delta_pct)+" vs start",
                        delta_color="normal" if delta_pct>=0 else "inverse")
    with m2:
        cp = sim["cash_balance"]/sim["total_value"] if sim["total_value"] else 0
        st.metric("💵 Cash Available", format_dollar(sim["cash_balance"]),
                  delta=f"{cp*100:.1f}% of portfolio")
    with m3: st.metric("📊 Open Positions", str(len(get_positions())))
    with m4:
        d = sim.get("day_number",0)
        st.metric("📅 Competition Day", f"Day {d}", delta=f"{max(0,11-d)} days remaining")
    st.divider()

    all_tickers = list(dict.fromkeys(ASSETS+[BENCHMARK]))
    prices_df   = fetch_prices(all_tickers, lookback_days=60)
    market_snap = get_market_snapshot(prices_df[[a for a in ASSETS if a in prices_df.columns]])
    if os.path.exists("screener_results.csv"):
        try: st.session_state["scores_df"] = pd.read_csv("screener_results.csv")
        except Exception: pass

    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "🏛️ Trading Floor","💼 Portfolio","📈 Performance",
        "🔭 Overview","📊 Backtest","📉 Backtest Performance"
    ])
    with tab1: render_trading_floor(sim, market_snap, prices_df, st.session_state.get("scores_df"))
    with tab2: render_portfolio(sim, market_snap)
    with tab3: render_performance(sim, prices_df)
    with tab4: render_overview(sim, prices_df)
    with tab5: render_backtest()
    with tab6: render_backtest_performance()


if __name__ == "__main__":
    main()
