import os, sys, time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime

# All files are in the same directory as app.py
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data", "Kore2_battery_performance.csv")

# Load models directly from root
def load_model():
    return joblib.load(os.path.join(BASE, "rf_classifier.pkl"))

def load_kmeans():
    return joblib.load(os.path.join(BASE, "kmeans_model.pkl"))

st.set_page_config(page_title="KOFA Battery SOH Monitor", page_icon="🛵", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0d1b2a 100%);
        border-right: 2px solid #f7b731;
    }
    .faulty-card {
        background: linear-gradient(135deg, #3a0000, #5a0000);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
    }
    .healthy-card {
        background: linear-gradient(135deg, #003a00, #005a00);
        border: 2px solid #44ff44;
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
    }
    .page-title {
        font-size: 2.2em;
        font-weight: bold;
        color: #f7b731;
        border-bottom: 3px solid #f7b731;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .kofa-header {
        background: linear-gradient(90deg, #f7b731, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8em;
        font-weight: 900;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e4a, #2a2a6a);
        border: 1px solid #f7b731;
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="metric-container"] label { color: #f7b731 !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: white !important; font-size: 1.8em !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #f7b731, #ff6b35);
        color: black; font-weight: bold; border: none;
        border-radius: 8px; padding: 10px 25px; font-size: 1em;
    }
    .filter-box {
        background: linear-gradient(135deg, #1e1e4a, #2a2a6a);
        border: 1px solid #f7b731;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #f7b731 !important; }
    p, li { color: #ddd !important; }
</style>
""", unsafe_allow_html=True)

FAULTY_COLOR  = "#ff4444"
HEALTHY_COLOR = "#44cc44"

# Battery lists
ORIGINAL_FAULTY  = ["KF-B101", "KF-I108", "KF-J109", "KF-L111"]
ORIGINAL_HEALTHY = ["KF-A100", "KF-C102", "KF-D103", "KF-E104",
                    "KF-F105", "KF-G106", "KF-H107", "KF-K110"]
NEW_FAULTY  = ["KF-M112", "KF-N113", "KF-O114"]
NEW_HEALTHY = ["KF-P115", "KF-Q116", "KF-R117", "KF-S118", "KF-T119"]
ALL_FAULTY   = ORIGINAL_FAULTY + NEW_FAULTY
ALL_HEALTHY  = ORIGINAL_HEALTHY + NEW_HEALTHY
ALL_BATTERIES = ALL_HEALTHY + ALL_FAULTY

SCAN_DAYS = ["2024-01-15","2024-01-16","2024-01-17","2024-01-18",
             "2024-01-19","2024-01-20","2024-01-21"]

# ── Core functions ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading telemetry ...")
def load_data():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    return df.sort_values(["battery_id","timestamp"]).reset_index(drop=True)

@st.cache_data(show_spinner="Cleaning data ...")
def clean_data(df):
    cols = ["speed_kph","voltage_V","current_A","soc_percent","state_of_health","ambient_temp_C"]
    dc = df.copy()
    dc[cols] = (dc.groupby("battery_id")[cols]
                  .apply(lambda g: g.interpolate(method="linear", limit_direction="both"))
                  .reset_index(level=0, drop=True))
    return dc

@st.cache_data(show_spinner="Running fleet diagnostics ...")
def run_screen(df):
    num_cols = ["speed_kph","voltage_V","current_A","soc_percent","state_of_health","ambient_temp_C"]
    dc = df.copy()
    dc[num_cols] = (dc.groupby("battery_id")[num_cols]
                      .apply(lambda g: g.interpolate(method="linear", limit_direction="both"))
                      .reset_index(level=0, drop=True))
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    recs = []
    for bid, g in dc.groupby("battery_id"):
        ride = g[g["state"]=="in_use"].dropna(subset=["soc_percent"]).sort_values("timestamp")
        if len(ride) < 3: continue
        t_hr = (ride["timestamp"]-ride["timestamp"].iloc[0]).dt.total_seconds()/3600
        recs.append(dict(
            battery_id=bid,
            drain_rate_pct_per_hr=-np.polyfit(t_hr, ride["soc_percent"],1)[0],
            ride_duration_min=(ride["timestamp"].iloc[-1]-ride["timestamp"].iloc[0]).total_seconds()/60,
            soc_at_shutdown=ride["soc_percent"].iloc[-1],
            voltage_std_inuse=ride["voltage_V"].std(),
            soh_pct=g["state_of_health"].mean(),
        ))
    out = pd.DataFrame(recs)
    feats = ["drain_rate_pct_per_hr","ride_duration_min","soc_at_shutdown","voltage_std_inuse","soh_pct"]
    Xs = StandardScaler().fit_transform(out[feats])
    out["cluster"] = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(Xs)
    bad = out.groupby("cluster")["drain_rate_pct_per_hr"].mean().idxmax()
    out["flagged"] = out["cluster"] == bad
    return out.sort_values("drain_rate_pct_per_hr", ascending=False).reset_index(drop=True)

def generate_live_reading(battery_id, seed=None, day_index=0):
    rng = np.random.default_rng(seed)
    is_faulty = battery_id in ALL_FAULTY
    deg = day_index * 0.3
    if is_faulty:
        speed=max(0,rng.normal(34,3)); voltage=rng.normal(45.0-deg*0.1,0.8)
        current=rng.normal(30+deg*0.1,2); soc=max(0,min(100,rng.normal(20-deg*0.2,3)))
        soh=rng.normal(67-deg*0.2,1); temp=rng.normal(28,1)
    else:
        speed=max(0,rng.normal(35,3)); voltage=rng.normal(51.0,0.3)
        current=rng.normal(17,1); soc=max(0,min(100,rng.normal(75,5)))
        soh=rng.normal(91,1); temp=rng.normal(27,1)
    return {"battery_id":battery_id,"speed_kph":round(speed,2),"voltage_V":round(voltage,2),
            "current_A":round(current,2),"soc_percent":round(soc,2),
            "state_of_health":round(soh,2),"ambient_temp_C":round(temp,2)}

def score_live_fleet(seed=None, day_index=0, scan_date=None):
    clf = load_model()
    features = ["speed_kph","voltage_V","current_A","soc_percent","ambient_temp_C"]
    rng_seed = seed if seed is not None else np.random.randint(0,99999)
    if scan_date is None: scan_date = SCAN_DAYS[day_index % len(SCAN_DAYS)]
    records = []
    for i, bid in enumerate(ALL_BATTERIES):
        reading = generate_live_reading(bid, seed=rng_seed+i, day_index=day_index)
        X = pd.DataFrame([{f:reading[f] for f in features}])
        pred = clf.predict(X)[0]; proba = clf.predict_proba(X)[0]
        records.append({"scan_date":scan_date,"battery_id":bid,"speed_kph":reading["speed_kph"],
                        "voltage_V":reading["voltage_V"],"current_A":reading["current_A"],
                        "soc_percent":reading["soc_percent"],"soh_percent":reading["state_of_health"],
                        "ambient_temp_C":reading["ambient_temp_C"],
                        "prediction":"Faulty" if pred==1 else "Healthy",
                        "confidence":round(float(proba[pred]),3),"flagged":bool(pred==1)})
    return pd.DataFrame(records).sort_values("flagged",ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner="Generating 7-day simulation ...")
def get_all_days():
    all_records = []
    for i, day in enumerate(SCAN_DAYS):
        results = score_live_fleet(seed=1000+i*100, day_index=i, scan_date=day)
        all_records.append(results)
    return pd.concat(all_records, ignore_index=True)

def score_reading(speed_kph, voltage_V, current_A, soc_percent, ambient_temp_C):
    clf = load_model()
    X = pd.DataFrame([{"speed_kph":speed_kph,"voltage_V":voltage_V,
                        "current_A":current_A,"soc_percent":soc_percent,
                        "ambient_temp_C":ambient_temp_C}])
    pred = clf.predict(X)[0]; proba = clf.predict_proba(X)[0]
    return {"prediction":"Faulty" if pred==1 else "Healthy","probability":round(float(proba[pred]),3)}

def send_alert(battery_id, soh, soc, faulty=True):
    try:
        from config import AT_USERNAME, AT_API_KEY, RIDER_PHONES
        import africastalking
        africastalking.initialize(AT_USERNAME, AT_API_KEY)
        sms = africastalking.SMS
        phone = RIDER_PHONES.get(battery_id, "+254797804812")
        if faulty:
            msg = "KOFA ALERT\nBattery "+battery_id+" is FAULTY.\nSOH: "+str(round(soh,1))+"% SOC: "+str(round(soc,1))+"%\nReturn to KOFA station immediately."
        else:
            msg = "KOFA CHECK\nBattery "+battery_id+" is HEALTHY.\nSOH: "+str(round(soh,1))+"%\nRide safe! - KOFA Team"
        result = sms.send(msg, [phone])
        return {"status":"sent","phone":phone,"response":result}
    except Exception as e:
        return {"status":"error","message":str(e)}

# Load data
df       = load_data()
df_clean = clean_data(df)
summary  = run_screen(df)

temp_means = df_clean.groupby("battery_id")["ambient_temp_C"].mean().round(1).reset_index()
temp_means.columns = ["battery_id","temp_mean_C"]
summary = summary.merge(temp_means, on="battery_id", how="left")

flagged_ids = summary.loc[summary["flagged"],"battery_id"].tolist()
healthy_ids = summary.loc[~summary["flagged"],"battery_id"].tolist()
all_days_data = get_all_days()

# SIDEBAR
with st.sidebar:
    st.markdown('<div class="kofa-header">🛵 KOFA SOH</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#aaa;font-size:0.8em;">Kore2 Fleet · CRISP-DM Deployment</p>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Go to", [
        "🏠 Fleet Overview","🔋 Battery Drill-Down",
        "📡 Live Monitor","⚡ Live Scorer","📋 Data Table"
    ], label_visibility="collapsed")
    st.markdown("---")
    f1,f2 = st.columns(2)
    f1.metric("Total", len(summary))
    f2.metric("🔴 Faulty", len(flagged_ids))
    f3,f4 = st.columns(2)
    f3.metric("🟢 Healthy", len(healthy_ids))
    f4.metric("Accuracy", "99.6%")
    st.markdown("---")
    st.markdown('<p style="color:#aaa;font-size:0.75em;">20 Batteries · 7 Days · RF + KMeans</p>', unsafe_allow_html=True)

# PAGE 1 — Fleet Overview
if page == "🏠 Fleet Overview":
    st.markdown('<div class="page-title">🛵 KOFA Kore2 Fleet Health Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#aaa;">Real-time diagnostics · K-Means + Random Forest · CRISP-DM Phase 6</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🔋 Fleet Size", f"{len(summary)} batteries")
    c2.metric("🔴 Faulty", f"{len(flagged_ids)}", delta=f"{len(flagged_ids)/len(summary):.0%} of fleet", delta_color="inverse")
    c3.metric("⚡ Avg Faulty Drain", f"{summary[summary.flagged].drain_rate_pct_per_hr.mean():.1f} %/hr", delta="High", delta_color="inverse")
    c4.metric("💚 Avg Healthy SOH", f"{summary[~summary.flagged].soh_pct.mean():.1f}%")
    st.markdown("---")
    col1,col2 = st.columns([1.4,1])
    with col1:
        st.markdown("### 🔋 Drain Rate per Battery")
        plot_df = summary.sort_values("drain_rate_pct_per_hr",ascending=False).copy()
        plot_df["Status"] = plot_df["flagged"].map({True:"⚠️ Faulty",False:"✅ Healthy"})
        healthy_only = summary[~summary.flagged]["drain_rate_pct_per_hr"]
        control_limit = healthy_only.mean()+2*healthy_only.std()
        fig = px.bar(plot_df,x="battery_id",y="drain_rate_pct_per_hr",color="Status",
                     color_discrete_map={"⚠️ Faulty":FAULTY_COLOR,"✅ Healthy":HEALTHY_COLOR},
                     labels={"drain_rate_pct_per_hr":"Drain Rate (%/hr)","battery_id":"Battery"})
        fig.add_hline(y=control_limit,line_dash="dash",line_color="#f7b731",
                      annotation_text=f"Safety limit: {control_limit:.0f} %/hr",annotation_font_color="#f7b731")
        fig.update_layout(plot_bgcolor="#0f0f23",paper_bgcolor="#0f0f23",font_color="white",height=380,legend=dict(font=dict(color="white")))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("### 📊 SOH vs Drain Rate")
        scatter_df = summary.copy()
        scatter_df["Status"] = scatter_df["flagged"].map({True:"⚠️ Faulty",False:"✅ Healthy"})
        fig2 = px.scatter(scatter_df,x="drain_rate_pct_per_hr",y="soh_pct",color="Status",text="battery_id",
                          color_discrete_map={"⚠️ Faulty":FAULTY_COLOR,"✅ Healthy":HEALTHY_COLOR},
                          labels={"drain_rate_pct_per_hr":"Drain Rate (%/hr)","soh_pct":"SOH (%)"})
        fig2.update_traces(textposition="top center",marker_size=12)
        fig2.update_layout(plot_bgcolor="#0f0f23",paper_bgcolor="#0f0f23",font_color="white",height=380,legend=dict(font=dict(color="white")))
        st.plotly_chart(fig2,use_container_width=True)
    st.markdown("### 📉 SOC Discharge Trajectories")
    fig3 = go.Figure()
    for bid,g in df_clean.groupby("battery_id"):
        ride = g[g["state"]=="in_use"].dropna(subset=["soc_percent"]).sort_values("timestamp")
        if ride.empty: continue
        t_min = (ride["timestamp"]-df_clean["timestamp"].min()).dt.total_seconds()/60
        is_bad = bid in flagged_ids
        fig3.add_trace(go.Scatter(x=t_min,y=ride["soc_percent"],mode="lines",name=bid,
                                   line=dict(color=FAULTY_COLOR if is_bad else HEALTHY_COLOR,width=3 if is_bad else 1.5),
                                   opacity=1.0 if is_bad else 0.6))
    fig3.update_layout(xaxis_title="Minutes into ride",yaxis_title="SOC (%)",
                       plot_bgcolor="#0f0f23",paper_bgcolor="#0f0f23",font_color="white",height=420,
                       legend=dict(orientation="h",font=dict(size=10,color="white"),yanchor="bottom",y=1.02,xanchor="left",x=0))
    st.plotly_chart(fig3,use_container_width=True)
    st.markdown("---")
    st.markdown("### ⚠️ Immediate Action Required")
    for bid in flagged_ids:
        row = summary[summary.battery_id==bid].iloc[0]
        st.markdown(f'<div class="faulty-card"><b style="color:#ff4444;font-size:1.1em;">🔴 {bid} — PULL FROM SERVICE</b><br><span style="color:#ddd;">Drain: <b style="color:#f7b731;">{row.drain_rate_pct_per_hr:.1f} %/hr</b> | SOH: <b style="color:#f7b731;">{row.soh_pct:.1f}%</b> | Shutdown SOC: <b style="color:#f7b731;">{row.soc_at_shutdown:.1f}%</b></span></div>', unsafe_allow_html=True)

# PAGE 2 — Battery Drill-Down
elif page == "🔋 Battery Drill-Down":
    st.markdown('<div class="page-title">🔋 Individual Battery Analysis</div>', unsafe_allow_html=True)
    bid = st.selectbox("Select Battery ID", summary["battery_id"].tolist())
    row = summary[summary.battery_id==bid].iloc[0]
    g   = df_clean[df_clean.battery_id==bid].copy()
    if row.flagged:
        st.markdown(f'<div class="faulty-card"><b style="color:#ff4444;font-size:1.3em;">🔴 {bid} — FAULTY · Pull from service immediately</b></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="healthy-card"><b style="color:#44ff44;font-size:1.3em;">🟢 {bid} — HEALTHY · Safe for next trip</b></div>', unsafe_allow_html=True)
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("⚡ Drain Rate", f"{row.drain_rate_pct_per_hr:.1f} %/hr")
    m2.metric("⏱ Ride Duration", f"{row.ride_duration_min:.0f} min")
    m3.metric("🔋 Shutdown SOC", f"{row.soc_at_shutdown:.1f}%")
    m4.metric("📊 Voltage Std", f"{row.voltage_std_inuse:.2f} V")
    m5.metric("💚 SOH", f"{row.soh_pct:.1f}%")
    st.markdown("---")
    ride  = g[g["state"]=="in_use"].sort_values("timestamp")
    color = FAULTY_COLOR if row.flagged else HEALTHY_COLOR
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### 📉 SOC While Riding")
        fig = px.line(ride,x="timestamp",y="soc_percent",color_discrete_sequence=[color],labels={"soc_percent":"SOC (%)","timestamp":"Time"})
        fig.update_layout(plot_bgcolor="#0f0f23",paper_bgcolor="#0f0f23",font_color="white")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown("### ⚡ Voltage Under Load")
        fig2 = px.line(ride,x="timestamp",y="voltage_V",color_discrete_sequence=[color],labels={"voltage_V":"Voltage (V)","timestamp":"Time"})
        fig2.update_layout(plot_bgcolor="#0f0f23",paper_bgcolor="#0f0f23",font_color="white")
        st.plotly_chart(fig2,use_container_width=True)
    st.markdown("### 🏍️ Speed & Current")
    fig3 = px.line(ride,x="timestamp",y=["speed_kph","current_A"],color_discrete_sequence=["#f7b731","#ff6b35"],labels={"value":"Reading","timestamp":"Time","variable":"Sensor"})
    fig3.update_layout(plot_bgcolor="#0f0f23",paper_bgcolor="#0f0f23",font_color="white",legend=dict(font=dict(color="white")))
    st.plotly_chart(fig3,use_container_width=True)

# PAGE 3 — Live Monitor
elif page == "📡 Live Monitor":
    st.markdown('<div class="page-title">📡 Live Fleet Monitor</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#aaa;">20 batteries scored across 7 days by the trained Random Forest model.</p>', unsafe_allow_html=True)
    col_btn,col_sms = st.columns([1,1])
    send_sms = col_sms.checkbox("📲 Send SMS Alerts", value=True)
    run_scan = col_btn.button("▶️ Run Live Scan", use_container_width=True)
    st.markdown("---")
    if run_scan:
        with st.spinner("Scanning 20 batteries ..."):
            seed = int(time.time()); day_index = seed % 7
            scan_date = SCAN_DAYS[day_index]
            results = score_live_fleet(seed=seed, day_index=day_index, scan_date=scan_date)
        faulty_now = results[results.flagged]; healthy_now = results[~results.flagged]
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("🔋 Scanned", len(results)); k2.metric("🔴 Faulty", len(faulty_now))
        k3.metric("🟢 Healthy", len(healthy_now)); k4.metric("📅 Date", scan_date)
        st.markdown("---")
        st.markdown("### 🔍 Filter Results")
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        fc1,fc2,fc3 = st.columns(3)
        status_filter  = fc1.selectbox("Status", ["All","Faulty Only","Healthy Only"])
        voltage_filter = fc2.slider("Min Voltage (V)", 40.0, 55.0, 40.0, 0.5)
        soc_filter     = fc3.slider("Min SOC (%)", 0.0, 100.0, 0.0, 1.0)
        st.markdown('</div>', unsafe_allow_html=True)
        filtered = results.copy()
        if status_filter == "Faulty Only": filtered = filtered[filtered.flagged]
        elif status_filter == "Healthy Only": filtered = filtered[~filtered.flagged]
        filtered = filtered[filtered.voltage_V >= voltage_filter]
        filtered = filtered[filtered.soc_percent >= soc_filter]
        st.markdown(f"### 📊 Results — {len(filtered)} batteries")
        for _,row in filtered.iterrows():
            icon = "🔴" if row.flagged else "🟢"; status = "FAULTY" if row.flagged else "HEALTHY"
            a,b,c,d,e,f = st.columns([1,1,1,1,1,1])
            a.markdown(f"**{icon} {row.battery_id}**"); b.markdown(f"V: `{row.voltage_V}V`")
            c.markdown(f"SOC: `{row.soc_percent:.1f}%`"); d.markdown(f"SOH: `{row.soh_percent:.1f}%`")
            e.markdown(f"**{status}**"); f.markdown(f"Conf: `{row.confidence:.0%}`")
            st.markdown("---")
        st.markdown("### ⚡ Live Voltage")
        filtered["Status"] = filtered["flagged"].map({True:"⚠️ Faulty",False:"✅ Healthy"})
        fig = px.bar(filtered.sort_values("voltage_V"),x="battery_id",y="voltage_V",color="Status",
                     color_discrete_map={"⚠️ Faulty":FAULTY_COLOR,"✅ Healthy":HEALTHY_COLOR},
                     labels={"voltage_V":"Voltage (V)","battery_id":"Battery"})
        fig.add_hline(y=49.0,line_dash="dash",line_color="#f7b731",annotation_text="Min healthy (49V)",annotation_font_color="#f7b731")
        fig.update_layout(plot_bgcolor="#0f0f23",paper_bgcolor="#0f0f23",font_color="white",height=400,legend=dict(font=dict(color="white")))
        st.plotly_chart(fig,use_container_width=True)
        if not faulty_now.empty:
            st.markdown("### 📲 SMS Alerts")
            for _,row in faulty_now.iterrows():
                st.markdown(f'<div class="faulty-card"><b style="color:#ff4444;">🔴 {row.battery_id} — FAULTY</b><br><span style="color:#ddd;">V: {row.voltage_V}V | SOC: {row.soc_percent:.1f}% | SOH: {row.soh_percent:.1f}% | Conf: {row.confidence:.0%}</span></div>', unsafe_allow_html=True)
                if send_sms:
                    with st.spinner(f"Sending SMS for {row.battery_id} ..."):
                        r = send_alert(row.battery_id, soh=row.soh_percent, soc=row.soc_percent, faulty=True)
                    if r["status"] == "sent": st.success(f"✅ SMS sent to {r['phone']}")
                    else: st.warning(f"⚠️ {r['message']}")
        else:
            st.markdown('<div class="healthy-card"><b style="color:#44ff44;">✅ All batteries healthy</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:linear-gradient(135deg,#1e1e4a,#2a2a6a);border:1px solid #f7b731;border-radius:12px;padding:30px;text-align:center;"><div style="font-size:3em;">🛵</div><h3 style="color:#f7b731;">Ready to Scan 20 Batteries</h3><p style="color:#aaa;">Click ▶️ Run Live Scan to score all 20 batteries.</p></div>', unsafe_allow_html=True)

# PAGE 4 — Live Scorer
elif page == "⚡ Live Scorer":
    st.markdown('<div class="page-title">⚡ Live Battery Scorer</div>', unsafe_allow_html=True)
    with st.form("live_form"):
        c1,c2,c3 = st.columns(3)
        speed   = c1.number_input("🏍️ Speed (kph)",   value=35.0, min_value=0.0,  max_value=120.0, step=0.5)
        voltage = c2.number_input("⚡ Voltage (V)",    value=50.5, min_value=40.0, max_value=60.0,  step=0.1)
        current = c3.number_input("🔌 Current (A)",    value=18.0, min_value=0.0,  max_value=60.0,  step=0.5)
        soc     = c1.number_input("🔋 SOC (%)",        value=80.0, min_value=0.0,  max_value=100.0, step=0.5)
        temp    = c2.number_input("🌡️ Temp (°C)",      value=27.0, min_value=15.0, max_value=50.0,  step=0.5)
        bid_sel = c3.selectbox("🏷️ Battery ID",       options=ALL_BATTERIES)
        send_sms_single = st.checkbox("📲 Send SMS", value=True)
        submitted = st.form_submit_button("🔍 Classify Battery", use_container_width=True)
    if submitted:
        result = score_reading(speed, voltage, current, soc, temp)
        soh_match = summary[summary.battery_id==bid_sel]
        soh_est = soh_match["soh_pct"].values[0] if len(soh_match) > 0 else 80.0
        if result["prediction"] == "Faulty":
            st.markdown(f'<div class="faulty-card"><h2 style="color:#ff4444;">🔴 FAULTY BATTERY</h2><p style="color:#ddd;">Battery <b>{bid_sel}</b> | Confidence: <b style="color:#f7b731;">{result["probability"]:.1%}</b></p><p style="color:#ff8888;">⚠️ Do NOT issue to rider.</p></div>', unsafe_allow_html=True)
            if send_sms_single:
                with st.spinner("Sending SMS ..."):
                    r = send_alert(bid_sel, soh=soh_est, soc=soc, faulty=True)
                if r["status"] == "sent": st.success(f"✅ SMS sent to {r['phone']}")
                else: st.warning(f"⚠️ {r['message']}")
        else:
            st.markdown(f'<div class="healthy-card"><h2 style="color:#44ff44;">🟢 BATTERY HEALTHY</h2><p style="color:#ddd;">Battery <b>{bid_sel}</b> | Confidence: <b style="color:#f7b731;">{result["probability"]:.1%}</b></p><p style="color:#88ff88;">✅ Safe to issue to rider.</p></div>', unsafe_allow_html=True)
            if send_sms_single:
                with st.spinner("Sending SMS ..."):
                    r = send_alert(bid_sel, soh=soh_est, soc=soc, faulty=False)
                if r["status"] == "sent": st.success(f"✅ SMS sent to {r['phone']}")
                else: st.warning(f"⚠️ {r['message']}")
    st.markdown("---")
    rc1,rc2 = st.columns(2)
    with rc1: st.markdown('<div class="healthy-card"><b style="color:#44ff44;">✅ Healthy</b><br><span style="color:#ddd;">Voltage: 50–52V | Current: 15–20A | SOC > 60%</span></div>', unsafe_allow_html=True)
    with rc2: st.markdown('<div class="faulty-card"><b style="color:#ff4444;">🔴 Faulty</b><br><span style="color:#ddd;">Voltage: 44–47V | Current: 25–32A | SOC < 30%</span></div>', unsafe_allow_html=True)

# PAGE 5 — Data Table
elif page == "📋 Data Table":
    st.markdown('<div class="page-title">📋 Fleet Data Table</div>', unsafe_allow_html=True)
    data_source = st.radio("Data Source", ["Training Data (12 batteries)","Simulated Live Data (20 batteries × 7 days)"], horizontal=True)
    st.markdown("---")
    if data_source == "Training Data (12 batteries)":
        display_df = summary.copy()
        display_df["status"] = display_df["flagged"].map({True:"🔴 FAULTY",False:"🟢 Healthy"})
        display_df["scan_date"] = "2024-01-15"
        st.markdown("### 🔍 Filters")
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        f1,f2,f3,f4 = st.columns(4)
        status_f  = f1.selectbox("🔴 Status", ["All","Faulty Only","Healthy Only"])
        battery_f = f2.multiselect("🔋 Battery", options=display_df["battery_id"].tolist(), default=display_df["battery_id"].tolist())
        drain_f   = f3.slider("⚡ Max Drain (%/hr)", 0.0, 30.0, 30.0, 0.5)
        soh_f     = f4.slider("💚 Min SOH (%)", 0.0, 100.0, 0.0, 1.0)
        st.markdown('</div>', unsafe_allow_html=True)
        if status_f == "Faulty Only": display_df = display_df[display_df.flagged]
        elif status_f == "Healthy Only": display_df = display_df[~display_df.flagged]
        if battery_f: display_df = display_df[display_df.battery_id.isin(battery_f)]
        display_df = display_df[display_df.drain_rate_pct_per_hr <= drain_f]
        display_df = display_df[display_df.soh_pct >= soh_f]
        st.markdown(f"### Showing {len(display_df)} batteries")
        st.dataframe(display_df[["battery_id","status","scan_date","drain_rate_pct_per_hr","ride_duration_min","soc_at_shutdown","voltage_std_inuse","soh_pct"]], use_container_width=True, hide_index=True)
    else:
        display_df = all_days_data.copy()
        display_df["status"] = display_df["flagged"].map({True:"🔴 FAULTY",False:"🟢 Healthy"})
        st.markdown("### 🔍 Filters")
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        f1,f2,f3,f4 = st.columns(4)
        date_options = sorted(display_df["scan_date"].unique())
        date_f    = f1.multiselect("📅 Date", options=date_options, default=date_options)
        status_f  = f2.selectbox("🔴 Status", ["All","Faulty Only","Healthy Only"])
        battery_f = f3.multiselect("🔋 Battery", options=sorted(display_df["battery_id"].unique()), default=sorted(display_df["battery_id"].unique()))
        soh_f     = f4.slider("💚 Min SOH (%)", 0.0, 100.0, 0.0, 1.0)
        st.markdown('</div>', unsafe_allow_html=True)
        if date_f: display_df = display_df[display_df["scan_date"].isin(date_f)]
        if status_f == "Faulty Only": display_df = display_df[display_df.flagged]
        elif status_f == "Healthy Only": display_df = display_df[~display_df.flagged]
        if battery_f: display_df = display_df[display_df.battery_id.isin(battery_f)]
        display_df = display_df[display_df.soh_percent >= soh_f]
        st.markdown(f"### Showing {len(display_df)} records")
        st.dataframe(display_df[["scan_date","battery_id","status","voltage_V","soc_percent","soh_percent","prediction","confidence"]], use_container_width=True, hide_index=True)
        if not display_df.empty:
            st.markdown("### 📈 Fault Trend by Day")
            trend = display_df.groupby(["scan_date","prediction"]).size().reset_index(name="count")
            fig = px.line(trend,x="scan_date",y="count",color="prediction",
                          color_discrete_map={"Faulty":FAULTY_COLOR,"Healthy":HEALTHY_COLOR},
                          labels={"scan_date":"Date","count":"Batteries","prediction":"Status"},markers=True)
            fig.update_layout(plot_bgcolor="#0f0f23",paper_bgcolor="#0f0f23",font_color="white",legend=dict(font=dict(color="white")))
            st.plotly_chart(fig,use_container_width=True)
    col1,col2 = st.columns(2)
    csv = display_df.to_csv(index=False)
    col1.download_button("⬇️ Download Filtered Report", data=csv, file_name="kofa_filtered_report.csv", mime="text/csv", use_container_width=True)
    full_csv = all_days_data.to_csv(index=False)
    col2.download_button("⬇️ Download Full 7-Day Report", data=full_csv, file_name="kofa_full_7day_report.csv", mime="text/csv", use_container_width=True)
