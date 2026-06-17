import streamlit as st
import pandas as pd
import requests
import json
import os

BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Game Ops Intelligence System v2",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #1f6feb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #388bfd;
        color: white;
    }
    .metric-card {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f6feb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .zone-clean {
        color: #2ea043;
        font-weight: bold;
    }
    .zone-watch {
        color: #58a6ff;
        font-weight: bold;
    }
    .zone-review {
        color: #d29922;
        font-weight: bold;
    }
    .zone-restricted {
        color: #db6d28;
        font-weight: bold;
    }
    .zone-flagged {
        color: #f85149;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get zone color classes
def get_zone_html(zone):
    z = zone.lower()
    if z == "clean":
        return f'<span class="zone-clean">{zone.upper()}</span>'
    elif z == "watch":
        return f'<span class="zone-watch">{zone.upper()}</span>'
    elif z == "review":
        return f'<span class="zone-review">{zone.upper()}</span>'
    elif z == "restricted":
        return f'<span class="zone-restricted">{zone.upper()}</span>'
    elif z == "flagged":
        return f'<span class="zone-flagged">{zone.upper()}</span>'
    return zone

st.sidebar.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard Overview", "Leaderboard Explorer", "Matchmaking Pools", "Suspicious Database", "Submit Player Score"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Backend API Status:**")
try:
    health_r = requests.get(f"{BASE_URL}/stats")
    if health_r.status_code == 200:
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.warning("🟡 API Connecting...")
except Exception:
    st.sidebar.error("🔴 API Offline")

st.sidebar.markdown("---")
st.sidebar.caption("Game Ops System v2 (Upgrade)")
st.sidebar.caption("Built with FastAPI, Streamlit, and Scikit-Learn")

# -------------------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# -------------------------------------------------------------
if page == "Dashboard Overview":
    st.title("📊 Game Operations Intelligence Dashboard")
    st.markdown("Real-time telemetry, anomaly distribution, and system-wide statistics.")
    
    try:
        r = requests.get(f"{BASE_URL}/stats")
        if r.status_code == 200:
            stats = r.json()
            
            # Key Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #58a6ff;">
                    <p style="color: #8b949e; margin: 0;">Total Players Analyzed</p>
                    <h2 style="margin: 5px 0;">{stats.get('total_players', 0):,}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #2ea043;">
                    <p style="color: #8b949e; margin: 0;">Clean / Watch Players</p>
                    <h2 style="margin: 5px 0;">{stats.get('clean_players', 0):,}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #f85149;">
                    <p style="color: #8b949e; margin: 0;">Banned & Restricted</p>
                    <h2 style="margin: 5px 0;">{stats.get('flagged_count', 0):,}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #d29922;">
                    <p style="color: #8b949e; margin: 0;">Top Region (By Avg Score)</p>
                    <h2 style="margin: 5px 0;">{stats.get('top_region_by_avg_score', 'N/A')}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("### Regional & Device Breakdown")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Players by Region**")
                reg_df = pd.DataFrame(list(stats.get('region_breakdown', {}).items()), columns=["Region", "Count"])
                st.bar_chart(reg_df.set_index("Region"))
            with c2:
                st.markdown("**Players by Device**")
                dev_df = pd.DataFrame(list(stats.get('device_breakdown', {}).items()), columns=["Device", "Count"])
                st.bar_chart(dev_df.set_index("Device"))
                
        else:
            st.error("Failed to load statistics from API.")
    except Exception as e:
        st.error(f"Could not connect to API server: {e}. Please ensure FastAPI is running.")
        
    st.markdown("---")
    st.markdown("### Pipeline Visualization Report")
    
    # Locate visualization v2 image
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "outputs", "game_ops_analysis_v2.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Game Operations Pipeline Performance, Security Zones, and Model Evaluations (V2)", use_container_width=True)
    else:
        st.info("Visual report 'game_ops_analysis_v2.png' not found. Run main_pipeline.py to generate it.")

# -------------------------------------------------------------
# PAGE 2: LEADERBOARD EXPLORER
# -------------------------------------------------------------
elif page == "Leaderboard Explorer":
    st.title("🏆 Leaderboard Explorer")
    st.markdown("Displays all clean and monitored (Watch) players sorted globally or by region. Banned or suspended players are automatically excluded.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        region = st.selectbox("Filter by Region", ["All", "India", "SEA", "Europe", "NA", "LatAm", "Middle_East"])
        limit = st.slider("Limit Rows", 10, 100, 20)
    
    try:
        url = f"{BASE_URL}/leaderboard?limit={limit}"
        if region != "All":
            url += f"&region={region}"
            
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            leaderboard = data.get("leaderboard", [])
            
            if leaderboard:
                df_lb = pd.DataFrame(leaderboard)
                
                # Format Columns
                df_lb_show = pd.DataFrame({
                    "Global Rank": df_lb["global_rank"],
                    "Region Rank": df_lb["region_rank"],
                    "Player ID": df_lb["player_id"],
                    "Region": df_lb["region"],
                    "Score": df_lb["score"],
                    "Kills": df_lb["kills"],
                    "Deaths": df_lb["deaths"],
                    "Skill Tier": df_lb["skill_tier"],
                    "Match Group": df_lb["match_group_id"].fillna("N/A"),
                    "Confidence Score": df_lb["confidence_score"]
                })
                
                st.dataframe(df_lb_show, use_container_width=True, hide_index=True)
                st.caption(f"Showing top {len(df_lb)} players in {region} region.")
            else:
                st.info("No leaderboard data available.")
        else:
            st.error("Error retrieving leaderboard data.")
    except Exception as e:
        st.error(f"Connection failed: {e}")

# -------------------------------------------------------------
# PAGE 3: MATCHMAKING POOLS
# -------------------------------------------------------------
elif page == "Matchmaking Pools":
    st.title("🤝 Skill & Ping Matchmaking Pools")
    st.markdown("Groups clean players into clusters based on skill score and network ping metrics.")
    
    try:
        r = requests.get(f"{BASE_URL}/matchmaking")
        if r.status_code == 200:
            groups = r.json()
            
            if groups:
                df_groups = pd.DataFrame(groups)
                
                # Show main table
                df_groups_show = pd.DataFrame({
                    "Group ID": df_groups["group_id"],
                    "Region": df_groups["region"],
                    "Player Count": df_groups["player_count"],
                    "Avg Ping": df_groups["avg_ping"].apply(lambda x: f"{x:.1f} ms"),
                    "Skill Tiers": df_groups["skill_tiers"].apply(lambda x: ", ".join(x))
                })
                
                st.dataframe(df_groups_show, use_container_width=True, hide_index=True)
                
                st.markdown("### Inspect Match Group")
                selected_group = st.selectbox("Select a Match Group to view player list", [g["group_id"] for g in groups])
                
                for g in groups:
                    if g["group_id"] == selected_group:
                        st.write(f"**Players in {selected_group} ({len(g['players'])}):**")
                        st.info(", ".join(g["players"]))
            else:
                st.info("No matchmaking groups found. Run /run-analysis first to group database players.")
        else:
            st.error("Error retrieving matchmaking data.")
    except Exception as e:
        st.error(f"Connection failed: {e}")

# -------------------------------------------------------------
# PAGE 4: SUSPICIOUS DATABASE
# -------------------------------------------------------------
elif page == "Suspicious Database":
    st.title("🚨 Suspicious Player Database")
    st.markdown("Lists all players with high risk metrics (Confidence Score > 40). These players are subjected to review, restrictions, or automatic bans.")
    
    try:
        r = requests.get(f"{BASE_URL}/flagged-players")
        if r.status_code == 200:
            flagged = r.json().get("flagged_players", [])
            
            if flagged:
                df_fl = pd.DataFrame(flagged)
                
                df_fl_show = pd.DataFrame({
                    "Player ID": df_fl["player_id"],
                    "Flag Reason": df_fl["flag_reason"],
                    "Kills": df_fl["kills"],
                    "KDR": df_fl["kdr"].apply(lambda x: f"{x:.2f}" if x else "0.00"),
                    "Score per Minute": df_fl["score_per_minute"].apply(lambda x: f"{x:.1f}" if x else "0.0"),
                    "Flagged Timestamp": df_fl["submitted_at"]
                })
                
                st.dataframe(df_fl_show, use_container_width=True, hide_index=True)
                st.caption(f"Found {len(flagged)} suspicious records flagged by rules or ML detectors.")
            else:
                st.success("🎉 No suspicious players flagged in the system!")
        else:
            st.error("Error retrieving suspicious players.")
    except Exception as e:
        st.error(f"Connection failed: {e}")

# -------------------------------------------------------------
# PAGE 5: SUBMIT PLAYER SCORE
# -------------------------------------------------------------
elif page == "Submit Player Score":
    st.title("🎯 Submit Player Score & Live Predictor")
    st.markdown("Submit match telemetry data for a player to check for anomalies and evaluate their skill score in real-time.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Player Match Telemetry Input")
        p_id = st.text_input("Player ID", "PC_DEMO_01")
        m_id = st.text_input("Match ID", "M_DEMO_999")
        region = st.selectbox("Region", ["India", "SEA", "Europe", "NA", "LatAm", "Middle_East"])
        device = st.selectbox("Device", ["Android", "iOS", "Console", "PC"])
        
        ping = st.slider("Ping (ms)", 1, 200, 45)
        score = st.number_input("Match Score", min_value=0, max_value=200000, value=2500)
        kills = st.number_input("Kills", min_value=0, max_value=1000, value=12)
        deaths = st.number_input("Deaths", min_value=0, max_value=1000, value=4)
        duration = st.number_input("Match Duration (seconds)", min_value=10, max_value=3600, value=300)
        
        submit_btn = st.button("Submit Score to System")
        
    with col2:
        st.subheader("Real-Time Prediction Analysis")
        if submit_btn:
            payload = {
                "player_id": p_id,
                "match_id": m_id,
                "region": region,
                "device": device,
                "ping": int(ping),
                "score": int(score),
                "kills": int(kills),
                "deaths": int(deaths),
                "match_duration_seconds": int(duration)
            }
            
            try:
                r = requests.post(f"{BASE_URL}/submit-score", json=payload)
                if r.status_code in [200, 201]:
                    res = r.json()
                    
                    st.success("Match telemetry successfully processed!")
                    
                    conf_score = res.get("confidence_score", 0.0)
                    zone = res.get("confidence_zone", "Clean")
                    
                    # Large visual zone badge
                    zone_html = get_zone_html(zone)
                    st.markdown(f"### Security Zone: {zone_html}", unsafe_allow_html=True)
                    
                    # Progress bar for confidence score
                    st.progress(int(conf_score))
                    st.metric(label="Cheat Confidence Score", value=f"{conf_score:.1f} / 100")
                    
                    st.markdown(f"**Recommended System Action:** `{res.get('action')}`")
                    
                    # Highlight details
                    c_hit = res.get("cheat_types_hit", [])
                    st.write(f"**Cheat Types Triggered (Primary Signal):** {', '.join(c_hit) if c_hit else 'None'}")
                    
                    c_conf = res.get("confirmed_cheats", [])
                    st.write(f"**Confirmed Cheats (Both Signals Hit):** {', '.join(c_conf) if c_conf else 'None'}")
                    
                    # Breakdown Expander
                    with st.expander("Detailed Suspicion Score Breakdown"):
                        st.json(res.get("score_breakdown", {}))
                        
                    with st.expander("Computed Match Feature Engineering"):
                        st.write(res.get("features", {}))
                else:
                    st.error(f"Error submitting data: {r.status_code} - {r.text}")
            except Exception as e:
                st.error(f"Failed to connect to API server: {e}")
        else:
            st.info("Input match telemetry on the left and click 'Submit Score' to run live AI/ML analysis.")
