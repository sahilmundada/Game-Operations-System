# ═══════════════════════════════════════════════════════════════════════════
# GAME OPS SYSTEM — MACHINE LEARNING & PREDICTION PIPELINE (v2 Upgrade)
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import datetime
from matplotlib import colormaps

# Reconfigure stdout to support unicode checkmarks on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "game_data_10000.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure output directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "kdr", "score_per_minute", "kill_rate", "efficiency", "death_rate",
    "survival_index", "score_kill_ratio", "ping_adjusted_score",
    "performance_index", "ping", "region_encoded", "device_encoded"
]

from game_ops.services.history import PlayerHistory


def build_player_histories(df):
    import random
    np.random.seed(42)
    random.seed(42)
    
    print("Building player histories...")
    
    histories = {}
    regions = df['region'].unique()
    
    # Generate regional session pools of match IDs and timestamps (1000 matches per region)
    region_pools = {}
    base_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=10)
    for r in regions:
        pool = []
        for i in range(1000):
            m_id = f"M_HIST_{r[:3].upper()}_{i:04d}"
            ts = (base_time + datetime.timedelta(minutes=i * 15)).isoformat()
            pool.append((m_id, ts))
        region_pools[r] = pool
        
    for idx, row in df.iterrows():
        p_id = row['player_id']
        region = row['region']
        
        # Decide number of matches (3-5)
        n_matches = random.randint(3, 5)
        
        pool = region_pools[region]
        selected_indices = sorted(random.sample(range(len(pool)), n_matches))
        
        match_history = []
        for i, p_idx in enumerate(selected_indices):
            m_id, ts = pool[p_idx]
            
            # Apply random variation +-15% to stats
            mult = 1.0 + random.uniform(-0.15, 0.15)
            
            s_score = max(0, int(row['score'] * mult))
            s_kills = max(0, int(row['kills'] * mult))
            s_deaths = max(0, int(row['deaths'] * mult))
            s_duration = max(10, int(row['match_duration_seconds'] * mult))
            
            s_kdr = s_kills / (s_deaths + 1)
            s_spm = s_score / (s_duration / 60.0)
            s_kr = s_kills / (s_duration / 60.0)
            s_eff = s_score / (s_kills + 1)
            s_dr = s_deaths / (s_duration / 60.0)
            s_si = 1.0 / (s_dr + 0.01)
            
            # Run the 9 rules on synthetic stats to determine historical flag status
            s_cheats = {
                'score_bot': s_spm > 900 and s_eff > 800,
                'kill_farmer': s_kr > 12 and s_kdr > 30 and s_duration < 200,
                'god_mode': s_deaths == 0 and s_kills > 60 and s_si > 500,
                'time_exploit': s_duration < 55 and s_score > 15000,
                'speed_hack': s_kr > 8 and s_duration < 120 and s_kdr > 20,
                'soft_cheat': (300 <= s_spm <= 600) and s_kdr > 12 and s_deaths < 2 and s_kills > 25,
                'score_inflate': s_score > 10000 and s_kills < 15 and s_eff > 500,
                'stat_padding': s_duration > 700 and s_kills > 40 and s_deaths < 3,
                'region_spoof': (region in ['LatAm', 'Middle_East']) and row['ping'] < 15 and s_spm > 400
            }
            
            s_conf = 0
            for name, confirmed in s_cheats.items():
                if confirmed:
                    pts = {
                        'score_bot': 55, 'kill_farmer': 60, 'god_mode': 65, 'time_exploit': 70,
                        'speed_hack': 60, 'soft_cheat': 30, 'score_inflate': 45, 'stat_padding': 35,
                        'region_spoof': 50
                    }[name]
                    s_conf += pts
                else:
                    prim_check = {
                        'score_bot': s_spm > 900,
                        'kill_farmer': s_kr > 12,
                        'god_mode': s_deaths == 0 and s_kills > 60,
                        'time_exploit': s_duration < 55,
                        'speed_hack': s_kr > 8 and s_duration < 120,
                        'soft_cheat': 300 <= s_spm <= 600,
                        'score_inflate': s_score > 10000 and s_kills < 15,
                        'stat_padding': s_duration > 700,
                        'region_spoof': (region in ['LatAm', 'Middle_East']) and row['ping'] < 15
                    }[name]
                    if prim_check:
                        s_conf += 15
            
            s_conf = min(100, s_conf)
            was_flagged = s_conf >= 61
            
            match_history.append({
                "match_id": m_id,
                "score": float(s_score),
                "kills": float(s_kills),
                "deaths": float(s_deaths),
                "score_per_minute": float(s_spm),
                "kdr": float(s_kdr),
                "was_flagged": was_flagged,
                "confidence_score": float(s_conf),
                "timestamp": ts
            })
            
        histories[p_id] = PlayerHistory(p_id, match_history)
        
    return histories

# ══════════════════════════════════════
# UPGRADE 3: CONFIDENCE SCORE SYSTEM
# ══════════════════════════════════════

def compute_confidence_score(player, history, iso_score, lof_score, iso_pred, lof_pred, tier_ceilings, device_limits):
    score = 0
    detail = {}
    
    spm = player['score_per_minute']
    eff = player['efficiency']
    kr = player['kill_rate']
    kdr = player['kdr']
    duration = player['match_duration_seconds']
    deaths = player['deaths']
    kills = player['kills']
    survival_index = player['survival_index']
    ping = player['ping']
    region = player['region']
    
    history_len = len(history.match_history) if history is not None else 0
    
    cheats = {
        'score_bot': {
            'primary': spm > 900,
            'secondary': eff > 800
        },
        'kill_farmer': {
            'primary': kr > 12,
            'secondary': kdr > 30 and duration < 200
        },
        'god_mode': {
            'primary': deaths == 0 and kills > 60,
            'secondary': survival_index > 500
        },
        'time_exploit': {
            'primary': duration < 55,
            'secondary': player['score'] > 15000
        },
        'speed_hack': {
            'primary': kr > 8 and duration < 120,
            'secondary': kdr > 20
        },
        'soft_cheat': {
            'primary': 300 <= spm <= 600,
            'secondary': kdr > 12 and deaths < 2 and kills > 25
        },
        'score_inflate': {
            'primary': player['score'] > 10000 and kills < 15,
            'secondary': eff > 500
        },
        'stat_padding': {
            'primary': duration > 700,
            'secondary': kills > 40 and deaths < 3
        },
        'region_spoof': {
            'primary': (region in ['LatAm', 'Middle_East']) and ping < 15,
            'secondary': spm > 400
        },
        'burst_cheat': {
            'primary': spm > 600,
            'secondary': history_len == 0
        }
    }
    
    confirmed_cheats = []
    unconfirmed_hits = []
    cheat_types_hit = []
    
    rule_pts = {
        'score_bot': 55,
        'kill_farmer': 60,
        'god_mode': 65,
        'time_exploit': 70,
        'speed_hack': 60,
        'soft_cheat': 30,
        'score_inflate': 45,
        'stat_padding': 35,
        'region_spoof': 50,
        'burst_cheat': 40
    }
    
    for c_name, signals in cheats.items():
        prim = bool(signals['primary'])
        sec = bool(signals['secondary'])
        confirmed = prim and sec
        
        if prim:
            cheat_types_hit.append(c_name)
            if confirmed:
                confirmed_cheats.append(c_name)
                score += rule_pts[c_name]
                detail[f"{c_name}_confirmed"] = rule_pts[c_name]
            else:
                unconfirmed_hits.append(c_name)
                score += 15
                detail[f"{c_name}_primary_only"] = 15
                
    # ML-BASED CONTRIBUTIONS
    # Isolation Forest
    if iso_score < -0.30:
        score += 30
        detail['iso_forest_score'] = 30
    elif iso_score < -0.15:
        score += 20
        detail['iso_forest_score'] = 20
    elif iso_score < -0.05:
        score += 10
        detail['iso_forest_score'] = 10
        
    # LOF
    if lof_score < -2.0:
        score += 25
        detail['lof_score'] = 25
    elif lof_score < -1.5:
        score += 15
        detail['lof_score'] = 15
    elif lof_score < -1.0:
        score += 8
        detail['lof_score'] = 8
        
    # Ensemble bonus
    if iso_pred == -1 and lof_pred == -1:
        score += 10
        detail['ensemble_bonus'] = 10
        
    # CONTEXT-BASED CONTRIBUTIONS
    # Tier ceiling
    tier = player.get('predicted_skill_tier', 'Bronze')
    ceiling = tier_ceilings.get(tier, 250.0)
    if spm > ceiling * 2.0:
        score += 20
        detail['tier_ceiling_exceeded_2.0'] = 20
    elif spm > ceiling * 1.5:
        score += 10
        detail['tier_ceiling_exceeded_1.5'] = 10
        
    # Device multiplier limit
    device = player['device']
    dev_limit = device_limits.get(device, 450.0)
    if spm > dev_limit:
        score += 8
        detail['device_multiplier_exceeded'] = 8
        
    # Ping too low for declared region
    if region in ['LatAm', 'Middle_East'] and ping < 15:
        score += 12
        detail['ping_too_low_region'] = 12
        
    # HISTORY-BASED MODIFIERS
    if history is not None:
        if history.veteran_status:
            score -= 10
            detail['veteran_discount'] = -10
        if history.flag_rate > 0.3:
            score += 20
            detail['history_repeat_offender'] = 20
        if history.trend == "suspicious_spike":
            score += 15
            detail['history_spike'] = 15
        if history.consistency_score < 0.3:
            score += 10
            detail['history_low_consistency'] = 10
        elif history.consistency_score > 0.8:
            score -= 5
            detail['history_high_consistency'] = -5
            
    # Cap at 100, floor at 0
    score = max(0, min(100, score))
    detail['total'] = score
    
    return {
        'confidence_score': float(score),
        'cheat_types_hit': cheat_types_hit,
        'confirmed_cheats': confirmed_cheats,
        'unconfirmed_hits': unconfirmed_hits,
        'score_breakdown': detail
    }


def get_confidence_zone_and_action(score):
    if score <= 20:
        return "Clean", "CLEAN", "Rank normally, full access"
    elif score <= 40:
        return "Watch", "WATCH", "Flag for monitoring, rank normally but log all matches"
    elif score <= 60:
        return "Review", "REVIEW", "Exclude from leaderboard, send to manual review queue"
    elif score <= 80:
        return "Restricted", "RESTRICTED", "Score withheld. Account rate-limited."
    else:
        return "Flagged", "FLAGGED", "Automatic ban flag, full exclusion, incident report"


def compute_group_metrics(group_players, is_unresolved=False):
    total = len(group_players)
    if total == 0:
        return {
            "avg_mmr": 0.0,
            "mmr_spread": 0.0,
            "avg_ping": 0.0,
            "ping_spread": 0.0,
            "avg_confidence": 0.0,
            "fairness_score": 0.0,
            "quality_label": "Unbalanced",
            "device_breakdown": {},
            "device_flag": "balanced",
            "skill_tiers_present": []
        }
        
    mmrs = [p['mmr'] for p in group_players]
    pings = [p['ping'] for p in group_players]
    confidences = [p['confidence_score'] for p in group_players]
    
    avg_mmr = float(np.mean(mmrs))
    mmr_spread = float(max(mmrs) - min(mmrs))
    skill_pts = 35.0 * (1.0 - min(1.0, mmr_spread / 0.4))
    
    avg_ping = float(np.mean(pings))
    ping_spread = float(max(pings) - min(pings))
    ping_pts = 30.0 * (1.0 - min(1.0, ping_spread / 100.0))
    
    avg_confidence = float(np.mean(confidences))
    cleanliness_pts = 15.0 * (1.0 - (avg_confidence / 40.0))
    
    # Device breakdown & flag
    device_breakdown = {}
    for p in group_players:
        d = p['device']
        device_breakdown[d] = device_breakdown.get(d, 0) + 1
        
    device_flag = "balanced"
    device_pts = 20.0
    
    devices = [p.get('device') for p in group_players]
    is_mobile = all(d in ['Android', 'iOS'] for d in devices if d is not None)
    
    if is_mobile:
        device_flag = "balanced"
        device_pts = 20.0
    elif total > 4:
        pc_count = device_breakdown.get('PC', 0)
        pc_ratio = pc_count / total
        if pc_ratio < 0.15:
            device_flag = "console_heavy"
            device_pts = 14.0
        elif pc_ratio > 0.65:
            if is_unresolved:
                device_flag = "pc_heavy_unresolved"
                device_pts = 3.0
            else:
                device_flag = "pc_heavy"
                device_pts = 8.0
                
    fairness_score = float(skill_pts + ping_pts + device_pts + cleanliness_pts)
    fairness_score = max(0.0, min(100.0, fairness_score))
    
    if fairness_score >= 85.0:
        quality_label = "Balanced"
    elif fairness_score >= 70.0:
        quality_label = "Competitive"
    elif fairness_score >= 50.0:
        quality_label = "Acceptable"
    else:
        quality_label = "Unbalanced"
        
    skill_tiers = list(set(p.get('skill_tier', 'Bronze') for p in group_players))
    
    return {
        "avg_mmr": round(avg_mmr, 4),
        "mmr_spread": round(mmr_spread, 4),
        "avg_ping": round(avg_ping, 2),
        "ping_spread": round(ping_spread, 2),
        "avg_confidence": round(avg_confidence, 2),
        "fairness_score": round(fairness_score, 2),
        "quality_label": quality_label,
        "device_breakdown": device_breakdown,
        "device_flag": device_flag,
        "skill_tiers_present": sorted(skill_tiers)
    }


def execute_device_swaps(groups, region):
    valid_group_ids = [gid for gid in groups.keys() if not gid.endswith('_GPing')]
    swaps_count = 0
    
    for gid in valid_group_ids:
        attempts = 0
        while attempts < 2:
            group = groups[gid]
            total = len(group)
            if total <= 4:
                break
                
            pc_players = [p for p in group if p['device'] == 'PC']
            pc_count = len(pc_players)
            pc_ratio = pc_count / total
            
            if pc_ratio > 0.65:
                if not pc_players:
                    break
                lowest_pc_player = min(pc_players, key=lambda x: x['mmr'])
                group_avg_mmr = np.mean([p['mmr'] for p in group])
                
                best_candidate = None
                best_candidate_gid = None
                best_diff = 999.0
                
                for other_gid in valid_group_ids:
                    if other_gid == gid:
                        continue
                    other_group = groups[other_gid]
                    for p in other_group:
                        if p['device'] == 'Console':
                            diff = abs(p['mmr'] - group_avg_mmr)
                            if diff <= 0.1:
                                if diff < best_diff:
                                    best_diff = diff
                                    best_candidate = p
                                    best_candidate_gid = other_gid
                                    
                if best_candidate is not None:
                    groups[gid].remove(lowest_pc_player)
                    groups[gid].append(best_candidate)
                    
                    groups[best_candidate_gid].remove(best_candidate)
                    groups[best_candidate_gid].append(lowest_pc_player)
                    
                    lowest_pc_player['match_group_id'] = best_candidate_gid
                    best_candidate['match_group_id'] = gid
                    
                    swaps_count += 1
                    attempts += 1
                else:
                    break
            else:
                break
                
    return swaps_count


# ══════════════════════════════════════
# UPDATED TRAINING PIPELINE
# ══════════════════════════════════════


def train_pipeline(data_path):
    # --- Step 1: Feature Engineering ---
    print("--- Phase 1: Feature Engineering ---")
    df = pd.read_csv(data_path, keep_default_na=False)
    
    # Calculate derived features
    df['kdr'] = df['kills'] / (df['deaths'] + 1)
    df['score_per_minute'] = df['score'] / (df['match_duration_seconds'] / 60.0)
    df['kill_rate'] = df['kills'] / (df['match_duration_seconds'] / 60.0)
    df['efficiency'] = df['score'] / (df['kills'] + 1)
    df['death_rate'] = df['deaths'] / (df['match_duration_seconds'] / 60.0)
    df['survival_index'] = 1.0 / (df['death_rate'] + 0.01)
    df['score_kill_ratio'] = df['score'] / (df['kills'] + 1)
    df['ping_adjusted_score'] = df['score'] / (1.0 + df['ping'] / 100.0)
    df['performance_index'] = df['score_per_minute'] / (df['kill_rate'] + 0.1)

    label_encoders = {}
    for col in ['region', 'device']:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))

    scaler = StandardScaler()
    df_scaled_features = pd.DataFrame(
        scaler.fit_transform(df[FEATURE_COLS]), 
        columns=FEATURE_COLS
    )
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    # --- Step 2: Build PlayerHistory for all 10,000 players ---
    print("\n--- Phase 2: Build Player History ---")
    histories = build_player_histories(df)
    joblib.dump(histories, os.path.join(MODELS_DIR, "player_history.pkl"))

    # --- Step 4: Run Isolation Forest & LOF ---
    print("\n--- Phase 4: Training ML Anomaly Detectors ---")
    scaled_data = df_scaled_features.values

    iso_forest = IsolationForest(
        n_estimators=300,
        contamination=0.04,  # Slightly higher for v2
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    df['iso_pred'] = iso_forest.fit_predict(scaled_data)
    df['iso_score'] = iso_forest.decision_function(scaled_data)
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest_model.pkl"))

    lof = LocalOutlierFactor(
        n_neighbors=25,
        contamination=0.04,
        novelty=True,
        n_jobs=-1
    )
    lof.fit(scaled_data)
    df['lof_pred'] = lof.predict(scaled_data)
    df['lof_score'] = lof.negative_outlier_factor_
    joblib.dump(lof, os.path.join(MODELS_DIR, "lof_model.pkl"))

    # --- Initial rough Skill prediction to calculate Tier Ceilings ---
    print("\n--- Phase 4b: Estimating Baseline Tier Ceilings ---")
    
    # Simple heuristic to identify dirty players for baseline
    dirty_rules = (df['score_per_minute'] > 800) | (df['kdr'] > 25)
    dirty_ml = (df['iso_pred'] == -1) & (df['lof_pred'] == -1)
    rough_clean = df[~(dirty_rules | dirty_ml)].copy()
    
    rough_skill_scaler = MinMaxScaler()
    rough_scaled = rough_skill_scaler.fit_transform(rough_clean[['kdr', 'score_per_minute', 'efficiency']])
    rough_clean['skill_score'] = 0.35 * rough_scaled[:, 0] + 0.35 * rough_scaled[:, 1] + 0.30 * rough_scaled[:, 2]
    
    rough_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rough_rf.fit(rough_clean[FEATURE_COLS].values, rough_clean['skill_score'].values)
    
    baseline_skills = rough_rf.predict(scaled_data)
    q90 = np.quantile(baseline_skills, 0.90)
    q70 = np.quantile(baseline_skills, 0.70)
    q40 = np.quantile(baseline_skills, 0.40)
    q15 = np.quantile(baseline_skills, 0.15)
    
    def assign_baseline_tier(score):
        if score >= q90: return "Pro"
        elif score >= q70: return "Platinum"
        elif score >= q40: return "Gold"
        elif score >= q15: return "Silver"
        else: return "Bronze"
        
    df['predicted_skill_tier'] = [assign_baseline_tier(s) for s in baseline_skills]
    
    # Calculate 95th percentile score_per_minute of clean players in each tier
    tier_ceilings = {}
    for tier in ["Pro", "Platinum", "Gold", "Silver", "Bronze"]:
        subset = rough_clean[rough_clean['player_id'].isin(df[df['predicted_skill_tier'] == tier]['player_id'])]
        if not subset.empty:
            tier_ceilings[tier] = float(np.percentile(subset['score_per_minute'], 95))
        else:
            tier_ceilings[tier] = 300.0 if tier == 'Pro' else 200.0 # safety fallbacks
            
    device_limits = {"Android": 350.0, "iOS": 450.0, "Console": 500.0, "PC": 600.0}
    
    joblib.dump({"tier_ceilings": tier_ceilings, "device_limits": device_limits}, os.path.join(MODELS_DIR, "confidence_config.pkl"))

    # --- Step 5: Compute confidence_score for every player ---
    print("\n--- Phase 5: Computing Confidence Scores ---")
    
    confidence_scores = []
    confidence_zones = []
    status_labels = []
    actions = []
    cheat_types_hits = []
    confirmed_cheats_lists = []
    unconfirmed_hits_lists = []
    score_breakdowns = []
    history_flag_rates = []
    history_trends = []
    consistency_scores = []
    veteran_statuses = []
    
    for idx, row in df.iterrows():
        p_id = row['player_id']
        hist = histories.get(p_id)
        
        # Calculate conf score
        res = compute_confidence_score(
            row.to_dict(), hist, row['iso_score'], row['lof_score'],
            row['iso_pred'], row['lof_pred'], tier_ceilings, device_limits
        )
        
        zone, status, act = get_confidence_zone_and_action(res['confidence_score'])
        
        confidence_scores.append(res['confidence_score'])
        confidence_zones.append(zone)
        status_labels.append(status)
        actions.append(act)
        cheat_types_hits.append(res['cheat_types_hit'])
        confirmed_cheats_lists.append(res['confirmed_cheats'])
        unconfirmed_hits_lists.append(res['unconfirmed_hits'])
        score_breakdowns.append(res['score_breakdown'])
        
        if hist is not None:
            history_flag_rates.append(hist.flag_rate)
            history_trends.append(hist.trend)
            consistency_scores.append(hist.consistency_score)
            veteran_statuses.append(hist.veteran_status)
        else:
            history_flag_rates.append(0.0)
            history_trends.append("stable")
            consistency_scores.append(1.0)
            veteran_statuses.append(False)
            
    df['confidence_score'] = confidence_scores
    df['confidence_zone'] = confidence_zones
    df['status'] = [z.lower() for z in confidence_zones] # compatibility
    df['status_label'] = status_labels
    df['action'] = actions
    df['cheat_types_hit'] = cheat_types_hits
    df['confirmed_cheats'] = confirmed_cheats_lists
    df['unconfirmed_hits'] = unconfirmed_hits_lists
    df['score_breakdown'] = [json.dumps(sb) for sb in score_breakdowns]
    
    df['history_flag_rate'] = history_flag_rates
    df['history_trend'] = history_trends
    df['consistency_score'] = consistency_scores
    df['veteran_status'] = veteran_statuses
    df['final_flagged'] = df['confidence_score'] >= 61
    
    # Save cheat thresholds rules dict
    cheat_type_rules = {
        "score_bot": "spm > 900 & eff > 800",
        "kill_farmer": "kr > 12 & kdr > 30 & duration < 200",
        "god_mode": "deaths == 0 & kills > 60 & si > 500",
        "time_exploit": "duration < 55 & score > 15000",
        "speed_hack": "kr > 8 & duration < 120 & kdr > 20",
        "soft_cheat": "300 <= spm <= 600 & kdr > 12 & deaths < 2 & kills > 25",
        "score_inflate": "score > 10000 & kills < 15 & eff > 500",
        "stat_padding": "duration > 700 & kills > 40 & deaths < 3",
        "region_spoof": "ping < 15 in LatAm/Middle_East & spm > 400",
        "burst_cheat": "spm > 600 & history_len == 0"
    }
    joblib.dump(cheat_type_rules, os.path.join(MODELS_DIR, "cheat_type_thresholds.pkl"))

    # --- Step 7: Skill scoring ONLY for confidence_score <= 40 ---
    print("\n--- Phase 7: Training Skill Prediction Model ---")
    clean_df = df[df['confidence_score'] <= 40].copy()
    
    skill_scaler = MinMaxScaler()
    skill_features = ['kdr', 'score_per_minute', 'efficiency']
    scaled_skill = skill_scaler.fit_transform(clean_df[skill_features])
    
    clean_df['skill_score'] = (
        0.35 * scaled_skill[:, 0] + 
        0.35 * scaled_skill[:, 1] + 
        0.30 * scaled_skill[:, 2]
    )
    joblib.dump(skill_scaler, os.path.join(MODELS_DIR, "skill_scaler.pkl"))

    X = df_scaled_features.loc[clean_df.index].values
    y = clean_df['skill_score'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(X_train, y_train)
    
    y_pred = rf_reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Skill Model Evaluation:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    joblib.dump(rf_reg, os.path.join(MODELS_DIR, "skill_model.pkl"))

    # Assign predicted skill score and final skill tiers to clean/watch players
    clean_df['predicted_skill_score'] = rf_reg.predict(X)
    
    thresholds = {
        'q90': clean_df['predicted_skill_score'].quantile(0.90),
        'q70': clean_df['predicted_skill_score'].quantile(0.70),
        'q40': clean_df['predicted_skill_score'].quantile(0.40),
        'q15': clean_df['predicted_skill_score'].quantile(0.15)
    }
    joblib.dump(thresholds, os.path.join(MODELS_DIR, "tier_thresholds.pkl"))

    def assign_final_tier(score):
        if score >= thresholds['q90']: return "Pro"
        elif score >= thresholds['q70']: return "Platinum"
        elif score >= thresholds['q40']: return "Gold"
        elif score >= thresholds['q15']: return "Silver"
        else: return "Bronze"

    clean_df['skill_tier'] = clean_df['predicted_skill_score'].apply(assign_final_tier)

    # Map back to full dataframe
    df['predicted_skill_score'] = np.nan
    df.loc[clean_df.index, 'predicted_skill_score'] = clean_df['predicted_skill_score']
    df['skill_tier'] = "Excluded (score > 40)"
    df.loc[clean_df.index, 'skill_tier'] = clean_df['skill_tier']
    # If confidence_score >= 61, label is Excluded (score >= 61)
    df.loc[df['confidence_score'] >= 61, 'skill_tier'] = "Excluded (score >= 61)"

    # --- Step 8: Matchmaking ONLY for confidence_score <= 40 ---
    print("\n--- Phase 8: Training Matchmaking Clusters ---")
    from sklearn.cluster import KMeans

    # Stage 1: Gate
    eligible_mask = df['confidence_score'] <= 40
    eligible_df = df[eligible_mask].copy()
    ineligible_df = df[~eligible_mask].copy()
    
    # Exclude ineligible players
    df.loc[~eligible_mask, 'match_group_id'] = "EXCLUDED"
    df.loc[~eligible_mask, 'match_group_reason'] = "confidence_score too high: " + df.loc[~eligible_mask, 'confidence_score'].astype(int).astype(str)
    
    print(f"Eligible for matchmaking : {len(eligible_df)} players")
    print(f"Excluded from matchmaking: {len(ineligible_df)} players (confidence > 40)")

    # Stage 2: Compute MMR
    mmr_scaler = MinMaxScaler()
    eligible_skills = eligible_df['predicted_skill_score'].values.reshape(-1, 1)
    skill_score_norm = mmr_scaler.fit_transform(eligible_skills).flatten()
    
    consistency = eligible_df['consistency_score'].values
    confidence = eligible_df['confidence_score'].values
    
    mmr_raw = (0.60 * skill_score_norm) + (0.25 * consistency) + (0.15 * (1.0 - confidence / 100.0))
    
    device_multipliers = {
        "PC": 0.93,
        "Console": 0.96,
        "iOS": 1.00,
        "Android": 1.02
    }
    device_mult_arr = eligible_df['device'].map(lambda x: device_multipliers.get(x, 1.0)).values
    mmr_adjusted = mmr_raw * device_mult_arr
    
    eligible_df['mmr'] = mmr_adjusted
    eligible_df['skill_score_norm'] = skill_score_norm

    # Initialize matchmaking configs and registries
    group_registry = {}
    matchmaking_config = {
        "regions": {},
        "cluster_to_group": {}
    }
    
    regions_list = ["India", "SEA", "Europe", "NA", "LatAm", "Middle_East"]
    processed_players_list = []
    total_swaps = 0

    for region in regions_list:
        region_players_all = eligible_df[eligible_df['region'] == region].copy()
        
        for pool_cat in ["Cross", "Mobile"]:
            if pool_cat == "Cross":
                region_players = region_players_all[region_players_all['device'].isin(['PC', 'Console'])].copy()
            else:
                region_players = region_players_all[region_players_all['device'].isin(['Android', 'iOS'])].copy()
                
            n_players = len(region_players)
            
            # Stage 3: Dynamic Clustering with Optimal K
            if n_players < 4:
                if n_players > 0:
                    skill_min = float(region_players['predicted_skill_score'].min())
                    skill_max = float(region_players['predicted_skill_score'].max())
                    ping_min = float(region_players['ping'].min())
                    ping_max = float(region_players['ping'].max())
                else:
                    skill_min, skill_max, ping_min, ping_max = 0.0, 1.0, 0.0, 100.0
                    
                if region not in matchmaking_config["regions"]:
                    matchmaking_config["regions"][region] = {}
                matchmaking_config["regions"][region][pool_cat] = {
                    "skill_min": skill_min,
                    "skill_max": skill_max,
                    "ping_min": ping_min,
                    "ping_max": ping_max,
                    "optimal_k": 1
                }
                
                if n_players > 0:
                    group_id = f"{region}_{pool_cat}_G1"
                    
                    players_in_group = []
                    for _, row in region_players.iterrows():
                        p_dict = row.to_dict()
                        p_dict['match_group_id'] = group_id
                        p_dict['match_group_reason'] = None
                        players_in_group.append(p_dict)
                        processed_players_list.append(p_dict)
                        
                    metrics = compute_group_metrics(players_in_group, is_unresolved=False)
                    group_registry[group_id] = {
                        "group_id": group_id,
                        "region": region,
                        "player_count": len(players_in_group),
                        "players": [p['player_id'] for p in players_in_group],
                        "player_records": [
                            {
                                "player_id": p['player_id'],
                                "mmr": p['mmr'],
                                "ping": p['ping'],
                                "confidence_score": p['confidence_score'],
                                "device": p['device'],
                                "skill_tier": p['skill_tier']
                            } for p in players_in_group
                        ],
                        **metrics
                    }
                continue
                
            skill_min = float(region_players['predicted_skill_score'].min())
            skill_max = float(region_players['predicted_skill_score'].max())
            ping_min = float(region_players['ping'].min())
            ping_max = float(region_players['ping'].max())
            
            region_players['ping_norm'] = (region_players['ping'] - ping_min) / (ping_max - ping_min + 1e-8)
            features_2d = region_players[['mmr', 'ping_norm']].values
            
            if n_players <= 15:
                optimal_k = max(1, n_players // 4)
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
                cluster_labels = kmeans.fit_predict(features_2d)
                region_players['cluster'] = cluster_labels
                joblib.dump(kmeans, os.path.join(MODELS_DIR, f"kmeans_{region}_{pool_cat}.pkl"))
            else:
                # Elbow point selection
                K_max = min(12, n_players // 3)
                inertias = []
                k_range = list(range(2, K_max + 1))
                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=15)
                    km.fit(features_2d)
                    inertias.append(km.inertia_)
                    
                total_reduction = inertias[0] - inertias[-1]
                optimal_k = 2
                if total_reduction > 1e-8:
                    for idx, k in enumerate(k_range[:-1]):
                        reduction = inertias[idx] - inertias[idx + 1]
                        if reduction < 0.15 * total_reduction:
                            optimal_k = k
                            break
                
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
                cluster_labels = kmeans.fit_predict(features_2d)
                region_players['cluster'] = cluster_labels
                joblib.dump(kmeans, os.path.join(MODELS_DIR, f"kmeans_{region}_{pool_cat}.pkl"))
                
            if region not in matchmaking_config["regions"]:
                matchmaking_config["regions"][region] = {}
            matchmaking_config["regions"][region][pool_cat] = {
                "skill_min": skill_min,
                "skill_max": skill_max,
                "ping_min": ping_min,
                "ping_max": ping_max,
                "optimal_k": int(optimal_k)
            }
            
            # Stage 4: Ping Variance Filter (High-Ping Outliers)
            outliers_in_pool = []
            cluster_groups = {}
            
            for c_id, group_df in region_players.groupby('cluster'):
                pings = group_df['ping'].values
                group_dicts = [row.to_dict() for _, row in group_df.iterrows()]
                
                if len(group_df) > 1:
                    ping_mean = np.mean(pings)
                    ping_std = np.std(pings)
                    if ping_std > 0:
                        is_outlier = np.abs(pings - ping_mean) > 1.5 * ping_std
                    else:
                        is_outlier = np.zeros(len(group_df), dtype=bool)
                else:
                    is_outlier = np.zeros(len(group_df), dtype=bool)
                    
                non_outliers = []
                for idx, outlier in enumerate(is_outlier):
                    p_item = group_dicts[idx]
                    if outlier:
                        p_item['match_group_id'] = f"{region}_{pool_cat}_GPing"
                        p_item['match_group_reason'] = f"High-ping outlier in cluster {c_id}"
                        outliers_in_pool.append(p_item)
                    else:
                        non_outliers.append(p_item)
                cluster_groups[int(c_id)] = non_outliers
                
            # sequential renaming and mapping
            pool_sequential_groups = {}
            seq_num = 0
            for c_id in sorted(cluster_groups.keys()):
                group_list = cluster_groups[c_id]
                if len(group_list) == 0:
                    continue
                seq_num += 1
                seq_gid = f"{region}_{pool_cat}_G{seq_num}"
                for p in group_list:
                    p['match_group_id'] = seq_gid
                    p['match_group_reason'] = None
                pool_sequential_groups[seq_gid] = group_list
                matchmaking_config["cluster_to_group"][(region, pool_cat, int(c_id))] = seq_gid
                
            # Stage 5: Device Audit & Swaps
            if pool_cat == "Cross":
                swaps_executed = execute_device_swaps(pool_sequential_groups, region)
                total_swaps += swaps_executed
                print(f"Executed {swaps_executed} Console/PC swaps in region {region} ({pool_cat})")
            
            # Save sequential groups and calculate quality metrics
            for seq_gid, group_list in pool_sequential_groups.items():
                total = len(group_list)
                pc_ratio = sum(1 for p in group_list if p['device'] == 'PC') / total if total > 0 else 0.0
                is_unresolved = (pc_ratio > 0.65) and (total > 4)
                
                metrics = compute_group_metrics(group_list, is_unresolved=is_unresolved)
                group_registry[seq_gid] = {
                    "group_id": seq_gid,
                    "region": region,
                    "player_count": len(group_list),
                    "players": [p['player_id'] for p in group_list],
                    "player_records": [
                        {
                            "player_id": p['player_id'],
                            "mmr": p['mmr'],
                            "ping": p['ping'],
                            "confidence_score": p['confidence_score'],
                            "device": p['device'],
                            "skill_tier": p['skill_tier']
                        } for p in group_list
                    ],
                    **metrics
                }
                processed_players_list.extend(group_list)
                
            # High-ping overflow group
            if outliers_in_pool:
                gping_id = f"{region}_{pool_cat}_GPing"
                metrics = compute_group_metrics(outliers_in_pool, is_unresolved=False)
                group_registry[gping_id] = {
                    "group_id": gping_id,
                    "region": region,
                    "player_count": len(outliers_in_pool),
                    "players": [p['player_id'] for p in outliers_in_pool],
                    "player_records": [
                        {
                            "player_id": p['player_id'],
                            "mmr": p['mmr'],
                            "ping": p['ping'],
                            "confidence_score": p['confidence_score'],
                            "device": p['device'],
                            "skill_tier": p['skill_tier']
                        } for p in outliers_in_pool
                    ],
                    **metrics
                }
                processed_players_list.extend(outliers_in_pool)

    # Reconstruct final processed DataFrame
    ineligible_dicts = [row.to_dict() for _, row in ineligible_df.iterrows()]
    for p in ineligible_dicts:
        p['match_group_id'] = "EXCLUDED"
        p['match_group_reason'] = f"confidence_score too high: {int(p['confidence_score'])}"
        p['mmr'] = np.nan
        
    full_processed_dicts = processed_players_list + ineligible_dicts
    df_processed = pd.DataFrame(full_processed_dicts)

    # Stage 7: Save Pickled Artifacts
    joblib.dump(group_registry, os.path.join(MODELS_DIR, "group_registry.pkl"))
    joblib.dump(mmr_scaler, os.path.join(MODELS_DIR, "mmr_scaler.pkl"))
    joblib.dump(matchmaking_config, os.path.join(MODELS_DIR, "matchmaking_config.pkl"))

    # Print summary statistics
    avg_fairness = np.mean([g['fairness_score'] for g in group_registry.values()]) if group_registry else 0.0
    quality_counts = {}
    for g in group_registry.values():
        lbl = g['quality_label']
        quality_counts[lbl] = quality_counts.get(lbl, 0) + 1
    pct_breakdown = {k: f"{(v / len(group_registry) * 100):.1f}%" for k, v in quality_counts.items()} if group_registry else {}
    
    print("\n" + "=" * 50)
    print("UPGRADED MATCHMAKING ENGINE SUMMARY STATS")
    print("=" * 50)
    print(f"Total groups formed        : {len(group_registry)}")
    print(f"Average Group Fairness Score: {avg_fairness:.2f}")
    print(f"Total Mobile/PC Swaps      : {total_swaps}")
    print(f"High-Ping Overflow Groups  : {sum(1 for g_id in group_registry if g_id.endswith('_GPing'))}")
    print(f"High-Ping Overflow Players : {sum(len(g['players']) for g_id, g in group_registry.items() if g_id.endswith('_GPing'))}")
    print("Quality Label Breakdown    :")
    for lbl, pct in pct_breakdown.items():
        print(f"  - {lbl}: {pct}")
    print("=" * 50 + "\n")

    # --- Step 9: Leaderboard ---
    print("\n--- Phase 9: Global Leaderboard Sorting ---")
    lb_players = df_processed[df_processed['confidence_score'] <= 40].copy()
    lb_players['zone_sort'] = lb_players['confidence_zone'].apply(lambda x: 0 if x == 'Clean' else 1)
    
    global_lb = lb_players.sort_values(
        by=['zone_sort', 'score', 'deaths', 'kills'],
        ascending=[True, False, True, False]
    ).reset_index(drop=True)
    global_lb['global_rank'] = global_lb.index + 1

    df_processed.to_csv(os.path.join(OUTPUT_DIR, "processed_training_data_v2.csv"), index=False)

    # --- Step 10: Generate updated 3x2 visualizations ---
    print("\n--- Phase 10: Generating Visualization Report ---")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle("GAME OPERATIONS SYSTEM v2 — ANOMALY & CONFIDENCE REPORT", fontsize=18, fontweight='bold')

    zones_data = [
        (df_processed[df_processed['confidence_score'] <= 20]['confidence_score'], 'green', 'Clean (0-20)'),
        (df_processed[(df_processed['confidence_score'] > 20) & (df_processed['confidence_score'] <= 40)]['confidence_score'], 'teal', 'Watch (21-40)'),
        (df_processed[(df_processed['confidence_score'] > 40) & (df_processed['confidence_score'] <= 60)]['confidence_score'], 'orange', 'Review (41-60)'),
        (df_processed[(df_processed['confidence_score'] > 60) & (df_processed['confidence_score'] <= 80)]['confidence_score'], 'coral', 'Restricted (61-80)'),
        (df_processed[df_processed['confidence_score'] > 80]['confidence_score'], 'red', 'Flagged (81-100)')
    ]
    for data, color, label in zones_data:
        if not data.empty:
            axes[0, 0].hist(data, bins=np.arange(0, 105, 5), color=color, alpha=0.7, label=label, edgecolor='black')
    axes[0, 0].axvline(20, color='gray', linestyle='--')
    axes[0, 0].axvline(40, color='gray', linestyle='--')
    axes[0, 0].axvline(60, color='gray', linestyle='--')
    axes[0, 0].axvline(80, color='gray', linestyle='--')
    axes[0, 0].set_title("Player confidence score distribution", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Confidence Score")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    cheat_counts = {}
    for c_name in ['score_bot', 'kill_farmer', 'god_mode', 'time_exploit', 'speed_hack', 'soft_cheat', 'score_inflate', 'stat_padding', 'region_spoof', 'burst_cheat']:
        cheat_counts[c_name] = sum(df_processed['confirmed_cheats'].apply(lambda x: c_name in x))
    sorted_cheats = sorted(cheat_counts.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_cheats]
    counts = [x[1] for x in sorted_cheats]
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(names)))
    axes[0, 1].barh(names, counts, color=colors)
    axes[0, 1].set_title("Confirmed cheat types detected", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Count")

    sns.scatterplot(
        x='consistency_score', y='confidence_score', hue='confidence_zone',
        hue_order=['Clean', 'Watch', 'Review', 'Restricted', 'Flagged'],
        palette={'Clean': 'green', 'Watch': 'teal', 'Review': 'orange', 'Restricted': 'coral', 'Flagged': 'red'},
        data=df_processed, alpha=0.6, s=15, ax=axes[1, 0]
    )
    axes[1, 0].axvline(0.3, color='red', linestyle='--')
    axes[1, 0].set_title("Player consistency vs suspicion level", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Consistency Score")
    axes[1, 0].set_ylabel("Confidence Score")

    zone_counts = df_processed['confidence_zone'].value_counts()
    ordered_zones = ['Clean', 'Watch', 'Review', 'Restricted', 'Flagged']
    counts_donut = [zone_counts.get(z, 0) for z in ordered_zones]
    colors_donut = ['green', 'teal', 'orange', 'coral', 'red']
    plot_counts = []
    plot_labels = []
    plot_colors = []
    for z, c, col in zip(ordered_zones, counts_donut, colors_donut):
        if c > 0:
            plot_counts.append(c)
            plot_labels.append(f"{z} ({c})")
            plot_colors.append(col)
    axes[1, 1].pie(plot_counts, labels=plot_labels, colors=plot_colors, autopct='%1.1f%%', startangle=90, 
                   wedgeprops=dict(width=0.4, edgecolor='w'))
    axes[1, 1].set_title("Player distribution by confidence zone", fontsize=12, fontweight='bold')

    top_20 = global_lb.head(20).copy()
    tier_colors = {
        'Bronze': 'gray',
        'Silver': 'steelblue',
        'Gold': 'gold',
        'Platinum': 'mediumpurple',
        'Pro': 'crimson'
    }
    colors_lb = [tier_colors.get(tier, 'gray') for tier in top_20['skill_tier']]
    sns.barplot(x='player_id', y='score', data=top_20, palette=colors_lb, ax=axes[2, 0], hue='player_id', legend=False)
    axes[2, 0].set_title("Top 20 players — leaderboard with confidence", fontsize=12, fontweight='bold')
    axes[2, 0].tick_params(axis='x', rotation=45)
    for idx, row in enumerate(top_20.itertuples()):
        axes[2, 0].text(idx, row.score + (top_20['score'].max() * 0.01), f"C:{int(row.confidence_score)}", 
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')
    patches = [plt.Rectangle((0,0),1,1, color=color) for color in tier_colors.values()]
    axes[2, 0].legend(patches, tier_colors.keys(), title="Skill Tier")

    importances = rf_reg.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLS[i] for i in indices]
    sorted_importances = importances[indices]
    norm = plt.Normalize(sorted_importances.min(), sorted_importances.max())
    imp_colors = plt.cm.RdYlGn(norm(sorted_importances))
    axes[2, 1].barh(sorted_features[:12], sorted_importances[:12], color=imp_colors[:12])
    axes[2, 1].invert_yaxis()
    axes[2, 1].set_title("Feature importance — skill model", fontsize=12, fontweight='bold')

    plt.tight_layout()
    analysis_plot_path = os.path.join(OUTPUT_DIR, "game_ops_analysis_v2.png")
    plt.savefig(analysis_plot_path, dpi=300)
    plt.close()



# ══════════════════════════════════════
# GAME OPS PREDICTOR CLASS (UPDATED)
# ══════════════════════════════════════

class GameOpsPredictor:
    def __init__(self):
        self._scaler = None
        self._skill_scaler = None
        self._iso_forest = None
        self._lof = None
        self._skill_model = None
        self._kmeans_models = {}
        self._label_encoders = {}
        self._feature_cols = None
        self._tier_thresholds = {}
        self._matchmaking_metadata = {}
        self._player_histories = {}
        self._confidence_config = {}
        self._cheat_type_rules = {}
        self._loaded = False

    def load_models(self):
        try:
            self._scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
            self._skill_scaler = joblib.load(os.path.join(MODELS_DIR, "skill_scaler.pkl"))
            self._iso_forest = joblib.load(os.path.join(MODELS_DIR, "isolation_forest_model.pkl"))
            self._lof = joblib.load(os.path.join(MODELS_DIR, "lof_model.pkl"))
            self._skill_model = joblib.load(os.path.join(MODELS_DIR, "skill_model.pkl"))
            self._label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
            self._feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
            self._tier_thresholds = joblib.load(os.path.join(MODELS_DIR, "tier_thresholds.pkl"))
            self._matchmaking_metadata = joblib.load(os.path.join(MODELS_DIR, "matchmaking_metadata.pkl"))
            
            # v2 Upgrades Load
            self._player_histories = joblib.load(os.path.join(MODELS_DIR, "player_history.pkl"))
            self._confidence_config = joblib.load(os.path.join(MODELS_DIR, "confidence_config.pkl"))
            self._cheat_type_rules = joblib.load(os.path.join(MODELS_DIR, "cheat_type_thresholds.pkl"))
            self._group_registry = joblib.load(os.path.join(MODELS_DIR, "group_registry.pkl"))
            self._mmr_scaler = joblib.load(os.path.join(MODELS_DIR, "mmr_scaler.pkl"))
            self._matchmaking_config = joblib.load(os.path.join(MODELS_DIR, "matchmaking_config.pkl"))
            
            for region in self._matchmaking_config['regions'].keys():
                for pool_cat in ["Cross", "Mobile"]:
                    model_name = f"kmeans_{region}_{pool_cat}.pkl"
                    self._kmeans_models[(region, pool_cat)] = joblib.load(os.path.join(MODELS_DIR, model_name))
            
            self._loaded = True
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            self._loaded = False
            return False

    def is_loaded(self) -> bool:
        return self._loaded

    def get_history(self, player_id: str) -> PlayerHistory:
        if player_id not in self._player_histories:
            self._player_histories[player_id] = PlayerHistory(player_id)
        return self._player_histories[player_id]

    def update_history(self, player_id: str, new_match_result: dict) -> None:
        history = self.get_history(player_id)
        history.add_match(
            match_id=new_match_result['match_id'],
            score=new_match_result['score'],
            kills=new_match_result['kills'],
            deaths=new_match_result['deaths'],
            score_per_minute=new_match_result['score_per_minute'],
            kdr=new_match_result['kdr'],
            was_flagged=new_match_result['was_flagged'],
            confidence_score=new_match_result['confidence_score'],
            timestamp=new_match_result.get('timestamp', datetime.datetime.now(datetime.timezone.utc).isoformat())
        )
        joblib.dump(self._player_histories, os.path.join(MODELS_DIR, "player_history.pkl"))

    def predict_player(self, player_data: dict, history=None) -> dict:
        if not self._loaded:
            raise RuntimeError("Models have not been loaded. Call load_models() first.")

        p_id = player_data['player_id']
        if history is None:
            history = self.get_history(p_id)

        # Feature Engineering
        kills = player_data['kills']
        deaths = player_data['deaths']
        score = player_data['score']
        ping = player_data['ping']
        duration = player_data['match_duration_seconds']
        region = player_data['region']
        device = player_data['device']

        kdr = kills / (deaths + 1)
        score_per_minute = score / (duration / 60.0) if duration > 0 else 0
        kill_rate = kills / (duration / 60.0) if duration > 0 else 0
        efficiency = score / (kills + 1)
        death_rate = deaths / (duration / 60.0) if duration > 0 else 0
        survival_index = 1.0 / (death_rate + 0.01)
        score_kill_ratio = score / (kills + 1)
        ping_adjusted_score = score / (1.0 + ping / 100.0)
        performance_index = score_per_minute / (kill_rate + 0.1)

        try:
            region_encoded = self._label_encoders['region'].transform([region])[0]
        except Exception:
            region_encoded = 0

        try:
            device_encoded = self._label_encoders['device'].transform([device])[0]
        except Exception:
            device_encoded = 0

        fv = {
            "kdr": kdr, "score_per_minute": score_per_minute, "kill_rate": kill_rate,
            "efficiency": efficiency, "death_rate": death_rate, "survival_index": survival_index,
            "score_kill_ratio": score_kill_ratio, "ping_adjusted_score": ping_adjusted_score,
            "performance_index": performance_index, "ping": ping,
            "region_encoded": region_encoded, "device_encoded": device_encoded
        }

        fv_df = pd.DataFrame([fv])[self._feature_cols]
        scaled_fv = self._scaler.transform(fv_df)

        # ML Predictions
        iso_pred = self._iso_forest.predict(scaled_fv)[0]
        iso_score = self._iso_forest.decision_function(scaled_fv)[0]
        lof_pred = self._lof.predict(scaled_fv)[0]
        lof_score = self._lof.score_samples(scaled_fv)[0]

        # Use temporary Random Forest to guess skill score to determine tier ceiling
        predicted_skill_rough = float(self._skill_model.predict(scaled_fv)[0])
        q90 = self._tier_thresholds['q90']
        q70 = self._tier_thresholds['q70']
        q40 = self._tier_thresholds['q40']
        q15 = self._tier_thresholds['q15']
        if predicted_skill_rough >= q90:
            predicted_tier = "Pro"
        elif predicted_skill_rough >= q70:
            predicted_tier = "Platinum"
        elif predicted_skill_rough >= q40:
            predicted_tier = "Gold"
        elif predicted_skill_rough >= q15:
            predicted_tier = "Silver"
        else:
            predicted_tier = "Bronze"

        fv_full = fv.copy()
        fv_full['predicted_skill_tier'] = predicted_tier
        fv_full['score'] = score
        fv_full['kills'] = kills
        fv_full['deaths'] = deaths
        fv_full['match_duration_seconds'] = duration
        fv_full['region'] = region
        fv_full['device'] = device

        # Compute final confidence score
        c_res = compute_confidence_score(
            fv_full, history, iso_score, lof_score, iso_pred, lof_pred,
            self._confidence_config.get('tier_ceilings', {}),
            self._confidence_config.get('device_limits', {})
        )

        conf_score = c_res['confidence_score']
        zone, status_lbl, act = get_confidence_zone_and_action(conf_score)

        # Skill & Matchmaking if confidence score <= 40
        skill_score = None
        skill_tier = "Excluded (score > 40)"
        if conf_score >= 61:
            skill_tier = "Excluded (score >= 61)"
        match_group = None
        match_group_reason = None

        if conf_score <= 40:
            skill_score = float(self._skill_model.predict(scaled_fv)[0])
            if skill_score >= q90: skill_tier = "Pro"
            elif skill_score >= q70: skill_tier = "Platinum"
            elif skill_score >= q40: skill_tier = "Gold"
            elif skill_score >= q15: skill_tier = "Silver"
            else: skill_tier = "Bronze"

            # 1. Normalize skill score using global mmr_scaler
            skill_norm = float(self._mmr_scaler.transform([[skill_score]])[0][0])
            
            # 2. Compute MMR
            consistency = history.consistency_score if history is not None else 1.0
            mmr_val = (0.60 * skill_norm) + (0.25 * consistency) + (0.15 * (1.0 - conf_score / 100.0))
            
            # Apply device multiplier
            device_multipliers = {"PC": 0.93, "Console": 0.96, "iOS": 1.00, "Android": 1.02}
            mmr_val = mmr_val * device_multipliers.get(device, 1.0)
            
            pool_cat = "Cross" if device in ["PC", "Console"] else "Mobile"
            
            # Find the target group matching region + pool category + predicted cluster
            if (region, pool_cat) in self._kmeans_models and region in self._matchmaking_config['regions'] and pool_cat in self._matchmaking_config['regions'][region]:
                meta = self._matchmaking_config['regions'][region][pool_cat]
                kmeans_model = self._kmeans_models[(region, pool_cat)]
                
                skill_norm_2d = (skill_score - meta['skill_min']) / (meta['skill_max'] - meta['skill_min'] + 1e-8)
                ping_norm_2d = (ping - meta['ping_min']) / (meta['ping_max'] - meta['ping_min'] + 1e-8)
                
                cluster_id = int(kmeans_model.predict([[skill_norm_2d, ping_norm_2d]])[0])
                target_gid = self._matchmaking_config['cluster_to_group'].get((region, pool_cat, cluster_id))
                if target_gid is None:
                    target_gid = f"{region}_{pool_cat}_G1"
            else:
                target_gid = f"{region}_{pool_cat}_G1"
                
            if target_gid not in self._group_registry:
                self._group_registry[target_gid] = {
                    "group_id": target_gid,
                    "region": region,
                    "player_count": 0,
                    "players": [],
                    "avg_mmr": 0.0,
                    "mmr_spread": 0.0,
                    "avg_ping": 0.0,
                    "ping_spread": 0.0,
                    "avg_confidence": 0.0,
                    "fairness_score": 100.0,
                    "quality_label": "Balanced",
                    "device_breakdown": {},
                    "device_flag": "balanced",
                    "skill_tiers_present": [],
                    "player_records": []
                }
                
            target_group = self._group_registry[target_gid]
            existing_records = target_group.get('player_records', [])
            
            is_ping_outlier = False
            if len(existing_records) > 0:
                pings = [p['ping'] for p in existing_records]
                ping_mean = np.mean(pings)
                ping_std = np.std(pings)
                if ping_std > 0 and abs(ping - ping_mean) > 1.5 * ping_std:
                    is_ping_outlier = True
                    
            simulated_player = {
                "player_id": p_id,
                "mmr": mmr_val,
                "ping": ping,
                "confidence_score": conf_score,
                "device": device,
                "skill_tier": skill_tier
            }
            
            if is_ping_outlier:
                best_gid = f"{region}_{pool_cat}_GPing"
                if best_gid not in self._group_registry:
                    self._group_registry[best_gid] = {
                        "group_id": best_gid,
                        "region": region,
                        "player_count": 0,
                        "players": [],
                        "avg_mmr": 0.0,
                        "mmr_spread": 0.0,
                        "avg_ping": 0.0,
                        "ping_spread": 0.0,
                        "avg_confidence": 0.0,
                        "fairness_score": 100.0,
                        "quality_label": "Balanced",
                        "device_breakdown": {},
                        "device_flag": "balanced",
                        "skill_tiers_present": [],
                        "player_records": []
                    }
                best_group = self._group_registry[best_gid]
                simulated_records = best_group.get('player_records', []) + [simulated_player]
                best_metrics = compute_group_metrics(simulated_records, is_unresolved=False)
            else:
                simulated_records = existing_records + [simulated_player]
                
                total = len(simulated_records)
                pc_count = sum(1 for p in simulated_records if p['device'] == 'PC')
                pc_ratio = pc_count / total if total > 0 else 0.0
                is_unresolved = (pc_ratio > 0.65) and (total > 4)
                
                sim_metrics = compute_group_metrics(simulated_records, is_unresolved=is_unresolved)
                degradation = target_group.get('fairness_score', 100.0) - sim_metrics['fairness_score']
                
                best_gid = target_gid
                best_metrics = sim_metrics
                
                if degradation > 10.0:
                    best_degradation = degradation
                    other_gids = [gid for gid in self._group_registry.keys() 
                                  if gid.startswith(f"{region}_{pool_cat}_G") and not gid.endswith("_GPing") and gid != target_gid]
                    
                    for other_gid in other_gids:
                        og_group = self._group_registry[other_gid]
                        og_records = og_group.get('player_records', [])
                        og_sim_records = og_records + [simulated_player]
                        
                        og_total = len(og_sim_records)
                        og_pc_count = sum(1 for p in og_sim_records if p['device'] == 'PC')
                        og_pc_ratio = og_pc_count / og_total if og_total > 0 else 0.0
                        og_is_unresolved = (og_pc_ratio > 0.65) and (og_total > 4)
                        
                        og_sim_metrics = compute_group_metrics(og_sim_records, is_unresolved=og_is_unresolved)
                        og_degradation = og_group.get('fairness_score', 100.0) - og_sim_metrics['fairness_score']
                        
                        if og_degradation < best_degradation:
                            best_degradation = og_degradation
                            best_gid = other_gid
                            best_metrics = og_sim_metrics
                            
            match_group = best_gid
            best_group = self._group_registry[best_gid]
            if 'player_records' not in best_group:
                best_group['player_records'] = []
            best_group['player_records'].append(simulated_player)
            best_group['players'].append(p_id)
            best_group['player_count'] = len(best_group['player_records'])
            
            for k, v in best_metrics.items():
                best_group[k] = v
                
            joblib.dump(self._group_registry, os.path.join(MODELS_DIR, "group_registry.pkl"))
        else:
            match_group = "EXCLUDED"
            match_group_reason = f"confidence_score too high: {int(conf_score)}"

        # Return dict matching enriched output requirements
        return {
            "player_id": p_id,
            "match_id": player_data['match_id'],
            "confidence_score": conf_score,
            "confidence_zone": zone,
            "status_label": status_lbl,
            "status": zone.lower(),
            "action": act,
            "cheat_types_hit": c_res['cheat_types_hit'],
            "confirmed_cheats": c_res['confirmed_cheats'],
            "unconfirmed_hits": c_res['unconfirmed_hits'],
            "score_breakdown": c_res['score_breakdown'],
            "history_summary": {
                "total_matches": history.total_matches,
                "flag_rate": history.flag_rate,
                "trend": history.trend,
                "consistency_score": history.consistency_score,
                "veteran_status": history.veteran_status
            },
            "skill_score": skill_score,
            "skill_tier": skill_tier,
            "match_group": match_group,
            "match_group_reason": match_group_reason,
            "computed_features": {
                "kdr": float(kdr),
                "score_per_min": float(score_per_minute),
                "kill_rate": float(kill_rate),
                "efficiency": float(efficiency)
            },
            "final_flagged": conf_score >= 61,
            "iso_anomaly_score": float(iso_score),
            "lof_anomaly_score": float(lof_score),
            "flag_reason": "ml_anomaly" if conf_score >= 61 else None,
            "flag_source": "ensemble" if conf_score >= 81 else "rule" if conf_score >= 61 else None
        }

    def predict_players_batch(self, players_data: list) -> list:
        if not self._loaded:
            raise RuntimeError("Models have not been loaded. Call load_models() first.")

        if not players_data:
            return []

        df = pd.DataFrame(players_data)
        df['kdr'] = df['kills'] / (df['deaths'] + 1)
        dur_min = df['match_duration_seconds'] / 60.0
        df['score_per_minute'] = np.where(df['match_duration_seconds'] > 0, df['score'] / dur_min, 0.0)
        df['kill_rate'] = np.where(df['match_duration_seconds'] > 0, df['kills'] / dur_min, 0.0)
        df['efficiency'] = df['score'] / (df['kills'] + 1)
        df['death_rate'] = np.where(df['match_duration_seconds'] > 0, df['deaths'] / dur_min, 0.0)
        df['survival_index'] = 1.0 / (df['death_rate'] + 0.01)
        df['score_kill_ratio'] = df['score'] / (df['kills'] + 1)
        df['ping_adjusted_score'] = df['score'] / (1.0 + df['ping'] / 100.0)
        df['performance_index'] = df['score_per_minute'] / (df['kill_rate'] + 0.1)

        classes_region = {c: i for i, c in enumerate(self._label_encoders['region'].classes_)}
        df['region_encoded'] = df['region'].map(lambda x: classes_region.get(x, 0))
        
        classes_dev = {c: i for i, c in enumerate(self._label_encoders['device'].classes_)}
        df['device_encoded'] = df['device'].map(lambda x: classes_dev.get(x, 0))

        fv_df = df[self._feature_cols]
        scaled_fv = self._scaler.transform(fv_df)

        iso_preds = self._iso_forest.predict(scaled_fv)
        iso_scores = self._iso_forest.decision_function(scaled_fv)
        lof_preds = self._lof.predict(scaled_fv)
        lof_scores = self._lof.score_samples(scaled_fv)
        skill_scores = self._skill_model.predict(scaled_fv)

        q90 = self._tier_thresholds['q90']
        q70 = self._tier_thresholds['q70']
        q40 = self._tier_thresholds['q40']
        q15 = self._tier_thresholds['q15']

        results = []
        for idx, row in df.iterrows():
            p_id = row['player_id']
            history = self.get_history(p_id)
            
            predicted_skill_rough = float(skill_scores[idx])
            if predicted_skill_rough >= q90:
                predicted_tier = "Pro"
            elif predicted_skill_rough >= q70:
                predicted_tier = "Platinum"
            elif predicted_skill_rough >= q40:
                predicted_tier = "Gold"
            elif predicted_skill_rough >= q15:
                predicted_tier = "Silver"
            else:
                predicted_tier = "Bronze"

            fv_full = row.to_dict()
            fv_full['predicted_skill_tier'] = predicted_tier

            c_res = compute_confidence_score(
                fv_full, history, float(iso_scores[idx]), float(lof_scores[idx]),
                int(iso_preds[idx]), int(lof_preds[idx]),
                self._confidence_config.get('tier_ceilings', {}),
                self._confidence_config.get('device_limits', {})
            )

            conf_score = c_res['confidence_score']
            zone, status_lbl, act = get_confidence_zone_and_action(conf_score)

            skill_score = None
            skill_tier = "Excluded (score > 40)"
            if conf_score >= 61:
                skill_tier = "Excluded (score >= 61)"
            match_group = None
            match_group_reason = None

            if conf_score <= 40:
                skill_score = float(skill_scores[idx])
                if skill_score >= q90: skill_tier = "Pro"
                elif skill_score >= q70: skill_tier = "Platinum"
                elif skill_score >= q40: skill_tier = "Gold"
                elif skill_score >= q15: skill_tier = "Silver"
                else: skill_tier = "Bronze"

                region = row['region']
                ping = int(row['ping'])
                device = row['device']
                
                skill_norm = float(self._mmr_scaler.transform([[skill_score]])[0][0])
                
                consistency = history.consistency_score if history is not None else 1.0
                mmr_val = (0.60 * skill_norm) + (0.25 * consistency) + (0.15 * (1.0 - conf_score / 100.0))
                
                device_multipliers = {"PC": 0.93, "Console": 0.96, "iOS": 1.00, "Android": 1.02}
                mmr_val = mmr_val * device_multipliers.get(device, 1.0)
                
                pool_cat = "Cross" if device in ["PC", "Console"] else "Mobile"
                
                if (region, pool_cat) in self._kmeans_models and region in self._matchmaking_config['regions'] and pool_cat in self._matchmaking_config['regions'][region]:
                    meta = self._matchmaking_config['regions'][region][pool_cat]
                    kmeans_model = self._kmeans_models[(region, pool_cat)]
                    
                    skill_norm_2d = (skill_score - meta['skill_min']) / (meta['skill_max'] - meta['skill_min'] + 1e-8)
                    ping_norm_2d = (ping - meta['ping_min']) / (meta['ping_max'] - meta['ping_min'] + 1e-8)
                    
                    cluster_id = int(kmeans_model.predict([[skill_norm_2d, ping_norm_2d]])[0])
                    target_gid = self._matchmaking_config['cluster_to_group'].get((region, pool_cat, cluster_id))
                    if target_gid is None:
                        target_gid = f"{region}_{pool_cat}_G1"
                else:
                    target_gid = f"{region}_{pool_cat}_G1"
                    
                if target_gid not in self._group_registry:
                    self._group_registry[target_gid] = {
                        "group_id": target_gid,
                        "region": region,
                        "player_count": 0,
                        "players": [],
                        "avg_mmr": 0.0,
                        "mmr_spread": 0.0,
                        "avg_ping": 0.0,
                        "ping_spread": 0.0,
                        "avg_confidence": 0.0,
                        "fairness_score": 100.0,
                        "quality_label": "Balanced",
                        "device_breakdown": {},
                        "device_flag": "balanced",
                        "skill_tiers_present": [],
                        "player_records": []
                    }
                    
                target_group = self._group_registry[target_gid]
                existing_records = target_group.get('player_records', [])
                
                is_ping_outlier = False
                if len(existing_records) > 0:
                    pings = [p['ping'] for p in existing_records]
                    ping_mean = np.mean(pings)
                    ping_std = np.std(pings)
                    if ping_std > 0 and abs(ping - ping_mean) > 1.5 * ping_std:
                        is_ping_outlier = True
                        
                simulated_player = {
                    "player_id": p_id,
                    "mmr": mmr_val,
                    "ping": ping,
                    "confidence_score": conf_score,
                    "device": device,
                    "skill_tier": skill_tier
                }
                
                if is_ping_outlier:
                    best_gid = f"{region}_{pool_cat}_GPing"
                    if best_gid not in self._group_registry:
                        self._group_registry[best_gid] = {
                            "group_id": best_gid,
                            "region": region,
                            "player_count": 0,
                            "players": [],
                            "avg_mmr": 0.0,
                            "mmr_spread": 0.0,
                            "avg_ping": 0.0,
                            "ping_spread": 0.0,
                            "avg_confidence": 0.0,
                            "fairness_score": 100.0,
                            "quality_label": "Balanced",
                            "device_breakdown": {},
                            "device_flag": "balanced",
                            "skill_tiers_present": [],
                            "player_records": []
                        }
                    best_group = self._group_registry[best_gid]
                    simulated_records = best_group.get('player_records', []) + [simulated_player]
                    best_metrics = compute_group_metrics(simulated_records, is_unresolved=False)
                else:
                    simulated_records = existing_records + [simulated_player]
                    
                    total = len(simulated_records)
                    pc_count = sum(1 for p in simulated_records if p['device'] == 'PC')
                    pc_ratio = pc_count / total if total > 0 else 0.0
                    is_unresolved = (pc_ratio > 0.65) and (total > 4)
                    
                    sim_metrics = compute_group_metrics(simulated_records, is_unresolved=is_unresolved)
                    degradation = target_group.get('fairness_score', 100.0) - sim_metrics['fairness_score']
                    
                    best_gid = target_gid
                    best_metrics = sim_metrics
                    
                    if degradation > 10.0:
                        best_degradation = degradation
                        other_gids = [gid for gid in self._group_registry.keys() 
                                      if gid.startswith(f"{region}_{pool_cat}_G") and not gid.endswith("_GPing") and gid != target_gid]
                        
                        for other_gid in other_gids:
                            og_group = self._group_registry[other_gid]
                            og_records = og_group.get('player_records', [])
                            og_sim_records = og_records + [simulated_player]
                            
                            og_total = len(og_sim_records)
                            og_pc_count = sum(1 for p in og_sim_records if p['device'] == 'PC')
                            og_pc_ratio = og_pc_count / og_total if og_total > 0 else 0.0
                            og_is_unresolved = (og_pc_ratio > 0.65) and (og_total > 4)
                            
                            og_sim_metrics = compute_group_metrics(og_sim_records, is_unresolved=og_is_unresolved)
                            og_degradation = og_group.get('fairness_score', 100.0) - og_sim_metrics['fairness_score']
                            
                            if og_degradation < best_degradation:
                                best_degradation = og_degradation
                                best_gid = other_gid
                                best_metrics = og_sim_metrics
                                
                match_group = best_gid
                best_group = self._group_registry[best_gid]
                if 'player_records' not in best_group:
                    best_group['player_records'] = []
                best_group['player_records'].append(simulated_player)
                best_group['players'].append(p_id)
                best_group['player_count'] = len(best_group['player_records'])
                
                for k, v in best_metrics.items():
                    best_group[k] = v
            else:
                match_group = "EXCLUDED"
                match_group_reason = f"confidence_score too high: {int(conf_score)}"

            results.append({
                "player_id": p_id,
                "match_id": row['match_id'],
                "confidence_score": conf_score,
                "confidence_zone": zone,
                "status_label": status_lbl,
                "status": zone.lower(),
                "action": act,
                "cheat_types_hit": c_res['cheat_types_hit'],
                "confirmed_cheats": c_res['confirmed_cheats'],
                "unconfirmed_hits": c_res['unconfirmed_hits'],
                "score_breakdown": c_res['score_breakdown'],
                "history_summary": {
                    "total_matches": history.total_matches,
                    "flag_rate": history.flag_rate,
                    "trend": history.trend,
                    "consistency_score": history.consistency_score,
                    "veteran_status": history.veteran_status
                },
                "skill_score": skill_score,
                "skill_tier": skill_tier,
                "match_group": match_group,
                "match_group_reason": match_group_reason,
                "computed_features": {
                    "kdr": float(row['kdr']),
                    "score_per_min": float(row['score_per_minute']),
                    "kill_rate": float(row['kill_rate']),
                    "efficiency": float(row['efficiency'])
                },
                "final_flagged": conf_score >= 61,
                "iso_anomaly_score": float(iso_scores[idx]),
                "lof_anomaly_score": float(lof_scores[idx]),
                "flag_reason": "ml_anomaly" if conf_score >= 61 else None,
                "flag_source": "ensemble" if conf_score >= 81 else "rule" if conf_score >= 61 else None
            })
            
        joblib.dump(self._group_registry, os.path.join(MODELS_DIR, "group_registry.pkl"))
        return results


    def predict_batch(self, csv_path: str) -> pd.DataFrame:
        df_batch = pd.read_csv(csv_path, keep_default_na=False)
        players_data = []
        for _, row in df_batch.iterrows():
            players_data.append({
                "player_id": row['player_id'],
                "match_id": row['match_id'],
                "region": row['region'],
                "device": row['device'],
                "ping": int(row['ping']),
                "score": int(row['score']),
                "kills": int(row['kills']),
                "deaths": int(row['deaths']),
                "match_duration_seconds": int(row['match_duration_seconds'])
            })
            
        predictions = self.predict_players_batch(players_data)

        df_out = df_batch.copy()
        df_out['confidence_score'] = [p['confidence_score'] for p in predictions]
        df_out['confidence_zone'] = [p['confidence_zone'] for p in predictions]
        df_out['status_label'] = [p['status_label'] for p in predictions]
        df_out['status'] = [p['status'] for p in predictions]
        df_out['action'] = [p['action'] for p in predictions]
        df_out['skill_score'] = [p['skill_score'] for p in predictions]
        df_out['skill_tier'] = [p['skill_tier'] for p in predictions]
        df_out['match_group_id'] = [p['match_group'] for p in predictions]
        df_out['final_flagged'] = [p['final_flagged'] for p in predictions]
        
        output_csv_path = os.path.join(OUTPUT_DIR, "predictions_output.csv")
        df_out.to_csv(output_csv_path, index=False)
        print(f"Batch predictions saved to: {output_csv_path}")
        return df_out


# ══════════════════════════════════════
# PHASE 10: DEMO RUN
# ══════════════════════════════════════

def run_demos(predictor):
    print("\n" + "=" * 50)
    print("RUNNING UPGRADED PIPELINE DEMOS")
    print("=" * 50)

    # Demo 1 — 5 clean players (confidence should be 0–25)
    print("\n--- Demo 1: Predict 5 Clean Players ---")
    clean_players = [
        {"player_id": "PC001", "match_id": "M001", "region": "India", "device": "Android", "ping": 45, "score": 3200, "kills": 18, "deaths": 4, "match_duration_seconds": 420},
        {"player_id": "PC002", "match_id": "M002", "region": "SEA", "device": "iOS", "ping": 25, "score": 4500, "kills": 24, "deaths": 2, "match_duration_seconds": 380},
        {"player_id": "PC003", "match_id": "M003", "region": "Europe", "device": "PC", "ping": 60, "score": 2800, "kills": 10, "deaths": 5, "match_duration_seconds": 600},
        {"player_id": "PC004", "match_id": "M004", "region": "NA", "device": "Console", "ping": 75, "score": 1500, "kills": 6, "deaths": 8, "match_duration_seconds": 480},
        {"player_id": "PC005", "match_id": "M005", "region": "LatAm", "device": "Android", "ping": 90, "score": 900, "kills": 2, "deaths": 10, "match_duration_seconds": 300}
    ]

    for p in clean_players:
        res = predictor.predict_player(p)
        print("─────────────────────────────────────────")
        print(f"Player        : {res['player_id']}")
        print(f"Confidence    : {res['confidence_score']:.0f} / 100  [{res['status_label']}]")
        print(f"Action        : {res['action']}")
        print(f"Cheat types   : {', '.join(res['cheat_types_hit']) if res['cheat_types_hit'] else 'None'}")
        print(f"Score detail  : {res['score_breakdown']}")
        print(f"History       : {res['history_summary']['total_matches']} matches | flag_rate={res['history_summary']['flag_rate']:.0%} | trend={res['history_summary']['trend']}")
        print(f"Skill tier    : {res['skill_tier']}")
        print("─────────────────────────────────────────")

    # Demo 2 — 10 cheaters showing all 10 cheat types
    print("\n--- Demo 2: Predict 10 Cheaters (Each Cheat Type) ---")
    
    # We define cheaters matching rules
    cheaters = [
        # 1. score_bot
        {"player_id": "PC_BOT_01", "match_id": "MC01", "region": "India", "device": "PC", "ping": 20, "score": 15000, "kills": 15, "deaths": 2, "match_duration_seconds": 60},
        # 2. kill_farmer
        {"player_id": "PC_KARM_02", "match_id": "MC02", "region": "SEA", "device": "Android", "ping": 50, "score": 5000, "kills": 50, "deaths": 1, "match_duration_seconds": 150},
        # 3. god_mode
        {"player_id": "PC_GOD_03", "match_id": "MC03", "region": "Europe", "device": "iOS", "ping": 40, "score": 8000, "kills": 65, "deaths": 0, "match_duration_seconds": 400},
        # 4. time_exploit
        {"player_id": "PC_TIME_04", "match_id": "MC04", "region": "NA", "device": "Console", "ping": 80, "score": 25000, "kills": 12, "deaths": 1, "match_duration_seconds": 45},
        # 5. speed_hack
        {"player_id": "PC_SPD_05", "match_id": "MC05", "region": "LatAm", "device": "PC", "ping": 60, "score": 4000, "kills": 25, "deaths": 1, "match_duration_seconds": 100},
        # 6. soft_cheat
        {"player_id": "PC_SOFT_06", "match_id": "MC06", "region": "Middle_East", "device": "Android", "ping": 100, "score": 3500, "kills": 28, "deaths": 1, "match_duration_seconds": 480},
        # 7. score_inflate
        {"player_id": "PC_INFL_07", "match_id": "MC07", "region": "India", "device": "iOS", "ping": 30, "score": 12000, "kills": 8, "deaths": 2, "match_duration_seconds": 300},
        # 8. stat_padding
        {"player_id": "PC_PAD_08", "match_id": "MC08", "region": "Europe", "device": "Console", "ping": 55, "score": 8000, "kills": 45, "deaths": 2, "match_duration_seconds": 750},
        # 9. region_spoof
        {"player_id": "PC_SPOOF_09", "match_id": "MC09", "region": "LatAm", "device": "PC", "ping": 8, "score": 3000, "kills": 12, "deaths": 3, "match_duration_seconds": 300},
        # 10. burst_cheat
        {"player_id": "PC_BURST_10", "match_id": "MC10", "region": "Middle_East", "device": "Android", "ping": 70, "score": 5000, "kills": 20, "deaths": 2, "match_duration_seconds": 300}
    ]

    for p in cheaters:
        # For Type 10 burst_cheat we explicitly remove its history to trigger it
        if p['player_id'] == "PC_BURST_10" and p['player_id'] in predictor._player_histories:
            del predictor._player_histories[p['player_id']]
            
        res = predictor.predict_player(p)
        print("─────────────────────────────────────────")
        print(f"Player        : {res['player_id']}")
        print(f"Confidence    : {res['confidence_score']:.0f} / 100  [{res['status_label']}]")
        print(f"Action        : {res['action']}")
        print(f"Cheat types   : {', '.join(res['cheat_types_hit']) if res['cheat_types_hit'] else 'None'}")
        print(f"Score detail  : {res['score_breakdown']}")
        print(f"History       : {res['history_summary']['total_matches']} matches | flag_rate={res['history_summary']['flag_rate']:.0%} | trend={res['history_summary']['trend']}")
        print(f"Skill tier    : {res['skill_tier']}")
        print("─────────────────────────────────────────")

    # Demo 3 — History impact demo
    print("\n--- Demo 3: History Impact Demo ---")
    
    # We define the base stats
    base_p = {"player_id": "TEMP_ID", "match_id": "M_TEST", "region": "India", "device": "PC", "ping": 40, "score": 3000, "kills": 10, "deaths": 4, "match_duration_seconds": 400}
    
    # Profile A: 15 clean matches (all score=3000, kills=10, deaths=4, spm=450, no flag)
    profile_a = PlayerHistory("P_A_VET")
    for i in range(15):
        profile_a.add_match(f"MA_{i}", 3000, 10, 4, 450.0, 2.0, False, 10.0, "")
    profile_a.recalculate()
    
    # Profile B: 8 matches, 4 flagged (flag_rate=0.5)
    profile_b = PlayerHistory("P_B_OFF")
    for i in range(8):
        flag = (i % 2 == 0)
        profile_b.add_match(f"MB_{i}", 3000 if not flag else 8000, 10 if not flag else 50, 4, 450.0 if not flag else 1200.0, 2.0, flag, 10.0 if not flag else 85.0, "")
    profile_b.recalculate()
    
    # Profile C: suspicious_spike trend (first 2 spm=200, last 2 spm=800)
    profile_c = PlayerHistory("P_C_SPIKE")
    profile_c.add_match("MC_1", 1000, 4, 4, 150.0, 1.0, False, 5.0, "")
    profile_c.add_match("MC_2", 1000, 4, 4, 150.0, 1.0, False, 5.0, "")
    profile_c.add_match("MC_3", 5000, 20, 2, 750.0, 10.0, False, 15.0, "")
    profile_c.add_match("MC_4", 6000, 24, 2, 900.0, 12.0, False, 20.0, "")
    profile_c.recalculate()
    
    for label, hist in [("Profile A (Veteran Status)", profile_a), ("Profile B (Repeat Offender)", profile_b), ("Profile C (Suspicious Spike)", profile_c)]:
        p_data = base_p.copy()
        p_data['player_id'] = hist.player_id
        res = predictor.predict_player(p_data, history=hist)
        print("─────────────────────────────────────────")
        print(f"Profile       : {label}")
        print(f"Player        : {res['player_id']}")
        print(f"Confidence    : {res['confidence_score']:.0f} / 100  [{res['status_label']}]")
        print(f"Score detail  : {res['score_breakdown']}")
        print(f"History Summary: {res['history_summary']['total_matches']} matches | flag_rate={res['history_summary']['flag_rate']:.0%} | trend={res['history_summary']['trend']} | consistency={res['history_summary']['consistency_score']:.2f}")
        print("─────────────────────────────────────────")


if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == "--demo-only" or sys.argv[1] == "--demo"):
        predictor = GameOpsPredictor()
        predictor.load_models()
        run_demos(predictor)
    elif len(sys.argv) > 1 and sys.argv[1] == "--with-demo":
        train_pipeline(DATA_PATH)
        predictor = GameOpsPredictor()
        predictor.load_models()
        run_demos(predictor)
    else:
        train_pipeline(DATA_PATH)
