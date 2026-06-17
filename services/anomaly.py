import sys
import os
# Ensure parent directory is in the path to run script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from game_ops.services.data_engine import generate_dataset, FEATURE_COLS

def run_anomaly_detection(df):
    """
    Runs a two-layer cheater detection system on the player dataset.
    
    Layer 1: Z-score Statistical Flagging on FEATURE_COLS (> 3.0 std above mean).
    Layer 2: Isolation Forest ML on FEATURE_COLS.
    
    Args:
        df (pd.DataFrame): The input player dataset with engineered features.
        
    Returns:
        tuple: (clean_df, flagged_df)
            - clean_df: DataFrame containing only legit (unflagged) players.
            - flagged_df: DataFrame containing all flagged players.
    """
    df_copy = df.copy()
    
    # --- LAYER 1: Z-score Statistical Flagging ---
    z_scores = pd.DataFrame()
    for col in FEATURE_COLS:
        mean_val = df_copy[col].mean()
        std_val = df_copy[col].std()
        if std_val == 0:
            z_scores[col] = 0.0
        else:
            z_scores[col] = (df_copy[col] - mean_val) / std_val
            
    flagged_z = (z_scores > 3.0).any(axis=1)
    
    # --- LAYER 2: Isolation Forest ML ---
    clf = IsolationForest(contamination=0.1, n_estimators=200, random_state=42)
    predictions = clf.fit_predict(df_copy[FEATURE_COLS])
    flagged_if = (predictions == -1)
    
    # --- COMBINED ---
    flagged_combined = flagged_z | flagged_if
    
    # Split into clean and flagged DataFrames
    clean_df = df_copy[~flagged_combined].reset_index(drop=True)
    flagged_df = df_copy[flagged_combined].reset_index(drop=True)
    
    return clean_df, flagged_df

def get_suspicious_players_report(df):
    """
    Builds the suspicious_players DataFrame with detailed diagnostic info.
    """
    df_copy = df.copy()
    
    # Recalculate Z-scores for detailed reporting
    z_scores = pd.DataFrame()
    for col in FEATURE_COLS:
        mean_val = df_copy[col].mean()
        std_val = df_copy[col].std()
        if std_val == 0:
            z_scores[col] = 0.0
        else:
            z_scores[col] = (df_copy[col] - mean_val) / std_val
            
    flagged_z = (z_scores > 3.0).any(axis=1)
    
    # Fit Isolation Forest
    clf = IsolationForest(contamination=0.1, n_estimators=200, random_state=42)
    predictions = clf.fit_predict(df_copy[FEATURE_COLS])
    decision_scores = clf.decision_function(df_copy[FEATURE_COLS])
    flagged_if = (predictions == -1)
    
    suspicious_rows = []
    for idx, row in df_copy.iterrows():
        is_flagged_z = flagged_z[idx]
        is_flagged_if = flagged_if[idx]
        
        if is_flagged_z or is_flagged_if:
            reasons = []
            if is_flagged_z:
                reasons.append("Z-Score Flagging")
            if is_flagged_if:
                reasons.append("Isolation Forest")
            flag_reason = " & ".join(reasons)
            
            # Record Z-score triggers
            z_triggers = []
            for col in FEATURE_COLS:
                z_val = z_scores.loc[idx, col]
                if z_val > 3.0:
                    z_triggers.append(f"{col} (z={z_val:.2f})")
            z_score_trigger = ", ".join(z_triggers) if z_triggers else "N/A"
            
            # Suspicious stats summary
            stats_summary = (
                f"Score: {row['score']} | Kills: {row['kills']} | Deaths: {row['deaths']} | "
                f"Duration: {row['match_duration_seconds']}s | Ping: {row['ping']}ms"
            )
            
            suspicious_rows.append({
                'player_id': row['player_id'],
                'flag_reason': flag_reason,
                'z_score_trigger': z_score_trigger,
                'isolation_score': round(decision_scores[idx], 4),
                'suspicious_stats_summary': stats_summary
            })
            
    return pd.DataFrame(suspicious_rows)

def print_evaluation_report(df):
    """
    Computes and prints evaluation metrics for individual layers and the combined system.
    """
    df_copy = df.copy()
    
    # Z-scores
    z_scores = pd.DataFrame()
    for col in FEATURE_COLS:
        mean_val = df_copy[col].mean()
        std_val = df_copy[col].std()
        if std_val == 0:
            z_scores[col] = 0.0
        else:
            z_scores[col] = (df_copy[col] - mean_val) / std_val
            
    flagged_z = (z_scores > 3.0).any(axis=1).astype(int)
    
    # Isolation Forest
    clf = IsolationForest(contamination=0.1, n_estimators=200, random_state=42)
    predictions = clf.fit_predict(df_copy[FEATURE_COLS])
    flagged_if = (predictions == -1).astype(int)
    
    # Combined
    flagged_combined = (flagged_z | flagged_if).astype(int)
    
    y_true = df_copy['ground_truth_label'].astype(int)
    
    def calculate_metrics(y_pred):
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return prec, rec, f1
        
    p_z, r_z, f1_z = calculate_metrics(flagged_z)
    p_if, r_if, f1_if = calculate_metrics(flagged_if)
    p_c, r_c, f1_c = calculate_metrics(flagged_combined)
    
    cm = confusion_matrix(y_true, flagged_combined)
    
    print("=" * 120)
    print("DETECTION SYSTEM EVALUATION METRICS")
    print("=" * 120)
    print(f"{'Layer / Model':<35} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
    print("-" * 120)
    print(f"{'Layer 1: Z-score Flagging':<35} | {p_z:<12.4f} | {r_z:<12.4f} | {f1_z:<12.4f}")
    print(f"{'Layer 2: Isolation Forest':<35} | {p_if:<12.4f} | {r_if:<12.4f} | {f1_if:<12.4f}")
    print(f"{'Combined System (Either Layer)':<35} | {p_c:<12.4f} | {r_c:<12.4f} | {f1_c:<12.4f}")
    print("-" * 120)
    print("\nConfusion Matrix for Combined System:")
    print("                      Predicted Legit    Predicted Cheater")
    print(f"Actual Legit:         {cm[0,0]:<18} {cm[0,1]:<18}")
    print(f"Actual Cheater:       {cm[1,0]:<18} {cm[1,1]:<18}")
    print("=" * 120)

if __name__ == '__main__':
    # Generate dataset
    df = generate_dataset(random_state=42)
    
    # Configure pandas for nice output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    # Run anomaly detection
    clean_df, flagged_df = run_anomaly_detection(df)
    
    # Build report
    suspicious_players = get_suspicious_players_report(df)
    
    # Print Suspicious Players Table
    print("\n" + "=" * 120)
    print("SUSPICIOUS PLAYERS REPORT")
    print("=" * 120)
    print(suspicious_players.to_string(index=False))
    print("=" * 120)
    print(f"\nExcluding flagged players partitions dataset into: Clean ({len(clean_df)} players), Flagged ({len(flagged_df)} players).\n")
    
    # Print Evaluation
    print_evaluation_report(df)
