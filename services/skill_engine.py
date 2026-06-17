import sys
import os
# Ensure parent directory is in the path to run script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from game_ops.services.data_engine import generate_dataset
from game_ops.services.anomaly import run_anomaly_detection

def compute_skill_scores(clean_df):
    """
    Computes a composite skill score and assigns skill tiers based on percentiles.
    
    1. Normalizes kdr, score_per_minute, efficiency to 0-1.
    2. Computes skill_score = (0.35 * kdr_norm) + (0.35 * score_per_min_norm) + (0.30 * efficiency_norm)
    3. Categorizes players into skill tiers:
       - Top 10% (>= 90th percentile) -> Pro
       - 10-30% (>= 70th percentile to < 90th percentile) -> Platinum
       - 30-60% (>= 40th percentile to < 70th percentile) -> Gold
       - 60-85% (>= 15th percentile to < 40th percentile) -> Silver
       - Bottom 15% (< 15th percentile) -> Bronze
       
    Args:
        clean_df (pd.DataFrame): DataFrame containing clean (non-flagged) players.
        
    Returns:
        pd.DataFrame: The DataFrame with 'skill_score' and 'skill_tier' columns added.
    """
    df_copy = clean_df.copy()
    
    if len(df_copy) == 0:
        df_copy['skill_score'] = []
        df_copy['skill_tier'] = []
        return df_copy
        
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Extract features for scaling
    features_to_scale = ['kdr', 'score_per_minute', 'efficiency']
    scaled_features = scaler.fit_transform(df_copy[features_to_scale])
    
    kdr_norm = scaled_features[:, 0]
    score_per_min_norm = scaled_features[:, 1]
    efficiency_norm = scaled_features[:, 2]
    
    # Calculate composite skill score
    df_copy['skill_score'] = (0.35 * kdr_norm) + (0.35 * score_per_min_norm) + (0.30 * efficiency_norm)
    
    # Compute percentile thresholds
    q15 = df_copy['skill_score'].quantile(0.15)
    q40 = df_copy['skill_score'].quantile(0.40)
    q70 = df_copy['skill_score'].quantile(0.70)
    q90 = df_copy['skill_score'].quantile(0.90)
    
    # Assign tiers
    def assign_tier(score):
        if score >= q90:
            return 'Pro'
        elif score >= q70:
            return 'Platinum'
        elif score >= q40:
            return 'Gold'
        elif score >= q15:
            return 'Silver'
        else:
            return 'Bronze'
            
    df_copy['skill_tier'] = df_copy['skill_score'].apply(assign_tier)
    
    return df_copy

if __name__ == '__main__':
    # Generate original dataset
    df = generate_dataset(random_state=42)
    
    # Filter anomalies
    clean_df, flagged_df = run_anomaly_detection(df)
    
    # Compute skill scores
    scored_df = compute_skill_scores(clean_df)
    
    # Sort players by skill_score descending
    sorted_df = scored_df.sort_values(by='skill_score', ascending=False)
    
    # Configure pandas for nice output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    # Print sorted players
    print("=" * 120)
    print("PLAYER SKILL ENGINE - CLEAN PLAYERS RANKING")
    print("=" * 120)
    columns_to_show = ['player_id', 'region', 'device', 'score_per_minute', 'kdr', 'efficiency', 'skill_score', 'skill_tier']
    print(sorted_df[columns_to_show].to_string(index=False))
    print("=" * 120)
