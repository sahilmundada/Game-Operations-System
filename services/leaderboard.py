import sys
import os
# Ensure parent directory is in the path to run script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from game_ops.services.data_engine import generate_dataset
from game_ops.services.anomaly import run_anomaly_detection
from game_ops.services.skill_engine import compute_skill_scores
from game_ops.services.matchmaking import run_matchmaking

def create_leaderboards(clean_df):
    """
    Creates global and region-wise leaderboards for clean players.
    
    Ranks players by: score DESC, deaths ASC, kills DESC
    
    Args:
        clean_df (pd.DataFrame): DataFrame containing clean players with skill_tier and match_group_id.
        
    Returns:
        tuple: (global_leaderboard_df, region_leaderboard_df)
            - global_leaderboard_df: Global leaderboard.
            - region_leaderboard_df: Region-wise leaderboards combined.
    """
    if len(clean_df) == 0:
        return pd.DataFrame(), pd.DataFrame()
        
    # Sort globally by score DESC, deaths ASC, kills DESC
    sorted_df = clean_df.sort_values(
        by=['score', 'deaths', 'kills'],
        ascending=[False, True, False]
    ).reset_index(drop=True)
    
    # 1. Global Leaderboard
    global_lb = sorted_df.copy()
    global_lb['global_rank'] = global_lb.index + 1
    global_cols = ['global_rank', 'player_id', 'region', 'score', 'kills', 'deaths', 'skill_tier', 'match_group_id']
    global_lb = global_lb[global_cols]
    
    # 2. Region-wise Leaderboard
    region_dfs = []
    for region, region_players in sorted_df.groupby('region', sort=True):
        region_sorted = region_players.sort_values(
            by=['score', 'deaths', 'kills'],
            ascending=[False, True, False]
        ).reset_index(drop=True)
        region_sorted['region_rank'] = region_sorted.index + 1
        region_dfs.append(region_sorted)
        
    region_lb = pd.concat(region_dfs).reset_index(drop=True)
    region_cols = ['region_rank', 'player_id', 'region', 'score', 'kills', 'deaths', 'skill_tier', 'match_group_id']
    region_lb = region_lb[region_cols]
    
    return global_lb, region_lb

if __name__ == '__main__':
    # 1. Generate Dataset
    df = generate_dataset(random_state=42)
    
    # 2. Run Anomaly Detection (Filter cheaters)
    clean_df, flagged_df = run_anomaly_detection(df)
    
    # 3. Compute Skill Scores
    scored_df = compute_skill_scores(clean_df)
    
    # 4. Run Matchmaking
    matched_df = run_matchmaking(scored_df)
    
    # 5. Generate Leaderboards
    global_lb, region_lb = create_leaderboards(matched_df)
    
    # Configure pandas for nice output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    # Print Global Leaderboard
    print("=" * 120)
    print("GLOBAL LEADERBOARD")
    print("=" * 120)
    print(global_lb.to_string(index=False))
    print("=" * 120)
    
    # Print Region-wise Leaderboard
    print("\n" + "=" * 120)
    print("REGION-WISE LEADERBOARD")
    print("=" * 120)
    # Print region by region for readability
    for region, group_df in region_lb.groupby('region'):
        print(f"\n--- {region.upper()} REGION ---")
        print(group_df.to_string(index=False))
    print("=" * 120)
