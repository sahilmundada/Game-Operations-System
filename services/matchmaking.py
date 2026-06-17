import sys
import os
# Ensure parent directory is in the path to run script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from game_ops.services.data_engine import generate_dataset
from game_ops.services.anomaly import run_anomaly_detection
from game_ops.services.skill_engine import compute_skill_scores

def run_matchmaking(df):
    """
    Groups players into balanced matches based on region, skill score, and ping.
    
    1. Splits players by region.
    2. Performs KMeans clustering within each region based on skill_score and normalized ping.
       n_clusters = max(1, len(region_players) // 4)
    3. Filters clusters to ensure no two players in a group differ by > 80ms ping.
       If they do, they are split greedily.
    4. Assigns match_group_id like "{Region}_G{index}".
    
    Args:
        df (pd.DataFrame): DataFrame containing players with skill_score and ping columns.
        
    Returns:
        pd.DataFrame: The input DataFrame with a 'match_group_id' column added.
    """
    df_copy = df.copy()
    
    if len(df_copy) == 0:
        df_copy['match_group_id'] = []
        return df_copy
        
    # Scale ping globally to make it comparable to skill_score (which is in [0, 1])
    scaler = MinMaxScaler()
    df_copy['ping_norm'] = scaler.fit_transform(df_copy[['ping']])
    
    # Dictionary to collect assigned match group IDs: player_id -> match_group_id
    match_assignments = {}
    
    # Process each region separately
    for region, region_players in df_copy.groupby('region'):
        region_players = region_players.copy()
        n_players = len(region_players)
        
        # Calculate number of clusters (each target cluster size is around 4 players)
        n_clusters = max(1, n_players // 4)
        
        # Train KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        region_players['cluster_label'] = kmeans.fit_predict(region_players[['skill_score', 'ping_norm']])
        
        # Process each KMeans cluster and apply the ping fairness filter
        region_group_counter = 1
        for cluster_id, cluster_players in region_players.groupby('cluster_label'):
            # Sort players by original ping for greedy splitting
            cluster_sorted = cluster_players.sort_values(by='ping')
            
            # Greedily split cluster if pings differ by more than 80ms
            current_subgroup = []
            for idx, row in cluster_sorted.iterrows():
                if not current_subgroup:
                    current_subgroup.append(row)
                else:
                    min_ping = current_subgroup[0]['ping']
                    # Check the difference with the player having the lowest ping in the subgroup
                    if row['ping'] - min_ping <= 80:
                        current_subgroup.append(row)
                    else:
                        # Close current subgroup and start a new one
                        for subgroup_player in current_subgroup:
                            match_assignments[subgroup_player['player_id']] = f"{region}_G{region_group_counter}"
                        region_group_counter += 1
                        current_subgroup = [row]
                        
            # Assign remaining players in the last subgroup
            if current_subgroup:
                for subgroup_player in current_subgroup:
                    match_assignments[subgroup_player['player_id']] = f"{region}_G{region_group_counter}"
                region_group_counter += 1
                
    # Map the match group assignments back to the original DataFrame
    df_copy['match_group_id'] = df_copy['player_id'].map(match_assignments)
    
    # Clean up temporary column
    df_copy = df_copy.drop(columns=['ping_norm'], errors='ignore')
    
    return df_copy

if __name__ == '__main__':
    # 1. Generate Dataset
    df = generate_dataset(random_state=42)
    
    # 2. Run Anomaly Detection (Filter cheaters)
    clean_df, flagged_df = run_anomaly_detection(df)
    
    # 3. Compute Skill Scores and Tiers
    scored_df = compute_skill_scores(clean_df)
    
    # 4. Run Matchmaking
    matched_df = run_matchmaking(scored_df)
    
    # Configure pandas for nice output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    # Print Matchmaking Table: player_id | region | ping | skill_tier | match_group_id
    columns_to_print = ['player_id', 'region', 'ping', 'skill_tier', 'skill_score', 'match_group_id']
    
    # Sort by match_group_id first for easier reading
    sorted_matched_df = matched_df[columns_to_print].sort_values(by=['match_group_id', 'skill_score'], ascending=[True, False])
    
    print("=" * 120)
    print("GAME OPERATIONS - MATCHMAKING ENGINE")
    print("=" * 120)
    print(sorted_matched_df.to_string(index=False))
    print("=" * 120)
    
    # Print a summary of match groups
    print("\nMatch Group Summaries:")
    print("-" * 120)
    for group_id, group_df in sorted_matched_df.groupby('match_group_id'):
        min_ping = group_df['ping'].min()
        max_ping = group_df['ping'].max()
        pings_diff = max_ping - min_ping
        print(f"Group: {group_id:<12} | Players: {len(group_df):<2} | Skill Tiers: {list(group_df['skill_tier'].unique())} | Ping Range: {min_ping}ms - {max_ping}ms (diff: {pings_diff}ms)")
    print("-" * 120)
