import numpy as np
import pandas as pd

FEATURE_COLS = ['score_per_minute', 'kill_rate', 'kdr', 'efficiency']

def generate_dataset(random_state=42):
    """
    Generates a synthetic dataset of 50 players, including exactly 5 suspicious players
    with impossible stats, and performs feature engineering on the raw data.
    
    Args:
        random_state (int): Seed for random number generation to ensure reproducibility.
        
    Returns:
        pd.DataFrame: The synthetic player dataset containing raw columns, ground truth labels,
                     and engineered features.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_state)
    
    # 1. SYNTHETIC DATASET GENERATION
    # Generate 50 unique player IDs (P001 - P050)
    player_ids = [f"P{i:03d}" for i in range(1, 51)]
    
    # Generate a pool of 10 matches and assign randomly to the 50 players
    match_pool = [f"M{i:03d}" for i in range(1, 11)]
    match_ids = list(np.random.choice(match_pool, size=50))
    
    # Randomly assign regions and devices
    regions = list(np.random.choice(['India', 'SEA', 'Europe', 'NA'], size=50))
    devices = list(np.random.choice(['Android', 'iOS', 'PC', 'Console'], size=50))
    
    # Generate realistic stats for normal players
    pings = list(np.random.randint(20, 201, size=50))  # Ping 20-200ms
    scores = list(np.random.randint(500, 5001, size=50))  # Score 500-5000
    kills = list(np.random.randint(1, 26, size=50))  # Kills 1-25
    deaths = list(np.random.randint(1, 16, size=50))  # Deaths 1-15
    match_durations = list(np.random.randint(300, 601, size=50))  # Duration 300-600 seconds
    
    # Initialize ground truth labels (0 = legit, 1 = cheater)
    ground_truth_labels = [0] * 50
    
    # Inject exactly 5 suspicious players (P046 - P050) at indices 45 to 49
    # Suspicious stats requirements:
    # - Score 80000+ in under 90 seconds
    # - 200+ kills with 0 deaths
    # - Kill rate > 15 kills/minute
    cheater_scores = np.random.randint(80000, 100000, size=5)
    cheater_durations = np.random.randint(45, 90, size=5)  # Under 90 seconds
    cheater_kills = np.random.randint(200, 251, size=5)  # 200+ kills
    
    for i in range(5):
        idx = 45 + i
        scores[idx] = int(cheater_scores[i])
        match_durations[idx] = int(cheater_durations[i])
        kills[idx] = int(cheater_kills[i])
        deaths[idx] = 0  # 0 deaths
        ground_truth_labels[idx] = 1  # Mark as cheater
        
    # Create the DataFrame
    df = pd.DataFrame({
        'player_id': player_ids,
        'match_id': match_ids,
        'region': regions,
        'device': devices,
        'ping': pings,
        'score': scores,
        'kills': kills,
        'deaths': deaths,
        'match_duration_seconds': match_durations,
        'ground_truth_label': ground_truth_labels
    })
    
    # 2. FEATURE ENGINEERING
    # kdr = kills / (deaths + 1)
    df['kdr'] = df['kills'] / (df['deaths'] + 1)
    
    # score_per_minute = score / (match_duration_seconds / 60)
    df['score_per_minute'] = df['score'] / (df['match_duration_seconds'] / 60)
    
    # kill_rate = kills / (match_duration_seconds / 60)
    df['kill_rate'] = df['kills'] / (df['match_duration_seconds'] / 60)
    
    # efficiency = score / (kills + 1)
    df['efficiency'] = df['score'] / (df['kills'] + 1)
    
    # death_rate = deaths / (match_duration_seconds / 60)
    df['death_rate'] = df['deaths'] / (df['match_duration_seconds'] / 60)
    
    return df

if __name__ == '__main__':
    # Generate the dataset
    df = generate_dataset(random_state=42)
    
    # Configure pandas formatting to display all columns and rows clearly in the terminal
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    # Print the DataFrame clearly with a section divider
    print("=" * 120)
    print("GAME OPERATIONS INTELLIGENCE SYSTEM - DATA FOUNDATION ENGINE")
    print("=" * 120)
    print("\nGenerated Synthetic Dataset (50 Players including 5 Suspicious Players P046-P050):")
    print("-" * 120)
    print(df.to_string(index=False))
    print("-" * 120)
    print("\nFeature Columns Exported:")
    print(FEATURE_COLS)
    print("=" * 120)
