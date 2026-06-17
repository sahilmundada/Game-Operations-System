import sys
import os
# Ensure parent directory is in the path to run script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from game_ops.services.data_engine import generate_dataset
from game_ops.services.anomaly import run_anomaly_detection, get_suspicious_players_report
from game_ops.services.skill_engine import compute_skill_scores
from game_ops.services.matchmaking import run_matchmaking

def generate_report(df, clean_df, flagged_df, suspicious_df, output_path="game_ops/outputs/game_ops_report.png"):
    """
    Generates 4 diagnostic plots in a 2x2 grid and saves the result as a PNG file.
    
    Plots:
    - Plot 1 (top-left): Bar chart of top 15 players by score, colored by skill tier.
    - Plot 2 (top-right): Scatter plot of skill_score vs ping, with match group ellipses and flagged marks.
    - Plot 3 (bottom-left): Box plot of score distribution by region, overlaid with a strip plot.
    - Plot 4 (bottom-right): Horizontal bar chart of anomaly scores of flagged players.
    
    Args:
        df (pd.DataFrame): The raw, full player dataset.
        clean_df (pd.DataFrame): The clean dataset with skill_score, skill_tier, and match_group_id.
        flagged_df (pd.DataFrame): The flagged player dataset.
        suspicious_df (pd.DataFrame): The suspicious player metadata report.
        output_path (str): The filename to save the report to.
    """
    # Set style for modern aesthetics
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GAME OPERATIONS INTELLIGENCE SYSTEM — REGION & SKILL ANALYTICS", fontsize=16, fontweight='bold', color='#1a1a1a')
    
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # ---------------------------------------------------------
    # Plot 1: Top 15 Players by Score
    # ---------------------------------------------------------
    top_15 = clean_df.sort_values(by='score', ascending=False).head(15).copy()
    tier_colors = {
        'Bronze': 'gray',
        'Silver': 'steelblue',
        'Gold': 'gold',
        'Platinum': 'mediumpurple',
        'Pro': 'crimson'
    }
    colors = [tier_colors[tier] for tier in top_15['skill_tier']]
    
    sns.barplot(x='player_id', y='score', data=top_15, palette=colors, ax=ax1, hue='player_id', legend=False)
    ax1.set_title("Top 15 Clean Players by Score (Colored by Tier)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Player ID", fontsize=10)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    
    # Create custom legend for tiers
    patches = [mpatches.Patch(color=color, label=tier) for tier, color in tier_colors.items()]
    ax1.legend(handles=patches, title="Skill Tier", loc='lower right')
    
    # ---------------------------------------------------------
    # Plot 2: Skill Score vs Ping
    # ---------------------------------------------------------
    # Recalculate skill score for all players using MinMaxScaler fitted on clean_df
    scaler = MinMaxScaler()
    scaler.fit(clean_df[['kdr', 'score_per_minute', 'efficiency']])
    
    # Process full dataset to get aligned skill scores
    df_aligned = df.copy()
    scaled_features = scaler.transform(df_aligned[['kdr', 'score_per_minute', 'efficiency']])
    df_aligned['skill_score'] = 0.35 * scaled_features[:, 0] + 0.35 * scaled_features[:, 1] + 0.30 * scaled_features[:, 2]
    
    # Segment back to clean and flagged for plotting
    plot_clean = df_aligned[df_aligned['player_id'].isin(clean_df['player_id'])].copy()
    plot_clean = plot_clean.merge(clean_df[['player_id', 'match_group_id', 'skill_tier']], on='player_id')
    plot_flagged = df_aligned[df_aligned['player_id'].isin(suspicious_df['player_id'])].copy()
    
    # Plot clean players
    sns.scatterplot(
        x='skill_score', y='ping', hue='region', style='region',
        data=plot_clean, s=100, alpha=0.8, ax=ax2, palette='Set1'
    )
    
    # Plot flagged players
    ax2.scatter(
        plot_flagged['skill_score'], plot_flagged['ping'],
        color='red', marker='X', s=150, linewidths=1.5, edgecolors='black', label='Flagged Player'
    )
    
    # Draw ellipses around each match group
    for group_id, group_df in plot_clean.groupby('match_group_id'):
        if len(group_df) >= 2:
            x_vals = group_df['skill_score'].values
            y_vals = group_df['ping'].values
            x_mean, y_mean = np.mean(x_vals), np.mean(y_vals)
            
            # Compute width and height with small expansion padding
            width = max(0.04, np.max(x_vals) - np.min(x_vals)) * 1.3
            height = max(15.0, np.max(y_vals) - np.min(y_vals)) * 1.3
            
            ellipse = Ellipse(
                (x_mean, y_mean), width=width, height=height,
                facecolor='none', edgecolor='black', linestyle='--', linewidth=0.8, alpha=0.3
            )
            ax2.add_patch(ellipse)
            ax2.text(x_mean, y_mean, group_id, fontsize=7, color='darkgreen', alpha=0.7, ha='center', va='center')
            
    ax2.set_title("Skill Score vs Ping (with Match Lobbies & Flags)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Skill Score (Normalized)", fontsize=10)
    ax2.set_ylabel("Ping (ms)", fontsize=10)
    # Refresh legend to include flagged marker
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(loc='upper right')
    
    # ---------------------------------------------------------
    # Plot 3: Box Plot of Score Distribution by Region
    # ---------------------------------------------------------
    sns.boxplot(x='region', y='score', data=clean_df, ax=ax3, palette='Pastel1', hue='region', legend=False)
    sns.stripplot(x='region', y='score', data=clean_df, color='black', alpha=0.5, jitter=0.2, size=5, ax=ax3)
    
    # Mark flagged players
    flagged_scores = df[df['player_id'].isin(suspicious_df['player_id'])].copy()
    sns.stripplot(
        x='region', y='score', data=flagged_scores,
        color='red', marker='X', size=12, linewidth=1, ax=ax3, jitter=0.1
    )
    
    ax3.set_title("Score Distribution by Region (Clean vs. Flagged)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Region", fontsize=10)
    ax3.set_ylabel("Score", fontsize=10)
    
    # Custom legend for box plot
    flagged_line = mlines.Line2D([], [], color='red', marker='X', linestyle='None', markersize=10, label='Flagged Player')
    clean_line = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=6, label='Clean Player')
    ax3.legend(handles=[clean_line, flagged_line], loc='upper right')
    
    # ---------------------------------------------------------
    # Plot 4: Anomaly Scores of Flagged Players
    # ---------------------------------------------------------
    sorted_suspicious = suspicious_df.sort_values(by='isolation_score', ascending=True).copy()
    
    sns.barplot(
        x='isolation_score', y='player_id', data=sorted_suspicious,
        palette='dark:crimson', ax=ax4, hue='player_id', legend=False
    )
    
    # Annotate bars with reasons
    for idx, row in enumerate(sorted_suspicious.itertuples()):
        score = row.isolation_score
        reason = row.flag_reason
        if score < 0:
            ax4.text(score - 0.005, idx, f" {reason}", va='center', ha='right', fontsize=8, color='black', fontweight='semibold')
        else:
            ax4.text(score + 0.005, idx, f" {reason}", va='center', ha='left', fontsize=8, color='black', fontweight='semibold')
            
    ax4.set_title("Isolation Forest Score of Flagged Players", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Isolation Score (Decision Function Value)", fontsize=10)
    ax4.set_ylabel("Player ID", fontsize=10)
    
    # Set x limits to prevent annotation cutoff
    x_min = sorted_suspicious['isolation_score'].min()
    x_max = sorted_suspicious['isolation_score'].max()
    ax4.set_xlim(x_min - 0.08, x_max + 0.08)
    
    # Resolve relative paths to the actual package directory to handle running from different CWDs
    if output_path and ("game_ops/outputs/" in output_path or output_path == "game_ops_report.png"):
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(package_dir, "outputs", os.path.basename(output_path))

    # Ensure target output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Report figure saved successfully to: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    # Dry run generator
    df = generate_dataset(random_state=42)
    clean_df, flagged_df = run_anomaly_detection(df)
    scored_df = compute_skill_scores(clean_df)
    matched_df = run_matchmaking(scored_df)
    suspicious_df = get_suspicious_players_report(df)
    
    generate_report(df, matched_df, flagged_df, suspicious_df, "game_ops/outputs/game_ops_report.png")
