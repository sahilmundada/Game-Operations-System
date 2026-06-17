import os
import numpy as np
import pandas as pd
from game_ops.api.database import SessionLocal, Player

def seed_database():
    # Resolve absolute path relative to this file's location to support running from any directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "outputs", "processed_training_data_v2.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run main_pipeline.py first to train models and generate the processed dataset.")
        return
        
    db = SessionLocal()
    try:
        # Recreate tables to apply schema updates
        from game_ops.api.database import Base, engine
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        
        # Read the processed training data
        df = pd.read_csv(csv_path, keep_default_na=False)
        
        # Replace empty strings or NaN with None
        df = df.replace({np.nan: None, "": None})
        
        players_to_insert = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            
            # Handle float conversions for columns that could be None/NaN
            float_cols = ['predicted_skill_score', 'kdr', 'score_per_minute', 'kill_rate', 
                          'efficiency', 'death_rate', 'survival_index', 'score_kill_ratio', 
                          'ping_adjusted_score', 'performance_index', 'iso_score', 'lof_score',
                          'confidence_score', 'history_flag_rate', 'consistency_score']
            for col in float_cols:
                if row_dict.get(col) is not None and row_dict[col] != "":
                    try:
                        row_dict[col] = float(row_dict[col])
                    except ValueError:
                        row_dict[col] = None
                else:
                    row_dict[col] = None
            
            # Handle integer conversions
            int_cols = ['ping', 'score', 'kills', 'deaths', 'match_duration_seconds', 
                        'ground_truth_label', 'region_encoded', 'device_encoded', 'iso_pred', 'lof_pred']
            for col in int_cols:
                if row_dict.get(col) is not None and row_dict[col] != "":
                    try:
                        row_dict[col] = int(row_dict[col])
                    except ValueError:
                        row_dict[col] = None
                else:
                    row_dict[col] = None
            
            # Handle boolean conversions
            bool_cols = ['rule_flagged', 'final_flagged', 'veteran_status']
            for col in bool_cols:
                if row_dict.get(col) is not None:
                    row_dict[col] = bool(row_dict[col])
            
            # Filter keys to match valid database columns
            valid_columns = set(Player.__table__.columns.keys())
            row_dict = {k: v for k, v in row_dict.items() if k in valid_columns}
            
            p = Player(**row_dict)
            players_to_insert.append(p)
            
        # Bulk save
        db.bulk_save_objects(players_to_insert)
        db.commit()
        print(f"Database successfully seeded with {len(players_to_insert)} players from processed_training_data.csv.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
    finally:
        db.close()
