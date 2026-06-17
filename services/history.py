import numpy as np

class PlayerHistory:
    def __init__(self, player_id: str, match_history: list = None):
        self.player_id = player_id
        self.match_history = match_history if match_history is not None else []
        self.total_matches = len(self.match_history)
        self.avg_score = 0.0
        self.avg_kdr = 0.0
        self.avg_score_per_min = 0.0
        self.flag_count = 0
        self.flag_rate = 0.0
        self.trend = "stable"
        self.veteran_status = False
        self.consistency_score = 1.0
        
        if self.match_history:
            self.recalculate()
            
    def recalculate(self):
        self.total_matches = len(self.match_history)
        if self.total_matches == 0:
            self.avg_score = 0.0
            self.avg_kdr = 0.0
            self.avg_score_per_min = 0.0
            self.flag_count = 0
            self.flag_rate = 0.0
            self.trend = "stable"
            self.veteran_status = False
            self.consistency_score = 1.0
            return
            
        scores = [m['score'] for m in self.match_history]
        kdrs = [m['kdr'] for m in self.match_history]
        spms = [m['score_per_minute'] for m in self.match_history]
        flaggeds = [m.get('was_flagged', False) for m in self.match_history]
        
        self.avg_score = float(np.mean(scores))
        self.avg_kdr = float(np.mean(kdrs))
        self.avg_score_per_min = float(np.mean(spms))
        
        self.flag_count = int(sum(flaggeds))
        self.flag_rate = float(self.flag_count / self.total_matches)
        
        # Trend detection: Compare last 2 matches vs first 2 matches
        if self.total_matches >= 4:
            first_2 = spms[:2]
            last_2 = spms[-2:]
            avg_first_2 = np.mean(first_2)
            avg_last_2 = np.mean(last_2)
            
            if avg_first_2 == 0:
                self.trend = "stable"
            else:
                ratio = avg_last_2 / avg_first_2
                if ratio > 1.50:
                    self.trend = "suspicious_spike"
                elif ratio > 1.10:
                    self.trend = "improving"
                elif ratio < 0.90:
                    self.trend = "declining"
                else:
                    self.trend = "stable"
        else:
            self.trend = "stable"
            
        # Veteran status: True if total_matches > 10 AND flag_rate < 0.05
        self.veteran_status = bool(self.total_matches > 10 and self.flag_rate < 0.05)
        
        # Consistency score
        if self.total_matches >= 2:
            mean_spm = np.mean(spms)
            std_spm = np.std(spms)
            if mean_spm == 0:
                self.consistency_score = 1.0
            else:
                cv = std_spm / mean_spm
                self.consistency_score = float(1 - min(1, cv))
        else:
            self.consistency_score = 1.0

    def add_match(self, match_id: str, score: float, kills: float, deaths: float,
                  score_per_minute: float, kdr: float, was_flagged: bool,
                  confidence_score: float, timestamp: str):
        self.match_history.append({
            "match_id": match_id,
            "score": score,
            "kills": kills,
            "deaths": deaths,
            "score_per_minute": score_per_minute,
            "kdr": kdr,
            "was_flagged": was_flagged,
            "confidence_score": confidence_score,
            "timestamp": timestamp
        })
        self.recalculate()
