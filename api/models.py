from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class ScoreSubmissionRequest(BaseModel):
    player_id: str = Field(..., min_length=1, description="Unique identifier for the player")
    match_id: str = Field(..., min_length=1, description="Identifier for the match")
    region: str = Field(..., min_length=1, description="Region of the player")
    device: str = Field(..., min_length=1, description="Device used by the player")
    ping: int = Field(..., ge=0, description="Ping in milliseconds")
    score: int = Field(..., ge=0, description="Score achieved")
    kills: int = Field(..., ge=0, description="Number of kills")
    deaths: int = Field(..., ge=0, description="Number of deaths")
    match_duration_seconds: int = Field(..., gt=0, description="Match duration in seconds")

class ScoreSubmissionResponse(BaseModel):
    status: str
    player_id: str
    flag_reason: Optional[str] = None
    features: Dict[str, float]
    confidence_score: Optional[float] = 0.0
    confidence_zone: Optional[str] = "Clean"
    status_label: Optional[str] = "CLEAN"
    action: Optional[str] = "Rank normally, full access"
    cheat_types_hit: Optional[List[str]] = []
    confirmed_cheats: Optional[List[str]] = []
    unconfirmed_hits: Optional[List[str]] = []
    score_breakdown: Optional[Dict[str, float]] = {}

class LeaderboardEntry(BaseModel):
    player_id: str
    region: str
    score: int
    kills: int
    deaths: int
    skill_tier: Optional[str] = None
    match_group_id: Optional[str] = None
    global_rank: Optional[int] = None
    region_rank: Optional[int] = None
    confidence_score: Optional[float] = 0.0

class LeaderboardResponse(BaseModel):
    leaderboard: List[LeaderboardEntry]
    total_players: int
    flagged_excluded: int

class FlaggedPlayerEntry(BaseModel):
    player_id: str
    flag_reason: str
    score: int
    kills: int
    kdr: float
    score_per_minute: float
    submitted_at: Optional[str] = None

class FlaggedPlayersResponse(BaseModel):
    flagged_players: List[FlaggedPlayerEntry]

class RunAnalysisResponse(BaseModel):
    analysis_complete: bool
    players_analyzed: int
    newly_flagged: List[str]
    match_groups_formed: int

class MatchGroupEntry(BaseModel):
    group_id: str
    region: str
    player_count: int
    skill_tiers: List[str]
    avg_ping: float
    players: List[str]

class MatchmakingResponse(BaseModel):
    matchgroups: List[MatchGroupEntry]

class StatsResponse(BaseModel):
    total_players: int
    flagged_count: int
    clean_players: int
    region_breakdown: Dict[str, int]
    device_breakdown: Dict[str, int]
    avg_score: float
    avg_kills: float
    avg_ping: float
    top_region_by_avg_score: Optional[str] = None

class PlayerRecord(BaseModel):
    player_id: str
    match_id: str
    region: str
    device: str
    ping: int
    score: int
    kills: int
    deaths: int
    match_duration_seconds: int
    ground_truth_label: int = 0
    kdr: float
    score_per_minute: float
    kill_rate: float
    efficiency: float
    death_rate: float
    survival_index: float
    score_kill_ratio: float
    ping_adjusted_score: float
    performance_index: float
    region_encoded: int
    device_encoded: int
    rule_flagged: bool = False
    iso_pred: int = 1
    iso_score: float = 0.0
    lof_pred: int = 1
    lof_score: float = 0.0
    final_flagged: bool = False
    is_flagged: bool = False  # Aliased to final_flagged for compatibility
    flag_source: Optional[str] = None
    flag_reason: Optional[str] = ""
    isolation_score: float = 0.0  # Aliased to iso_score for compatibility
    predicted_skill_score: Optional[float] = 0.0
    skill_score: Optional[float] = 0.0  # Aliased to predicted_skill_score for compatibility
    skill_tier: Optional[str] = "Bronze"
    match_group_id: Optional[str] = None
    submitted_at: Optional[str] = None
    
    # Upgrades v2
    confidence_score: float = 0.0
    confidence_zone: str = "Clean"
    status_label: str = "CLEAN"
    action: str = "Rank normally, full access"
    cheat_types_hit: Optional[List[str]] = []
    confirmed_cheats: Optional[List[str]] = []
    unconfirmed_hits: Optional[List[str]] = []
    score_breakdown: Optional[Dict[str, float]] = {}
    history_flag_rate: float = 0.0
    history_trend: str = "stable"
    consistency_score: float = 1.0
    veteran_status: bool = False
