from fastapi import APIRouter, Query
from game_ops.api.database import SessionLocal, Player
from game_ops.api.models import LeaderboardResponse, LeaderboardEntry

router = APIRouter()

@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    region: str = Query("all", description="Filter by region (all, India, SEA, Europe, NA, LatAm, Middle_East)"),
    limit: int = Query(20, ge=1, description="Limit the number of ranked players returned")
):
    """
    Retrieves the leaderboard of clean and watch (non-flagged/restricted) players.
    Includes only players with confidence_score <= 40.
    Sorts Clean players (0-20) first, then Watch players (21-40) at bottom.
    Within each zone, sorted by score DESC -> deaths ASC -> kills DESC.
    """
    db = SessionLocal()
    try:
        # Get count stats
        total_players = db.query(Player).count()
        flagged_excluded = db.query(Player).filter(Player.confidence_score > 40).count()
        
        # Get all clean/watch players (confidence_score <= 40)
        clean_players = db.query(Player).filter(
            Player.confidence_score <= 40
        ).all()
        
        # Sort globally: Clean first (0), Watch second (1), then score DESC, deaths ASC, kills DESC
        clean_players_sorted = sorted(
            clean_players,
            key=lambda x: (
                0 if (x.confidence_score is None or x.confidence_score <= 20) else 1,
                -x.score if x.score is not None else 0,
                x.deaths if x.deaths is not None else 0,
                -x.kills if x.kills is not None else 0
            )
        )
        
        # Assign global and regional ranks
        region_counters = {}
        leaderboard_data = []
        for idx, player in enumerate(clean_players_sorted):
            player_dict = player.to_dict()
            player_dict["global_rank"] = idx + 1
            
            r = player.region
            region_counters[r] = region_counters.get(r, 0) + 1
            player_dict["region_rank"] = region_counters[r]
            
            leaderboard_data.append(player_dict)
            
        # Apply region filter if necessary
        if region.lower() != "all":
            filtered_data = [p for p in leaderboard_data if p["region"].lower() == region.lower()][:limit]
        else:
            filtered_data = leaderboard_data[:limit]
            
        # Construct response entries
        leaderboard_entries = [
            LeaderboardEntry(
                player_id=p["player_id"],
                region=p["region"],
                score=p["score"],
                kills=p["kills"],
                deaths=p["deaths"],
                skill_tier=p.get("skill_tier", "Bronze"),
                match_group_id=p.get("match_group_id"),
                global_rank=p.get("global_rank"),
                region_rank=p.get("region_rank"),
                confidence_score=p.get("confidence_score", 0.0)
            )
            for p in filtered_data
        ]
    finally:
        db.close()
        
    return LeaderboardResponse(
        leaderboard=leaderboard_entries,
        total_players=total_players,
        flagged_excluded=flagged_excluded
    )
