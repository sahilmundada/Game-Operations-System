import json
from fastapi import APIRouter, HTTPException, status
from game_ops.api.database import SessionLocal, Player
from game_ops.api.models import FlaggedPlayersResponse, FlaggedPlayerEntry, PlayerRecord

router = APIRouter()

@router.get("/flagged-players", response_model=FlaggedPlayersResponse)
async def get_flagged_players():
    """
    Returns all players with confidence_score > 40 in the system sorted by score DESC.
    """
    db = SessionLocal()
    try:
        # Query players with confidence_score > 40
        flagged_players = db.query(Player).filter(Player.confidence_score > 40).all()
        # Sort by score DESC
        flagged_sorted = sorted(flagged_players, key=lambda x: -(x.score or 0))
        
        entries = [
            FlaggedPlayerEntry(
                player_id=p.player_id,
                flag_reason=p.flag_reason or "ml_anomaly",
                score=p.score,
                kills=p.kills,
                kdr=p.kdr,
                score_per_minute=p.score_per_minute,
                submitted_at=p.submitted_at
            )
            for p in flagged_sorted
        ]
    finally:
        db.close()
        
    return FlaggedPlayersResponse(flagged_players=entries)

@router.get("/players/{player_id}", response_model=PlayerRecord)
async def get_player_record(player_id: str):
    """
    Retrieves the complete profile of a single player.
    Returns HTTP 404 if the player does not exist.
    """
    db = SessionLocal()
    try:
        p = db.query(Player).filter(Player.player_id == player_id).first()
        if not p:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Player with ID '{player_id}' was not found in the system"
            )
        # Convert to dictionary and map compatibility aliases
        p_dict = p.to_dict()
        p_dict['is_flagged'] = p_dict['final_flagged']
        p_dict['isolation_score'] = p_dict['iso_score']
        p_dict['skill_score'] = p_dict['predicted_skill_score']
        
        # Deserialize JSON/list strings from the database for Pydantic validation
        p_dict['cheat_types_hit'] = p_dict['cheat_types_hit'].split(",") if p_dict.get('cheat_types_hit') else []
        p_dict['confirmed_cheats'] = p_dict['confirmed_cheats'].split(",") if p_dict.get('confirmed_cheats') else []
        p_dict['unconfirmed_hits'] = p_dict['unconfirmed_hits'].split(",") if p_dict.get('unconfirmed_hits') else []
        p_dict['score_breakdown'] = json.loads(p_dict['score_breakdown']) if p_dict.get('score_breakdown') else {}
    finally:
        db.close()
        
    return PlayerRecord(**p_dict)
