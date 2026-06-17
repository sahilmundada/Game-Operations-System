from typing import List
from fastapi import APIRouter, Request
import pandas as pd
import numpy as np
import json
from game_ops.api.database import SessionLocal, Player
from game_ops.api.models import RunAnalysisResponse, MatchGroupEntry, StatsResponse

router = APIRouter()

@router.post("/run-analysis", response_model=RunAnalysisResponse)
async def run_analysis(request: Request):
    """
    Triggers the machine learning prediction pipeline on all stored players.
    Loads players from PostgreSQL, runs anomaly detection + skill scoring + matchmaking,
    and updates database records in bulk.
    """
    predictor = request.app.state.predictor
    
    db = SessionLocal()
    try:
        players = db.query(Player).all()
        if not players:
            return RunAnalysisResponse(
                analysis_complete=True,
                players_analyzed=0,
                newly_flagged=[],
                match_groups_formed=0
            )
            
        mappings = []
        newly_flagged = []
        match_groups = set()
        
        players_data = [
            {
                "player_id": p.player_id,
                "match_id": p.match_id,
                "region": p.region,
                "device": p.device,
                "ping": p.ping,
                "score": p.score,
                "kills": p.kills,
                "deaths": p.deaths,
                "match_duration_seconds": p.match_duration_seconds,
                "ground_truth_label": p.ground_truth_label
            }
            for p in players
        ]
        
        predictions = predictor.predict_players_batch(players_data)
        
        for idx, p in enumerate(players):
            res = predictions[idx]
            was_flagged = p.confidence_score >= 61 if p.confidence_score is not None else p.final_flagged
            now_flagged = res['confidence_score'] >= 61
            if now_flagged and not was_flagged:
                newly_flagged.append(p.player_id)
                
            if res['confidence_score'] <= 40 and res['match_group'] is not None:
                match_groups.add(res['match_group'])
                
            mappings.append({
                "player_id": p.player_id,
                "final_flagged": now_flagged,
                "is_flagged": now_flagged,
                "flag_source": res.get('flag_source'),
                "flag_reason": res.get('flag_reason'),
                "iso_score": res['iso_anomaly_score'],
                "lof_score": res['lof_anomaly_score'],
                "predicted_skill_score": res['skill_score'],
                "skill_score": res['skill_score'],
                "skill_tier": res['skill_tier'],
                "match_group_id": res['match_group'],
                "match_group_reason": res.get('match_group_reason'),
                
                # Upgrades v2
                "confidence_score": res['confidence_score'],
                "confidence_zone": res['confidence_zone'],
                "cheat_types_hit": ",".join(res['cheat_types_hit']),
                "confirmed_cheats": ",".join(res['confirmed_cheats']),
                "score_breakdown": json.dumps(res['score_breakdown']),
                "history_flag_rate": res['history_summary']['flag_rate'],
                "history_trend": res['history_summary']['trend'],
                "consistency_score": res['history_summary']['consistency_score'],
                "veteran_status": res['history_summary']['veteran_status']
            })
            
        # Bulk update in database
        db.bulk_update_mappings(Player, mappings)
        db.commit()
        
        num_match_groups = len(match_groups)
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
        
    return RunAnalysisResponse(
        analysis_complete=True,
        players_analyzed=len(players),
        newly_flagged=newly_flagged,
        match_groups_formed=num_match_groups
    )

@router.get("/matchmaking", response_model=List[MatchGroupEntry])
async def get_matchmaking_groups(request: Request):
    """
    Returns the current match groups formed after running analysis.
    """
    predictor = request.app.state.predictor
    registry = getattr(predictor, '_group_registry', {})
    
    group_entries = []
    for g_id, g in sorted(registry.items()):
        group_entries.append(
            MatchGroupEntry(
                group_id=g['group_id'],
                region=g['region'],
                player_count=g['player_count'],
                skill_tiers=g['skill_tiers_present'],
                avg_ping=g['avg_ping'],
                players=g['players'],
                avg_mmr=g.get('avg_mmr', 0.0),
                mmr_spread=g.get('mmr_spread', 0.0),
                ping_spread=g.get('ping_spread', 0.0),
                device_breakdown=g.get('device_breakdown', {}),
                device_flag=g.get('device_flag', 'balanced'),
                avg_confidence=g.get('avg_confidence', 0.0),
                fairness_score=g.get('fairness_score', 0.0),
                quality_label=g.get('quality_label', 'Balanced'),
                skill_tiers_present=g['skill_tiers_present']
            )
        )
        
    return group_entries

@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Calculates summary statistics across all players in the PostgreSQL database.
    """
    db = SessionLocal()
    try:
        players = db.query(Player).all()
        if not players:
            return StatsResponse(
                total_players=0,
                flagged_count=0,
                clean_players=0,
                region_breakdown={},
                device_breakdown={},
                avg_score=0.0,
                avg_kills=0.0,
                avg_ping=0.0,
                top_region_by_avg_score=None
            )
            
        total_players = len(players)
        # We define clean players as those in Clean or Watch zones (confidence <= 40)
        flagged_players = [p for p in players if p.confidence_score is not None and p.confidence_score > 40]
        clean_players = [p for p in players if p.confidence_score is not None and p.confidence_score <= 40]
        
        flagged_count = len(flagged_players)
        clean_count = len(clean_players)
        
        regions = [p.region for p in players]
        devices = [p.device for p in players]
        
        region_breakdown = {r: regions.count(r) for r in set(regions)}
        device_breakdown = {d: devices.count(d) for d in set(devices)}
        
        avg_score = float(np.mean([p.score for p in players]))
        avg_kills = float(np.mean([p.kills for p in players]))
        avg_ping = float(np.mean([p.ping for p in clean_players])) if clean_players else 0.0
        
        top_region = None
        if clean_players:
            region_scores = {}
            region_counts = {}
            for p in clean_players:
                reg = p.region
                region_scores[reg] = region_scores.get(reg, 0.0) + p.score
                region_counts[reg] = region_counts.get(reg, 0) + 1
                
            avg_region_scores = {r: region_scores[r] / region_counts[r] for r in region_scores}
            top_region = max(avg_region_scores, key=avg_region_scores.get)
    finally:
        db.close()
        
    return StatsResponse(
        total_players=total_players,
        flagged_count=flagged_count,
        clean_players=clean_count,
        region_breakdown=region_breakdown,
        device_breakdown=device_breakdown,
        avg_score=round(avg_score, 2),
        avg_kills=round(avg_kills, 2),
        avg_ping=round(avg_ping, 2),
        top_region_by_avg_score=top_region
    )
