import datetime
import json
from fastapi import APIRouter, status, Response, Request
from game_ops.api.database import SessionLocal, Player
from game_ops.api.models import ScoreSubmissionRequest, ScoreSubmissionResponse

router = APIRouter()

@router.post("/submit-score", response_model=ScoreSubmissionResponse)
async def submit_score(submission: ScoreSubmissionRequest, response: Response, request: Request):
    """
    Submits a player score record, runs ML predictions and rules,
    and stores the complete record in PostgreSQL.
    """
    predictor = request.app.state.predictor
    
    p_data = {
        "player_id": submission.player_id,
        "match_id": submission.match_id,
        "region": submission.region,
        "device": submission.device,
        "ping": submission.ping,
        "score": submission.score,
        "kills": submission.kills,
        "deaths": submission.deaths,
        "match_duration_seconds": submission.match_duration_seconds,
        "ground_truth_label": 0
    }
    
    # Run ML & Rule Anomaly/Skill engine
    res = predictor.predict_player(p_data)
    
    # Compute derived features for database storage
    kdr = submission.kills / (submission.deaths + 1)
    score_per_minute = submission.score / (submission.match_duration_seconds / 60.0)
    kill_rate = submission.kills / (submission.match_duration_seconds / 60.0)
    efficiency = submission.score / (submission.kills + 1)
    death_rate = submission.deaths / (submission.match_duration_seconds / 60.0)
    survival_index = 1.0 / (death_rate + 0.01)
    score_kill_ratio = submission.score / (submission.kills + 1)
    ping_adjusted_score = submission.score / (1.0 + submission.ping / 100.0)
    performance_index = score_per_minute / (kill_rate + 0.1)
    
    try:
        region_encoded = int(predictor._label_encoders['region'].transform([submission.region])[0])
    except Exception:
        region_encoded = 0
        
    try:
        device_encoded = int(predictor._label_encoders['device'].transform([submission.device])[0])
    except Exception:
        device_encoded = 0
        
    final_flagged = res['confidence_score'] >= 61
    rule_flagged = any(c in res['confirmed_cheats'] for c in res['confirmed_cheats'])
    
    iso_pred = res['iso_pred'] if 'iso_pred' in res else ( -1 if res['iso_anomaly_score'] < -0.05 else 1 )
    lof_pred = res['lof_pred'] if 'lof_pred' in res else ( -1 if res['lof_anomaly_score'] < -1.0 else 1 )
    
    # Update PlayerHistory in memory & persist
    new_match_result = {
        "match_id": submission.match_id,
        "score": float(submission.score),
        "kills": float(submission.kills),
        "deaths": float(submission.deaths),
        "score_per_minute": float(score_per_minute),
        "kdr": float(kdr),
        "was_flagged": final_flagged,
        "confidence_score": float(res['confidence_score']),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    predictor.update_history(submission.player_id, new_match_result)
    
    # Open database session
    db = SessionLocal()
    try:
        db_player = db.query(Player).filter(Player.player_id == submission.player_id).first()
        is_new = db_player is None
        
        player_dict = {
            "player_id": submission.player_id,
            "match_id": submission.match_id,
            "region": submission.region,
            "device": submission.device,
            "ping": submission.ping,
            "score": submission.score,
            "kills": submission.kills,
            "deaths": submission.deaths,
            "match_duration_seconds": submission.match_duration_seconds,
            "ground_truth_label": db_player.ground_truth_label if db_player else 0,
            
            # Engineered
            "kdr": kdr,
            "score_per_minute": score_per_minute,
            "kill_rate": kill_rate,
            "efficiency": efficiency,
            "death_rate": death_rate,
            "survival_index": survival_index,
            "score_kill_ratio": score_kill_ratio,
            "ping_adjusted_score": ping_adjusted_score,
            "performance_index": performance_index,
            "region_encoded": region_encoded,
            "device_encoded": device_encoded,
            
            # Flagging
            "rule_flagged": rule_flagged,
            "iso_pred": int(iso_pred),
            "iso_score": res['iso_anomaly_score'],
            "lof_pred": int(lof_pred),
            "lof_score": res['lof_anomaly_score'],
            "final_flagged": final_flagged,
            "flag_source": res.get('flag_source'),
            "flag_reason": res.get('flag_reason'),
            
            # Skill & Matchmaking
            "predicted_skill_score": res['skill_score'],
            "skill_tier": res['skill_tier'],
            "match_group_id": res['match_group'],
            "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            
            # v2 Upgrades
            "confidence_score": res['confidence_score'],
            "confidence_zone": res['confidence_zone'],
            "cheat_types_hit": ",".join(res['cheat_types_hit']),
            "confirmed_cheats": ",".join(res['confirmed_cheats']),
            "score_breakdown": json.dumps(res['score_breakdown']),
            "history_flag_rate": res['history_summary']['flag_rate'],
            "history_trend": res['history_summary']['trend'],
            "consistency_score": res['history_summary']['consistency_score'],
            "veteran_status": res['history_summary']['veteran_status']
        }
        
        if is_new:
            db_player = Player(**player_dict)
            db.add(db_player)
            response.status_code = status.HTTP_201_CREATED
            status_msg = "new_submission_created"
        else:
            for k, v in player_dict.items():
                setattr(db_player, k, v)
            response.status_code = status.HTTP_200_OK
            status_msg = "submission_updated"
            
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
        
    features = {
        "kdr": kdr,
        "score_per_minute": score_per_minute,
        "kill_rate": kill_rate,
        "efficiency": efficiency
    }
    
    return ScoreSubmissionResponse(
        status=status_msg,
        player_id=submission.player_id,
        flag_reason=res.get('flag_reason') if final_flagged else None,
        features=features,
        confidence_score=res['confidence_score'],
        confidence_zone=res['confidence_zone'],
        status_label=res['status_label'],
        action=res['action'],
        cheat_types_hit=res['cheat_types_hit'],
        confirmed_cheats=res['confirmed_cheats'],
        unconfirmed_hits=res['unconfirmed_hits'],
        score_breakdown=res['score_breakdown']
    )
