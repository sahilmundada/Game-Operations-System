from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/game_ops"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Player(Base):
    __tablename__ = "players"
    
    player_id = Column(String, primary_key=True, index=True)
    match_id = Column(String, index=True)
    region = Column(String)
    device = Column(String)
    ping = Column(Integer)
    score = Column(Integer)
    kills = Column(Integer)
    deaths = Column(Integer)
    match_duration_seconds = Column(Integer)
    ground_truth_label = Column(Integer, default=0)
    
    # Engineered Features
    kdr = Column(Float)
    score_per_minute = Column(Float)
    kill_rate = Column(Float)
    efficiency = Column(Float)
    death_rate = Column(Float)
    survival_index = Column(Float)
    score_kill_ratio = Column(Float)
    ping_adjusted_score = Column(Float)
    performance_index = Column(Float)
    region_encoded = Column(Integer)
    device_encoded = Column(Integer)
    
    # Anomaly Detection Output
    rule_flagged = Column(Boolean, default=False)
    iso_pred = Column(Integer)
    iso_score = Column(Float)
    lof_pred = Column(Integer)
    lof_score = Column(Float)
    final_flagged = Column(Boolean, default=False)
    flag_source = Column(String)
    flag_reason = Column(String)
    
    # Skill & Matchmaking
    predicted_skill_score = Column(Float)
    skill_tier = Column(String, default="Bronze")
    match_group_id = Column(String)
    match_group_reason = Column(String)
    submitted_at = Column(String)
    
    # v2 Upgrades: Confidence Score & History
    confidence_score = Column(Float)
    confidence_zone = Column(String)
    cheat_types_hit = Column(String)
    confirmed_cheats = Column(String)
    score_breakdown = Column(String)
    history_flag_rate = Column(Float)
    history_trend = Column(String)
    consistency_score = Column(Float)
    veteran_status = Column(Boolean)
    
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

def init_db():
    Base.metadata.create_all(bind=engine)
