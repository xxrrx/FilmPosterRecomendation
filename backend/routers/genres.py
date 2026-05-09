"""
routers/genres.py — /api/genres/rules
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database import get_db, Rule
from schemas import RulesResponse, RuleItem

router = APIRouter(prefix="/api/genres", tags=["genres"])


@router.get("/rules", response_model=RulesResponse)
def get_rules(
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    min_lift:       float = Query(1.0, ge=0.0),
    limit:          int   = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Danh sach association rules the loai phim."""
    query = (
        db.query(Rule)
        .filter(Rule.confidence >= min_confidence, Rule.lift >= min_lift)
        .order_by(Rule.confidence.desc(), Rule.lift.desc())
        .limit(limit)
    )
    rules = query.all()

    return RulesResponse(
        total = len(rules),
        rules = [
            RuleItem(
                id         = r.id,
                antecedent = r.antecedent,
                consequent = r.consequent,
                support    = round(r.support, 4),
                confidence = round(r.confidence, 4),
                lift       = round(r.lift, 4),
            )
            for r in rules
        ],
    )
