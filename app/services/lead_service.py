from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CommercialFit, LeadProfile, TechnicalFit
from app.services.qualification_service import resolve_fit_state


async def get_or_create_lead_profile(session: AsyncSession, conversation_id: uuid.UUID) -> LeadProfile:
    profile = await session.get(LeadProfile, conversation_id)
    if profile is not None:
        return profile
    profile = LeadProfile(
        id=conversation_id,
        technical_fit=TechnicalFit.pending.value,
        commercial_fit=CommercialFit.pending.value,
    )
    session.add(profile)
    await session.flush()
    return profile


async def apply_fit_update(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    technical_fit: str | None,
    commercial_fit: str | None,
) -> LeadProfile:
    profile = await get_or_create_lead_profile(session, conversation_id)
    if technical_fit:
        profile.technical_fit = resolve_fit_state(profile.technical_fit, technical_fit)
    if commercial_fit:
        profile.commercial_fit = resolve_fit_state(profile.commercial_fit, commercial_fit)
    profile.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return profile
