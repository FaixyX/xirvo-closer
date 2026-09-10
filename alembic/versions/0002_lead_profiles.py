"""Create lead_profiles and backfill existing conversation IDs.

Revision ID: 0002_lead_profiles
Revises: 0001_initial_rag
Create Date: 2026-08-20
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_lead_profiles"
down_revision: Union[str, Sequence[str], None] = "0001_initial_rag"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "lead_profiles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "technical_fit",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "commercial_fit",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "technical_fit IN ('pending', 'qualified', 'unqualified')",
            name="ck_lead_profiles_technical_fit",
        ),
        sa.CheckConstraint(
            "commercial_fit IN ('pending', 'qualified', 'negotiation_required', 'unqualified')",
            name="ck_lead_profiles_commercial_fit",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        """
        INSERT INTO lead_profiles (id, technical_fit, commercial_fit, created_at, updated_at)
        SELECT DISTINCT conversation_id, 'pending', 'pending', NOW(), NOW()
        FROM messages
        WHERE conversation_id NOT IN (SELECT id FROM lead_profiles)
        """
    )


def downgrade() -> None:
    op.drop_table("lead_profiles")
