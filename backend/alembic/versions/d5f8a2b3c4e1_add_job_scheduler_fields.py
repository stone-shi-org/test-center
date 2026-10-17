"""add job scheduler fields (missed_run_policy, next_run_at, last_scheduled_at)

Revision ID: d5f8a2b3c4e1
Revises: c4e9b1a2d3f0
Create Date: 2026-10-16 22:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5f8a2b3c4e1'
down_revision: Union[str, Sequence[str], None] = 'c4e9b1a2d3f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('missed_run_policy', sa.String(length=32), server_default='skip', nullable=False)
        )
        batch_op.add_column(
            sa.Column('next_run_at', sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column('last_scheduled_at', sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.create_index(batch_op.f('ix_jobs_next_run_at'), ['next_run_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_jobs_next_run_at'))
        batch_op.drop_column('last_scheduled_at')
        batch_op.drop_column('next_run_at')
        batch_op.drop_column('missed_run_policy')
