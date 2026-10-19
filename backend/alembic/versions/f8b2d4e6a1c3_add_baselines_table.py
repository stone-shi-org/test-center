"""add baselines table

Revision ID: f8b2d4e6a1c3
Revises: e7a1c3d5f6b2
Create Date: 2026-10-18 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f8b2d4e6a1c3'
down_revision: Union[str, Sequence[str], None] = 'e7a1c3d5f6b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'baselines',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('history', sa.JSON(), server_default='[]', nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_baselines_run_id', 'baselines', ['run_id'], unique=False)
    op.create_index('uq_baselines_name_lower', 'baselines', [sa.text('lower(name)')], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('uq_baselines_name_lower', table_name='baselines')
    op.drop_index('ix_baselines_run_id', table_name='baselines')
    op.drop_table('baselines')
