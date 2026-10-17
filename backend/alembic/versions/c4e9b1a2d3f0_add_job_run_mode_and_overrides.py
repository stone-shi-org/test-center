"""add job run_mode, overrides, and description

Revision ID: c4e9b1a2d3f0
Revises: 789af5e4b646
Create Date: 2026-10-16 22:12:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4e9b1a2d3f0'
down_revision: Union[str, Sequence[str], None] = '789af5e4b646'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('description', sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column('run_mode', sa.String(length=32), server_default='sequential', nullable=False)
        )
        batch_op.add_column(
            sa.Column('overrides', sa.JSON(), server_default='{}', nullable=False)
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.drop_column('overrides')
        batch_op.drop_column('run_mode')
        batch_op.drop_column('description')
