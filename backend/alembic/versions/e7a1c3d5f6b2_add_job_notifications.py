"""add job notifications column

Revision ID: e7a1c3d5f6b2
Revises: d5f8a2b3c4e1
Create Date: 2026-10-17 00:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7a1c3d5f6b2'
down_revision: Union[str, Sequence[str], None] = 'd5f8a2b3c4e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('notifications', sa.JSON(), server_default='{}', nullable=False)
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.drop_column('notifications')
