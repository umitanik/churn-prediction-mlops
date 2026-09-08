from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '62fb86a94ab0'
down_revision: Union[str, Sequence[str], None] = 'a7d6dc158ef1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Add the scored features and the model version column."""
    with op.batch_alter_table('customer_churn_prediction_logs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('card_type', sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column('satisfaction_score', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('point_earned', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('model_version', sa.String(length=64), nullable=True))

def downgrade() -> None:
    """Drop the four columns. The values in them are not recoverable."""
    with op.batch_alter_table('customer_churn_prediction_logs', schema=None) as batch_op:
        batch_op.drop_column('model_version')
        batch_op.drop_column('point_earned')
        batch_op.drop_column('satisfaction_score')
        batch_op.drop_column('card_type')