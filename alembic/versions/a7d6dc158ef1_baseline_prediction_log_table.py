from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'a7d6dc158ef1'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('customer_churn_prediction_logs',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('customer_id', sa.Integer(), nullable=True),
    sa.Column('surname', sa.String(length=50), nullable=True),
    sa.Column('credit_score', sa.Integer(), nullable=False),
    sa.Column('geography', sa.String(length=50), nullable=False),
    sa.Column('gender', sa.String(length=10), nullable=False),
    sa.Column('age', sa.Integer(), nullable=False),
    sa.Column('tenure', sa.Integer(), nullable=False),
    sa.Column('balance', sa.Float(), nullable=False),
    sa.Column('num_of_products', sa.Integer(), nullable=False),
    sa.Column('has_cr_card', sa.Integer(), nullable=False),
    sa.Column('is_active_member', sa.Integer(), nullable=False),
    sa.Column('estimated_salary', sa.Float(), nullable=False),
    sa.Column('prediction_label', sa.String(length=20), nullable=False),
    sa.Column('churn_probability', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Drop the table. Destructive: every stored prediction is lost."""
    op.drop_table('customer_churn_prediction_logs')
