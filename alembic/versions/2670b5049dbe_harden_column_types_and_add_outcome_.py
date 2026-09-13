"""harden column types and add outcome feedback

Three things, all in service of making the audit log usable as a training
source:

* created_at becomes timestamptz. The values were always written in UTC, but
  the column type did not say so; the USING clause pins that interpretation
  during the conversion instead of trusting the session time zone.
* Money and probability move from double precision to exact numerics.
* actual_label and labeled_at are added, nullable, to hold the customer's real
  outcome once it is known (POST /feedback/{log_id}). They stay NULL for rows
  written before this revision.

Written in batch mode so it also applies to SQLite in the test suite; on
PostgreSQL batch mode emits ordinary ALTER statements.

Revision ID: 2670b5049dbe
Revises: 62fb86a94ab0
Create Date: 2026-09-13 21:56:15.347199

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '2670b5049dbe'
down_revision: Union[str, Sequence[str], None] = '62fb86a94ab0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Widen the audit log for ground truth and tighten its types."""
    with op.batch_alter_table('customer_churn_prediction_logs') as batch_op:
        batch_op.add_column(sa.Column('actual_label', sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column('labeled_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.alter_column(
            'created_at',
            existing_type=sa.DateTime(),
            type_=sa.DateTime(timezone=True),
            existing_nullable=False,
            postgresql_using="created_at AT TIME ZONE 'UTC'",
        )
        batch_op.alter_column(
            'balance',
            existing_type=sa.Float(),
            type_=sa.Numeric(precision=12, scale=2, asdecimal=False),
            existing_nullable=False,
        )
        batch_op.alter_column(
            'estimated_salary',
            existing_type=sa.Float(),
            type_=sa.Numeric(precision=12, scale=2, asdecimal=False),
            existing_nullable=False,
        )
        batch_op.alter_column(
            'churn_probability',
            existing_type=sa.Float(),
            type_=sa.Numeric(precision=7, scale=6, asdecimal=False),
            existing_nullable=False,
        )


def downgrade() -> None:
    """Revert the types and drop the feedback columns. Labels in them are lost."""
    with op.batch_alter_table('customer_churn_prediction_logs') as batch_op:
        batch_op.alter_column(
            'churn_probability',
            existing_type=sa.Numeric(precision=7, scale=6, asdecimal=False),
            type_=sa.Float(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            'estimated_salary',
            existing_type=sa.Numeric(precision=12, scale=2, asdecimal=False),
            type_=sa.Float(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            'balance',
            existing_type=sa.Numeric(precision=12, scale=2, asdecimal=False),
            type_=sa.Float(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            'created_at',
            existing_type=sa.DateTime(timezone=True),
            type_=sa.DateTime(),
            existing_nullable=False,
        )
        batch_op.drop_column('labeled_at')
        batch_op.drop_column('actual_label')
