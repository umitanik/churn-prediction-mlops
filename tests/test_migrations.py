import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect

from src.database.db import Base, engine
from tests.conftest import PROJECT_ROOT

SCORED_FEATURE_COLUMNS = {"card_type", "satisfaction_score", "point_earned"}


@pytest.fixture(scope="module")
def alembic_config():
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    return config


def test_migrations_produce_the_schema_the_models_describe(alembic_config):
    """No pending autogenerate diff.

    Editing a model without writing a migration leaves the two silently out of
    step: the application expects a column the database does not have, and the
    failure only appears on the first insert against a real deployment.
    """
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        differences = compare_metadata(context, Base.metadata)

    assert differences == [], f"Models and migrations disagree: {differences}"


def test_migration_chain_is_linear(alembic_config):
    """A branched history cannot be upgraded without an explicit merge."""
    script = ScriptDirectory.from_config(alembic_config)

    assert len(script.get_heads()) == 1


def test_every_migration_is_reversible(alembic_config):
    """Each revision must define a real downgrade.

    A migration that cannot be undone turns a bad deploy into a restore-from-
    backup exercise.
    """
    script = ScriptDirectory.from_config(alembic_config)

    for revision in script.walk_revisions():
        source = (PROJECT_ROOT / "alembic" / "versions").glob(f"{revision.revision}_*.py")
        body = next(source).read_text()
        downgrade = body.split("def downgrade()", 1)[1]

        assert "op." in downgrade, f"{revision.revision} has an empty downgrade()"


def test_scored_features_are_persisted(alembic_config):
    """The columns that make the log usable as a training source.

    Without these a stored row cannot reproduce the feature vector the model
    was given, so the audit trail cannot be turned back into a dataset.
    """
    columns = {c["name"] for c in inspect(engine).get_columns("customer_churn_prediction_logs")}

    assert SCORED_FEATURE_COLUMNS <= columns
    assert "model_version" in columns


def test_downgrade_and_upgrade_round_trip(alembic_config):
    """Applying the chain backwards and forwards again must be a no-op."""
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")

    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        differences = compare_metadata(context, Base.metadata)

    assert differences == []
