"""Test package.

This file matters: without it pytest imports conftest.py as the top-level
module ``conftest`` while ``from tests.conftest import ...`` in the test modules
creates a second, independent copy. Anything the fixtures compute once - the
temporary database path, for instance - would then exist twice with different
values.
"""
