import os

import pytest

_REQUIRES_OPENAI_MARKER: str = "requires_openai"


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``requires_openai`` marker.

    Args:
        config: The active pytest configuration object.
    """
    config.addinivalue_line(
        "markers",
        f"{_REQUIRES_OPENAI_MARKER}: test needs a live OPENAI_API_KEY to run.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip live OpenAI tests when no API key is available.

    Pull requests opened from forks do not receive repository secrets, so
    ``OPENAI_API_KEY`` is empty in those CI runs. Tests that call the real
    OpenAI API are skipped instead of failing with an empty ``Bearer`` header.

    Args:
        config: The active pytest configuration object.
        items: The collected test items to inspect and mark for skipping.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return

    skip_marker: pytest.MarkDecorator = pytest.mark.skip(
        reason="OPENAI_API_KEY not set; skipping live OpenAI integration test."
    )
    for item in items:
        if item.get_closest_marker(_REQUIRES_OPENAI_MARKER) is not None:
            item.add_marker(skip_marker)
