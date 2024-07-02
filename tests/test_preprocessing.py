"""Test module of preprocessing."""

from sys import path as sys_path

sys_path.append("../src")
from elasticity.data.preprocessing import round_price_effect


def test_round_price_effect() -> None:
    """Test round_price_effect."""
    assert round_price_effect(9.54) == 9.5
    assert round_price_effect(9.99) == 9.95
    assert round_price_effect(15.82) == 15.5
    assert round_price_effect(103.5) == 103
    assert round_price_effect(687) == 685

