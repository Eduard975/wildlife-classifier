import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_crops", Path(__file__).resolve().parents[1] / "scripts/build_crops.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def test_square_inside_frame_and_square():
    for box in [(0.0, 0.0, 0.1, 0.1), (0.95, 0.9, 0.05, 0.1), (0.4, 0.4, 0.2, 0.3), (0, 0, 1, 1)]:
        l, t, r, b = bc.square(*box, 2048, 1536)
        assert 0 <= l and 0 <= t and r <= 2048 and b <= 1536
        assert abs((r - l) - (b - t)) <= 1


def test_square_contains_box_when_it_fits():
    l, t, r, b = bc.square(0.4, 0.4, 0.1, 0.2, 1000, 1000)
    assert l <= 400 and t <= 400 and r >= 500 and b >= 600
