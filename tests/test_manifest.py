"""Invariants on the built manifest. Skipped until build_crops.py has produced it."""
from pathlib import Path

import pandas as pd
import pytest

M = Path(__file__).resolve().parents[1] / "data/manifests/crops.parquet"
pytestmark = pytest.mark.skipif(not M.exists(), reason="manifest not built")


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(M)


def test_no_location_in_two_splits(df):
    assert (df.groupby("location_id").split.nunique() == 1).all()


def test_no_image_in_two_splits(df):
    assert (df.groupby("url").split.nunique() == 1).all()


def test_every_class_in_every_split(df):
    ok = df[df.ok]
    assert (ok.groupby(["label", "split"]).size().unstack().fillna(0) > 0).all().all()


def test_known_classes_and_licence(df):
    assert set(df.label) <= {"bear", "wolf", "lynx", "wild_boar", "other"}
    assert df.licence.notna().all()


def test_failure_rate_low(df):
    assert (~df.ok).mean() < 0.02
