from rafl.data.split import assign_split, split_locations, no_location_overlap

FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}


def test_assignment_is_deterministic():
    assert assign_split("loc_42", FRACS) == assign_split("loc_42", FRACS)


def test_every_location_one_split_no_overlap():
    locs = [f"loc_{i}" for i in range(500)]
    m = split_locations(locs, FRACS)
    assert set(m) == set(locs)
    assert no_location_overlap(m)


def test_fractions_roughly_respected():
    locs = [f"loc_{i}" for i in range(5000)]
    m = split_locations(locs, FRACS)
    counts = {"train": 0, "val": 0, "test": 0}
    for sp in m.values():
        counts[sp] += 1
    # within 3 percentage points of target for 5000 locations
    assert abs(counts["train"] / 5000 - 0.70) < 0.03
    assert abs(counts["val"] / 5000 - 0.15) < 0.03
    assert abs(counts["test"] / 5000 - 0.15) < 0.03
