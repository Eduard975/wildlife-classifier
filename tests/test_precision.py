from rafl.edge.precision import Policy, replay


def r(alert=False, danger_vote=0.0, boxes=1, max_weak=0.5, ms=100.0):
    return {"alert": alert, "danger_vote": danger_vote, "boxes": boxes, "max_weak": max_weak, "ms": ms}


def test_trigger_near_threshold_and_weak_boxes():
    p = Policy()
    assert p.trigger(r(danger_vote=0.5)) == "near"
    assert p.trigger(r(danger_vote=0.30)) is None  # below the margin
    assert p.trigger(r(danger_vote=0.9, alert=True)) is None
    assert p.trigger(r(boxes=0, max_weak=0.15)) == "weak"
    assert p.trigger(r(boxes=0, max_weak=0.05)) is None


def test_level_follows_battery_temperature_and_cpu():
    p = Policy()
    assert p.level({"battery_soc": 0.8, "temp_c": 40, "cpu_load": 0.1}) == "L2"
    assert p.level({"battery_soc": 0.8, "temp_c": 40, "cpu_load": 0.9}) == "L1"  # CPU busy: stay on the NPU
    assert p.level({"battery_soc": 0.8, "temp_c": 80, "cpu_load": 0.1}) == "L1"
    assert p.level({"battery_soc": 0.4, "temp_c": 40, "cpu_load": 0.1}) == "L1"
    assert p.level({"battery_soc": 0.2, "temp_c": 40, "cpu_load": 0.1}) is None
    assert p.frames({"battery_soc": 0.2}, 4) == 2 and p.frames({"battery_soc": 0.8}, 4) == 4


def test_replay_uses_escalated_result_and_counts_cost():
    log = [{"truth": "wolf", "levels": {"L0": r(danger_vote=0.5), "L1": r(), "L2": r(alert=True, danger_vote=0.7, ms=300)}},
           {"truth": "empty", "levels": {"L0": r(boxes=0, max_weak=0.0), "L1": r(), "L2": r()}}]
    out = replay(log, Policy(), {"battery_soc": 0.8, "temp_c": 40, "cpu_load": 0.1})
    assert out["alerts_dangerous"] == 1 and out["escalated"] == 1 and out["ms"] == 500.0
    out = replay(log, Policy(), {"battery_soc": 0.1, "temp_c": 40, "cpu_load": 0.1})
    assert out["alerts_dangerous"] == 0 and out["escalated"] == 0
