from rafl.edge.telemetry import Telemetry, _util
from rafl.fl.controller import ControllerConfig, ResourceAwareController
from rafl.fl.resources import ResourceEnv


def test_util_from_jiffies():
    a = [100, 0, 100, 700, 100, 0, 0, 0]
    b = [150, 0, 150, 850, 100, 0, 0, 0]
    assert abs(_util(a, b) - 0.4) < 1e-9


def test_observe_has_simulation_keys_and_drives_controller():
    env = ResourceEnv(["c0"])
    tel = Telemetry(iface="lo")
    tel.record_upload(6000, 1.0)
    obs = tel.observe(battery_soc=0.8, n_samples=308)
    assert set(env.observe("c0")) <= set(obs)
    assert obs["uplink_kbps"] == 48.0
    plan = ResourceAwareController(ControllerConfig(), n_params=6405)(1, "board", obs)
    assert plan.participate in (True, False)
