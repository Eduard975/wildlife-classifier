from rafl.fl.controller import ControllerConfig, ResourceAwareController, payload_bytes
from rafl.fl.resources import EnvConfig, ResourceEnv
from rafl.fl.sim import SimConfig, fixed_policy, run
from test_fl import _toy

N = 1280 * 5 + 5


def _state(**kw):
    s = {"battery_soc": 0.8, "uplink_kbps": 100.0, "cpu_load": 0.1, "temp_c": 45.0,
         "deadline_s": 12.0, "train_s_per_sample_epoch": 2e-4, "n_samples": 100}
    s.update(kw)
    return s


def test_controller_picks_widest_fitting_width():
    ctl = ResourceAwareController(ControllerConfig(max_bits=8), N)
    assert ctl(1, "a", _state()).bits == 8
    # 8-bit (~6.4 KB) needs ~7 kbps at a 7.2 s budget; 4-bit needs ~3.6 kbps
    assert ctl(1, "a", _state(uplink_kbps=5.0)).bits == 4
    assert ctl(1, "a", _state(uplink_kbps=2.5)).bits == 2
    assert not ctl(1, "a", _state(uplink_kbps=0.5)).participate


def test_controller_protects_detection_duty():
    ctl = ResourceAwareController(ControllerConfig(), N)
    assert not ctl(1, "a", _state(battery_soc=0.1)).participate
    assert not ctl(1, "a", _state(cpu_load=0.9)).participate
    assert not ctl(1, "a", _state(temp_c=90.0)).participate
    assert ctl(1, "a", _state()).bits == 4  # max_bits default


def test_payload_matches_quantizer():
    from rafl.fl.quant import quantize
    import numpy as np
    rng = np.random.default_rng(0)
    for b in (2, 4, 8, 16, 32):
        got = quantize(np.ones(N - 5, np.float32) + rng.random(N - 5), b, rng)[1] + \
              quantize(rng.random(5).astype(np.float32), b, rng)[1]
        ctl = ResourceAwareController(ControllerConfig(), N)
        assert abs(ctl._bytes(b) - got) <= 1, b


def test_env_deadline_and_offline():
    env = ResourceEnv(["a"], EnvConfig(deadline_s=1.0))
    env.step(1)
    env._bw["a"] = 10.0  # kbps
    assert not env.settle("a", 10, 1, 25_000).accepted  # 20 s upload
    assert env.settle("a", 10, 1, 500).accepted
    env.c["a"].soc = 0.0
    assert not env.online("a")


def test_sim_with_env_runs_and_accounts_waste():
    x, meta, classes = _toy()
    cfg = SimConfig(rounds=10, clients_per_round=4, label_noise=0.0, eval_every=10)
    f = lambda ids: ResourceEnv(ids, EnvConfig(bw_median_kbps=0.5, deadline_s=1.0))
    h = run(x, meta, classes, cfg, fixed_policy(32), env_factory=f, log=lambda r: None)
    assert h[-1]["cum_wasted_mb"] > 0 and "camera_offline_pct" in h[-1]
    ctl = ResourceAwareController(ControllerConfig(), 16 * 3 + 3)
    h2 = run(x, meta, classes, cfg, ctl, env_factory=f, log=lambda r: None)
    assert h2[-1]["cum_wasted_mb"] <= h[-1]["cum_wasted_mb"]
