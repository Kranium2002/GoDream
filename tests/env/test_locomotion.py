"""Phase 1 locomotion gate: the borrowed policy walks, turns, stops and stands in the real simulator."""
import pytest

from learned_judgment.env.acceptance import run_locomotion, run_stop

pytestmark = pytest.mark.system

@pytest.mark.parametrize("vx", [0.3, 0.6, 0.9])
def test_trots_thirty_seconds(config, vx):
    r = run_locomotion(config, vx=vx, yaw_rate=0.0, duration_s=30, seed=config["runtime"]["seed"])
    assert not r["fallen"]
    assert abs(r["mean_speed_m_s"] - vx) <= config["robot"]["acceptance"]["speed_tolerance_fraction"] * vx

@pytest.mark.parametrize("sign", [1, -1])
def test_turns(config, sign):
    yaw = sign * 0.5 * max(config["robot"]["acceptance"]["yaw_rates_supported_by_policy"])
    r = run_locomotion(config, vx=0.3, yaw_rate=yaw, duration_s=10, seed=config["runtime"]["seed"])
    assert not r["fallen"]
    assert abs(r["mean_yaw_rate_rad_s"] - yaw) <= config["robot"]["acceptance"]["yaw_tolerance_fraction"] * abs(yaw)

def test_stops_and_stands(config):
    # 5 s at 0.6 m/s, then 5 s at zero command, in one episode
    r = run_stop(config, vx=0.6, trot_s=5, stop_s=5, seed=config["runtime"]["seed"])
    assert not r["fallen"]
    assert r["final_speed_m_s"] < config["robot"]["acceptance"]["stop_speed_m_s"]

def test_stands_thirty_seconds(config):
    assert not run_locomotion(config, vx=0.0, yaw_rate=0.0, duration_s=30, seed=config["runtime"]["seed"])["fallen"]
