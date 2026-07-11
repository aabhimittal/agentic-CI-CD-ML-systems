"""Tests for the fleet / robotics-task simulator."""

from __future__ import annotations

from agentic_deploy.robotics_sim import BehaviorProfile, FleetSimulator


def _signals(snap) -> dict:
    # Compare the generated signals, not the wall-clock timestamp.
    d = snap.to_dict()
    d.pop("timestamp", None)
    return d


def test_seed_is_deterministic():
    profile = BehaviorProfile(cand_task_success_rate=0.8, cand_error_rate=0.09)
    a = FleetSimulator(profile, fleet_size=50, seed=42).observe(50.0, 0)
    b = FleetSimulator(profile, fleet_size=50, seed=42).observe(50.0, 0)
    assert _signals(a) == _signals(b)


def test_zero_traffic_matches_baseline():
    # At 0% candidate traffic the fleet is entirely on the baseline model.
    profile = BehaviorProfile(
        base_task_success_rate=0.98,
        cand_task_success_rate=0.5,   # a terrible candidate...
        noise=0.0,                    # ...but 0% traffic + no noise → pure baseline
    )
    snap = FleetSimulator(profile, fleet_size=50, seed=1).observe(0.0, 0)
    assert abs(snap.task_success_rate - 0.98) < 1e-9
    assert snap.safety_incidents == 0


def test_full_traffic_matches_candidate():
    profile = BehaviorProfile(
        base_error_rate=0.01,
        cand_error_rate=0.09,
        noise=0.0,
    )
    snap = FleetSimulator(profile, fleet_size=50, seed=1).observe(100.0, 0)
    assert abs(snap.service_error_rate - 0.09) < 1e-9


def test_no_safety_rate_means_no_incidents():
    profile = BehaviorProfile(cand_safety_rate=0.0)
    snap = FleetSimulator(profile, fleet_size=100, seed=7).observe(100.0, 5)
    assert snap.safety_incidents == 0


def test_latent_regression_dormant_before_activation():
    # A regression that activates at step 2 behaves like baseline at step 0/1.
    profile = BehaviorProfile(
        base_task_success_rate=0.98,
        cand_task_success_rate=0.5,
        cand_safety_rate=10.0,
        regression_activates_at_step=2,
        noise=0.0,
    )
    sim = FleetSimulator(profile, fleet_size=50, seed=3)
    early = sim.observe(50.0, 1)   # dormant → baseline behavior, no incidents
    assert early.safety_incidents == 0
    assert abs(early.task_success_rate - 0.98) < 1e-9


def test_safety_incidents_appear_when_active():
    profile = BehaviorProfile(cand_safety_rate=12.0, regression_activates_at_step=0)
    # High rate at full candidate traffic → at least one incident (seeded).
    snap = FleetSimulator(profile, fleet_size=50, seed=0).observe(100.0, 0)
    assert snap.safety_incidents >= 1


def test_from_dict_ignores_unknown_keys():
    profile = BehaviorProfile.from_dict({"cand_error_rate": 0.05, "bogus": 123})
    assert profile.cand_error_rate == 0.05
