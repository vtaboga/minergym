from importlib import resources
from pathlib import Path

import minergym.simulation as simulation
import pytest
from minergym.data.building import crawlspace
from minergym.data.weather import honolulu


def test_simulation_runs() -> None:
    sim = simulation.EnergyPlusSimulation(crawlspace, honolulu, None, {}, max_steps=100)

    obs, done = sim.start()
    while not done:
        assert obs is None
        obs, done = sim.step({})


def test_simulation_obs() -> None:
    obs_template = {
        "temp": simulation.VariableHole("ZONE AIR TEMPERATURE", "crawlspace")
    }

    sim = simulation.EnergyPlusSimulation(
        crawlspace, honolulu, obs_template, {}, max_steps=100
    )
    obs, done = sim.start()
    assert obs["temp"] != 0.0


def test_simulation_invalid_variable() -> None:
    # an invalid variable
    obs_template = simulation.VariableHole("", "")

    sim = simulation.EnergyPlusSimulation(
        crawlspace, honolulu, obs_template, {}, max_steps=100
    )

    with pytest.raises(simulation.InvalidVariable) as exn_info:
        obs, done = sim.start()


def test_simulation_invalid_meter() -> None:
    obs_template = simulation.MeterHole("")

    sim = simulation.EnergyPlusSimulation(
        crawlspace, honolulu, obs_template, {}, max_steps=100
    )

    with pytest.raises(simulation.InvalidMeter) as exn_info:
        obs, done = sim.start()


def test_simulation_time() -> None:
    obs = {}
    obs["current_time"] = simulation.FunctionHole(simulation.api.exchange.current_time)
    obs["day_of_year"] = simulation.FunctionHole(simulation.api.exchange.day_of_year)

    sim = simulation.EnergyPlusSimulation(crawlspace, honolulu, obs, {}, max_steps=1000)
    obs, done = sim.start()
    while True:
        new_obs, done = sim.step({})
        if done:
            break
        # We would expect the day of year to go up linearly. However, this is
        # not currently (2024-11-11T12:34:20) the case. It seems that after the
        # initial warmup is done, a single day will pass (day 202) before a
        # second period of warmup which will be followed by the start of the
        # real RunPeriod (day 1).
        assert new_obs["day_of_year"] - obs["day_of_year"] in [0, 1]
        obs = new_obs


def test_simulation_missing_file() -> None:
    sim = simulation.EnergyPlusSimulation(Path("does_not_exist"), honolulu, {}, {})

    with pytest.raises(Exception):
        sim.start()


def test_simulation_repr() -> None:
    sim = simulation.EnergyPlusSimulation(crawlspace, honolulu, {}, {})
    print(sim)


def test_get_api_endpoints() -> None:
    sim = simulation.EnergyPlusSimulation(crawlspace, honolulu, {}, {})
    sim.start()

    # Will crash is some unknown api object is encountered
    sim.get_api_endpoints()


def test_simulation_set_actuator_value() -> None:
    actuators = {
        "heating_sch": simulation.ActuatorHole(
            component_type="Schedule:Compact",
            control_type="Schedule Value",
            actuator_key="heating_sch",
        ),
        "cooling_sch": simulation.ActuatorHole(
            component_type="Schedule:Compact",
            control_type="Schedule Value",
            actuator_key="cooling_sch",
        ),
    }

    sim = simulation.EnergyPlusSimulation(
        crawlspace, honolulu, actuators, actuators, max_steps=100
    )
    sim.start()

    obs, done = sim.step({"heating_sch": 15})
    # Right after we set the actuator, it should have the right value.
    assert obs["heating_sch"] == 15


def test_simulation_structured_actuators() -> None:
    actuators = {
        "zone_1": {
            "heating_sch": simulation.ActuatorHole(
                component_type="Schedule:Compact",
                control_type="Schedule Value",
                actuator_key="heating_sch",
            ),
            "cooling_sch": simulation.ActuatorHole(
                component_type="Schedule:Compact",
                control_type="Schedule Value",
                actuator_key="cooling_sch",
            ),
        }
    }

    sim = simulation.EnergyPlusSimulation(
        crawlspace, honolulu, actuators, actuators, max_steps=100
    )
    sim.start()

    obs, done = sim.step({"zone_1": {"heating_sch": 15}})
    assert obs["zone_1"]["heating_sch"] == 15


def test_simulation_stop() -> None:
    sim = simulation.EnergyPlusSimulation(crawlspace, honolulu, None, {}, max_steps=100)

    obs, done = sim.start()

    assert isinstance(sim.state, simulation.StateStarted)

    thread = sim.state.ep_thread

    for _ in range(10):
        assert obs is None
        obs, done = sim.step({})

    sim.stop()

    assert isinstance(sim.state, simulation.StateDone)

    assert not thread.is_alive()


def test_register_callbacks_energyplus_simulation(monkeypatch) -> None:
    calls: list[tuple[str, object, object]] = []

    def fake_begin(ep_state, cb):
        calls.append(("begin_system_timestep_before_predictor", ep_state, cb))

    def fake_inside(ep_state, cb):
        calls.append(("inside_system_iteration_loop", ep_state, cb))

    monkeypatch.setattr(
        simulation.api.runtime,
        "callback_begin_system_timestep_before_predictor",
        fake_begin,
    )
    monkeypatch.setattr(
        simulation.api.runtime,
        "callback_inside_system_iteration_loop",
        fake_inside,
    )

    sim = simulation.EnergyPlusSimulation(
        Path("building.epJSON"), Path("weather.epw"), {}, {}, verbose=False
    )
    ep_state = simulation.c_void_p()
    sim.register_callbacks(ep_state)

    assert ("begin_system_timestep_before_predictor", ep_state, sim.callback_timestep) in calls
    assert (
        "inside_system_iteration_loop",
        ep_state,
        sim._callback_inside_system_iteration_loop,
    ) in calls
