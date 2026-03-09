import gymnasium
import minergym.environment as environment
import minergym.simulation as simulation
import numpy as np
import pytest
from minergym.data.building import crawlspace
from minergym.data.weather import honolulu


def make_energyplus():
    return simulation.EnergyPlusSimulation(
        crawlspace,
        honolulu,
        None,
        {},
        max_steps=100,
    )


empty_space = gymnasium.spaces.Box(
    low=np.array([]), high=np.array([]), shape=(0,), dtype=np.float32
)


def test_environment() -> None:
    env = environment.EnergyPlusEnvironment(
        make_energyplus,
        lambda _: 0.0,
        empty_space,
        lambda _: np.array([]),
        empty_space,
        lambda _: {},
    )

    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, done, _, _ = env.step(env.action_space.sample())


# def test_reset() -> None:
#     env = environment.EnergyPlusEnvironment(
#         make_energyplus,
#         lambda _: 0.0,
#         empty_space,
#         lambda _: np.array([]),
#         empty_space
#     )
