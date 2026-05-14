"""The simulation module provides a .step api, but without a reward function and
without a well shaped observation and action space. This module seeks to wrap
the more "primitive" api into a gymnasium environment that we can pass without
trouble to any gymnasium consumer.

"""

import gc
import logging
import shutil
import threading
import typing
from pathlib import Path

import gymnasium

import minergym.simulation as simulation

logger = logging.getLogger(__name__)

ObsType = typing.TypeVar("ObsType")
ActType = typing.TypeVar("ActType")

_DEFAULT_THREAD_JOIN_TIMEOUT: float = 10.0


class EnergyPlusEnvironment(gymnasium.Env, typing.Generic[ObsType, ActType]):
    """What does environment.EnergyPlusEnvironment have that
    simulation.EnergyPlusSimulation doesn't?

    1. The ability to be reset()
    2. A well defined observation and action space.
    3. A well defined reward function.

    While this class wraps all of those components together, the user will still
    have to provide their own reward function, action & observation reshaper and
    a way to initialize an EnergyPlusSimulation object.

    """

    metadata = {"render_modes": []}

    """This function will be called each time we need to .reset() the
    environment."""
    make_energyplus: typing.Callable[[], simulation.EnergyPlusSimulation]

    reward_function: typing.Callable[
        [typing.Any],  # Raw observation
        float,  # Reward value
    ]

    observation_space: gymnasium.Space[ObsType]

    """This function will be called to transform the output of the raw
    EnergyPlus controller into a point of the observation_space."""
    observation_transform: typing.Callable[
        [typing.Any],  # Raw observation
        ObsType,  # Observation space
    ]

    action_space: gymnasium.Space[ActType]
    """This function will be called to transform a point in the action_space a
    raw EnergyPlus action."""
    action_transform: typing.Callable[
        [ActType],  # Action space
        typing.Any,  # Raw action
    ]

    ep: simulation.EnergyPlusSimulation | None

    def __init__(
        self,
        make_energyplus: typing.Callable[[], simulation.EnergyPlusSimulation],
        reward_fn: typing.Callable[[typing.Any], float],
        observation_space: gymnasium.Space[ObsType],
        observation_transform: typing.Callable[[typing.Any], ObsType],
        action_space: gymnasium.Space[ActType],
        action_transform: typing.Callable[[ActType], typing.Any],
        eplus_output_dir: Path | None = None,
        cleanup_output_dir_on_close: bool = False,
        thread_join_timeout: float = _DEFAULT_THREAD_JOIN_TIMEOUT,
    ):
        super(EnergyPlusEnvironment, self).__init__()
        self.make_energyplus = make_energyplus
        self.reward_fn = reward_fn
        self.observation_space = observation_space
        self.observation_transform = observation_transform

        self.action_space = action_space
        self.action_transform = action_transform
        self.ep = None

        self.eplus_output_dir: Path | None = eplus_output_dir
        self.cleanup_output_dir_on_close: bool = cleanup_output_dir_on_close
        self.thread_join_timeout: float = thread_join_timeout

    def close(self) -> None:
        # Gate cleanup on whether a simulation was actually running.
        # close() is called polymorphically from reset() (after the subclass
        # may have already nulled self.ep), so this guard makes it idempotent.
        had_ep = self.ep is not None
        if had_ep:
            # Capture the thread reference before try_stop() transitions the
            # simulation state from StateStarted to StateDone (which drops
            # the ep_thread attribute from the state object).
            ep_sim_state = getattr(self.ep, "state", None)
            ep_thread = getattr(ep_sim_state, "ep_thread", None)

            self.ep.try_stop()

            if isinstance(ep_thread, threading.Thread) and ep_thread.is_alive():
                ep_thread.join(timeout=self.thread_join_timeout)
                if ep_thread.is_alive():
                    logger.warning(
                        "EnergyPlus thread did not exit within %.1fs; "
                        "resources may leak.",
                        self.thread_join_timeout,
                    )

            self.ep = None

        gc.collect()

        if had_ep and self.cleanup_output_dir_on_close and self.eplus_output_dir is not None:
            shutil.rmtree(self.eplus_output_dir, ignore_errors=True)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ) -> typing.Tuple[typing.Any, dict[str, typing.Any]]:
        super().reset()
        self.close()

        # Recreate the output directory so EnergyPlus has a clean destination.
        # close() may have rmtree'd it (when cleanup_output_dir_on_close=True),
        # so mkdir here ensures the directory exists before make_energyplus().
        if self.eplus_output_dir is not None:
            self.eplus_output_dir.mkdir(parents=True, exist_ok=True)

        self.ep = self.make_energyplus()
        obs, over = self.ep.start()
        return self.observation_transform(obs), {"raw_observation": obs}

    def step(self, action) -> typing.Tuple[ObsType, float, bool, bool, typing.Any]:
        if self.ep is None:
            raise Exception("Environment has not been started")
        # Do something about the actions
        a = self.action_transform(action)
        obs, finished = self.ep.step(a)
        # print(f"{obs}, {finished}")
        if not finished:
            transformed_obs = self.observation_transform(obs)
            self.last_obs = transformed_obs
            return (
                transformed_obs,
                self.reward_fn(obs),
                False,
                False,
                {"raw_observation": obs},
            )
        else:
            return (self.last_obs, 0.0, True, False, {})
