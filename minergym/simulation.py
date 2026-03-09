"""It would be nice to be able to use energyplus through some familiar reset &
step API. However, energyplus was not made for this. This module implements all
the code glue needed to make the .step function work.

Note, however, that at this point, the concept of a reward doesn't exist yet.
This module only cares about control.

"""

import collections
import logging
import os
import pathlib
import queue
import threading
import traceback
import weakref
from ctypes import c_void_p
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar, Union

import optree
import optree.typing
import pyenergyplus.api

from .channel import Channel

logger = logging.getLogger(__name__)


api = pyenergyplus.api.EnergyPlusAPI()


@dataclass
class ManagedState:
    inner: c_void_p

    def __init__(self):
        s = api.state_manager.new_state()
        self.inner = s

        def delete_state(ep_state: c_void_p):
            api.state_manager.delete_state(ep_state)
            logger.debug("Deleting energyplus state.")

        weakref.finalize(self, delete_state, s)


# Template holes and handles


@dataclass(frozen=True, slots=True)
class VariableHole:
    variable_name: str
    variable_key: str


@dataclass(frozen=True, slots=True)
class VariableHandle:
    handle: int


class InvalidVariable(Exception):
    def __init__(self, var: VariableHole):
        super().__init__(
            f"Invalid variable: name='{var.variable_name}', key='{var.variable_key}'"
        )


@dataclass(frozen=True, slots=True)
class ActuatorHole:
    component_type: str
    control_type: str
    actuator_key: str


@dataclass(frozen=True, slots=True)
class ActuatorHandle:
    handle: int


class InvalidActuator(Exception):
    def __init__(self, act: ActuatorHole):
        super().__init__(
            f"Invalid actuator: component_type='{act.component_type}', "
            f"control_type='{act.control_type}', actuator_key='{act.actuator_key}'"
        )


@dataclass(frozen=True, slots=True)
class MeterHole:
    meter_name: str


@dataclass(frozen=True, slots=True)
class MeterHandle:
    handle: int


class InvalidMeter(Exception):
    def __init__(self, met: MeterHole):
        super().__init__(f"Invalid meter: name='{met.meter_name}'")


@dataclass(frozen=True, slots=True)
class FunctionHole:
    function: Callable[[c_void_p], Any]


# types for things

_AnyHandle = VariableHandle | MeterHandle | ActuatorHandle | FunctionHole
AnyHole = VariableHole | MeterHole | ActuatorHole | FunctionHole


# communication choregraphy


@dataclass(frozen=True, slots=True)
class ShutDown:
    pass


@dataclass(frozen=True, slots=True)
class RunAction:
    act: Any


@dataclass(frozen=True, slots=True)
class ICrashed:
    exception: Exception


@dataclass(frozen=True, slots=True)
class IShutDown:
    pass


@dataclass(frozen=True, slots=True)
class IWantAction:
    response: Channel[RunAction | ShutDown]


@dataclass(frozen=True, slots=True)
class IGotObservation:
    observation: Any


E2PMessage = IWantAction | IGotObservation | IShutDown | ICrashed


class InvalidStateException(Exception):
    def __init__(self, wanted, got):
        super().__init__(f"Simulation is in invalid state. Wanted: {wanted}, is: {got}")


@dataclass(slots=True, frozen=True)
class StateInit:
    pass


@dataclass(slots=True)
class StateStarting:
    ep_state: ManagedState
    ep_thread: threading.Thread

    channel: Channel[E2PMessage]

    number_of_warmup_phases_completed: int = 0


@dataclass(slots=True)
class StateStarted:
    ep_state: ManagedState
    ep_thread: threading.Thread

    channel: Channel[E2PMessage]

    observation_handles: Any
    actuator_handles: Any

    last_observation: Any


@dataclass(slots=True, frozen=True)
class StateDone:
    last_observation: Any


@dataclass(slots=True, frozen=True)
class StateCrashed:
    pass


SimulationState = Union[StateInit, StateStarted, StateDone, StateCrashed]


@dataclass(slots=True)
class EnergyPlusSimulation:
    """"""

    """The building file used for the simulation."""
    building_path: Path

    """The weather file used for the simulation."""
    weather_path: Path

    """A PyTree containing `Variable` and `Meter` leaves."""
    observation_template: optree.typing.PyTree[AnyHole]

    actuators: optree.typing.PyTree[ActuatorHole]

    warmup_phases: int = 5

    n_steps: int = field(default=0, init=False)

    """The amount of steps before the simulation exits."""
    max_steps: int = 105_119

    """The directory in which energyplus will write its log files."""
    log_dir: Path = Path("eplus_output")

    """Wether to let energyplus print a bunch of stuff to stdout."""
    verbose: bool = True

    state: SimulationState = field(default=StateInit(), init=False)
    _last_set_actuators: dict[int, float] = field(default_factory=dict, init=False)
    _last_raw_action: Any | None = field(default=None, init=False)

    def _reverse_step(self):
        """Send the current observation, then receive an action and run it."""

        def get_handle_value(ep_state: c_void_p, han: _AnyHandle) -> float:
            if isinstance(han, VariableHandle):
                return api.exchange.get_variable_value(ep_state, han.handle)
            elif isinstance(han, MeterHandle):
                return api.exchange.get_meter_value(ep_state, han.handle)
            elif isinstance(han, ActuatorHandle):
                return api.exchange.get_actuator_value(ep_state, han.handle)
            elif isinstance(han, FunctionHole):
                result = han.function(ep_state)
                return result
            else:
                raise Exception(f"got a weird thing: {han}")

        if isinstance(self.state, StateStarted):
            state = self.state


            debug_actuators = os.environ.get("MINERGYM_DEBUG_ACTUATORS", "").strip() == "1"
            if debug_actuators and self._last_set_actuators:
                # Check if EnergyPlus changed actuator values between timesteps.
                for handle, prev in list(self._last_set_actuators.items()):
                    try:
                        cur = api.exchange.get_actuator_value(state.ep_state.inner, handle)
                    except Exception:
                        continue
                    if abs(float(cur) - float(prev)) > 1e-6:
                        logger.warning(
                            "Actuator value changed between timesteps (possible override): "
                            f"handle={handle} prev={prev} cur={cur}"
                        )

            obs = optree.tree_map(
                lambda han: get_handle_value(state.ep_state.inner, han),
                self.state.observation_handles,
            )

            self.state.last_observation = obs
            self.state.channel.put(IGotObservation(obs))

            response_chan = Channel[RunAction | ShutDown]()
            self.state.channel.put(IWantAction(response_chan))

            response = response_chan.get()
            if isinstance(response, RunAction):
                act = response.act
                # Cache the latest raw action so subclasses/callbacks can
                # re-apply actuator values at a later calling point if needed.
                self._last_raw_action = act
                if debug_actuators:
                    try:
                        logger.info(f"Applying raw action to EnergyPlus: {act}")
                    except Exception:
                        logger.info("Applying raw action to EnergyPlus (unprintable)")

                # same path and set its value.
                for accessor in optree.tree_accessors(act):
                    h: ActuatorHandle = accessor(self.state.actuator_handles)
                    the_value = accessor(act)

                    before = None
                    if debug_actuators:
                        try:
                            before = api.exchange.get_actuator_value(
                                self.state.ep_state.inner, h.handle
                            )
                        except Exception:
                            before = None

                    api.exchange.set_actuator_value(
                        self.state.ep_state.inner, h.handle, the_value
                    )

                    after = None
                    if debug_actuators:
                        try:
                            after = api.exchange.get_actuator_value(
                                self.state.ep_state.inner, h.handle
                            )
                        except Exception:
                            after = None
                        logger.info(
                            f"set_actuator_value handle={h.handle} value={the_value} "
                            f"before={before} after={after}"
                        )

                    # Track last value we attempted to set for override detection.
                    self._last_set_actuators[h.handle] = float(the_value)
            elif isinstance(response, ShutDown):
                api.runtime.stop_simulation(self.state.ep_state.inner)
                return

    def callback_timestep(self, _) -> None:
        if isinstance(self.state, StateStarting):
            if not api.exchange.api_data_fully_ready(self.state.ep_state.inner):
                return

            if api.exchange.warmup_flag(self.state.ep_state.inner):
                return

            # The energyplus simulator has 5 warmup phases. If we start
            # evaluating setpoints and sending observations before all the
            # warmup phases are all done, the policy will see the date jump
            # around, which is bad.
            if self.state.number_of_warmup_phases_completed < self.warmup_phases:
                return

            try:
                obs, act = self.construct_handles(self.state.ep_state.inner)
                new_state = StateStarted(
                    self.state.ep_state,
                    self.state.ep_thread,
                    self.state.channel,
                    obs,
                    act,
                    None,
                )
            except Exception as e:
                api.runtime.stop_simulation(self.state.ep_state.inner)
                old_state = self.state
                self.state = StateCrashed()
                old_state.channel.put(ICrashed(e))
                return

            self.state = new_state  # type: ignore

            try:
                self._reverse_step()
            except Exception as e:
                api.runtime.stop_simulation(self.state.ep_state.inner)
                old_state = self.state
                self.state = StateCrashed()
                old_state.channel.put(ICrashed(e))
                return

        elif isinstance(self.state, StateStarted):
            self._reverse_step()
        else:
            raise Exception("TODO")

        self.n_steps += 1

    def register_callbacks(self, ep_state: c_void_p) -> None:
        """Register runtime callbacks.

        Split out for testability and extensibility: subclasses may register
        additional callbacks while keeping the core "pause, get action, step"
        protocol unchanged.
        """
        api.runtime.callback_after_predictor_after_hvac_managers(
            ep_state, self.callback_timestep
        )
        # Re-apply the last action inside the HVAC iteration loop. Some HVAC
        # component actuators can be overwritten later in the same timestep by
        # EnergyPlus managers/controllers; this calling point is late enough for
        # the override to "stick".
        # api.runtime.callback_inside_system_iteration_loop(
        #     ep_state, self._callback_inside_system_iteration_loop
        # )

    # def _callback_inside_system_iteration_loop(self, _state: c_void_p) -> None:
    #     # This callback runs frequently (per HVAC system iteration). It MUST be
    #     # non-blocking and fast.
    #     if not isinstance(self.state, StateStarted):
    #         return

    #     # Avoid interfering with warmup/sizing phases.
    #     if not api.exchange.api_data_fully_ready(self.state.ep_state.inner):
    #         return
    #     if api.exchange.warmup_flag(self.state.ep_state.inner):
    #         return

    #     act = self._last_raw_action
    #     if act is None:
    #         return

    #     # Re-apply actuator values so they are not overwritten by later HVAC logic.
    #     for accessor in optree.tree_accessors(act):
    #         h: ActuatorHandle = accessor(self.state.actuator_handles)
    #         the_value = accessor(act)
    #         api.exchange.set_actuator_value(
    #             self.state.ep_state.inner, h.handle, the_value
    #         )

    def construct_handles(self, state: c_void_p) -> tuple[Any, Any]:
        if self.verbose:
            print("constructing handles")

        # Most of Variable, Meter, Actuator need to be converted (by a
        # running simulation) into a not-so-human-readable numerical handle.
        # This is what we do here.

        def get_hole_handle(o: AnyHole) -> _AnyHandle:
            if isinstance(o, VariableHole):
                var = o
                han = api.exchange.get_variable_handle(
                    state,
                    var.variable_name,
                    var.variable_key,
                )
                if han < 0:
                    raise InvalidVariable(var)
                return VariableHandle(han)
            elif isinstance(o, MeterHole):
                met = o
                han = api.exchange.get_meter_handle(state, met.meter_name)
                if han < 0:
                    raise InvalidMeter(met)
                return MeterHandle(han)
            elif isinstance(o, ActuatorHole):
                act = o
                han = api.exchange.get_actuator_handle(
                    state,
                    act.component_type,
                    act.control_type,
                    act.actuator_key,
                )
                if han < 0:
                    raise InvalidActuator(act)
                return ActuatorHandle(han)
            elif isinstance(o, FunctionHole):
                # We don't need to turn it into anything else.
                return o
            else:
                raise Exception(f"got a weird thing: {o}")

        observation_handles = optree.tree_map(
            get_hole_handle,
            self.observation_template,
        )

        def get_actuator_handle(act: ActuatorHole) -> ActuatorHandle:
            han = api.exchange.get_actuator_handle(
                state, act.component_type, act.control_type, act.actuator_key
            )
            if han < 0:
                raise InvalidActuator(act)
            return ActuatorHandle(han)

        actuator_handles = optree.tree_map(
            get_actuator_handle,
            self.actuators,
        )

        return observation_handles, actuator_handles

    def start(self) -> tuple[Any, bool]:
        managed_ep_state: ManagedState = ManagedState()

        def eplus_thread():
            """Thread running the energyplus simulation."""
            if not isinstance(self.state, StateStarting):
                raise InvalidStateException()
            args: list[str | bytes] = [
                "-d",
                str(self.log_dir),
                "-w",
                str(self.weather_path),
                str(self.building_path),
            ]

            exit_code = api.runtime.run_energyplus(
                managed_ep_state.inner,
                args,
            )

            old_state = self.state

            if exit_code == 0:
                if isinstance(self.state, StateStarted):
                    self.state = StateDone(self.state.last_observation)
                    old_state.channel.put(IShutDown())
                elif isinstance(self.state, StateCrashed):
                    # If the simulation exited abnormally, we don't need to change
                    # the state as that has already been done by the code that
                    # caught the exception.
                    pass
                elif isinstance(self.state, StateStarting):
                    old_state = self.state
                    self.state = StateCrashed()
                    old_state.channel.put(
                        ICrashed(
                            Exception(
                                "The simulation exited before it finished starting. "
                                "Perhaps your warmup_phases parameter is too high? "
                                f"{old_state.number_of_warmup_phases_completed=}"
                            )
                        )
                    )

                else:
                    assert False, "Should be unreachable."
            else:
                old_state = self.state
                if isinstance(self.state, StateStarting):
                    # probably a problem in the epjson file or something like that
                    self.state = StateCrashed()
                    old_state.channel.put(
                        ICrashed(
                            Exception(
                                "Simulation crashed before it could produce a single observation."
                            )
                        )
                    )

                elif isinstance(self.state, StateStarted):
                    # probably an "invalid input" problem
                    self.state = StateCrashed()
                    old_state.channel.put(
                        ICrashed(Exception("Simulation crashed while running."))
                    )
                else:
                    assert False, "Should be unreachable."

        # We must request access to Variable before the simulation is started.
        def request_var(var):
            if isinstance(var, VariableHole):
                api.exchange.request_variable(
                    managed_ep_state.inner,
                    var.variable_name,
                    var.variable_key,
                )

        # breakpoint()
        optree.tree_map(
            request_var,
            self.observation_template,
            is_leaf=lambda x: isinstance(x, VariableHole),
        )

        self.register_callbacks(managed_ep_state.inner)

        def warmup_callback(state: c_void_p):
            if self.verbose:
                print("warmup phase complete")

            if isinstance(self.state, StateStarting):
                self.state.number_of_warmup_phases_completed += 1

        api.runtime.set_console_output_status(managed_ep_state.inner, self.verbose)

        api.runtime.callback_after_new_environment_warmup_complete(
            managed_ep_state.inner, warmup_callback
        )

        thread = threading.Thread(target=eplus_thread, daemon=True)
        new_state = StateStarting(
            managed_ep_state,
            thread,
            Channel(),
        )
        self.state = new_state
        thread.start()

        msg = self.state.channel.get()
        if isinstance(msg, IGotObservation):
            return msg.observation, False
        elif isinstance(msg, ICrashed):
            raise msg.exception
        else:
            assert False, "Should be unreachable."

    def step(self, action: dict[str, float]) -> tuple[Any, bool]:
        if isinstance(self.state, StateStarted):
            msg1 = self.state.channel.get()
            if isinstance(msg1, IWantAction):
                msg1.response.put(RunAction(action))
                msg2 = self.state.channel.get()
                if isinstance(msg2, IGotObservation):
                    obs = msg2.observation
                    return obs, False
                elif isinstance(msg2, IShutDown):
                    return self.state.last_observation, True
                elif isinstance(msg2, ICrashed):
                    raise msg2.exception
                else:
                    assert False, "Should be unreachable."
            else:
                raise Exception("TODO")
        elif isinstance(self.state, StateDone):
            return self.state.last_observation, True
        else:
            raise InvalidStateException(StateStarted, self.state)

    def stop(self):
        """When in a started state, stop the simulation."""
        if not isinstance(self.state, StateStarted):
            raise InvalidStateException(StateStarted, self.state)

        state = self.state
        msg1 = state.channel.get()
        if isinstance(msg1, IWantAction):
            msg1.response.put(ShutDown())

            msg2 = state.channel.get()
            if isinstance(msg2, IShutDown):
                return  # All is good
            else:
                assert False, "Should be unreachable."

        else:
            assert False, "Should be unreachable."

    def try_stop(self):
        """When in any state, try to stop the simulation to free resources."""
        if isinstance(self.state, StateInit):
            # nothing to do
            return
        if isinstance(self.state, StateStarted):
            self.stop()
        elif isinstance(self.state, StateStarting):
            raise NotImplemented
        elif isinstance(self.state, StateCrashed):
            # nothing to do
            return

    def get_api_endpoints(
        self,
    ) -> list[Union[ActuatorHole, VariableHole, MeterHole]]:
        if not isinstance(self.state, StateStarted):
            raise InvalidStateException(StateStarted, self.state)

        exchange_points = api.exchange.get_api_data(self.state.ep_state.inner)

        out: list[Union[ActuatorHole, VariableHole, MeterHole]] = []
        for v in exchange_points:
            if v.what == "Actuator":
                out.append(ActuatorHole(v.name, v.type, v.key))
            elif v.what == "OutputVariable":
                out.append(VariableHole(v.name, v.key))
            elif v.what == "OutputMeter":
                out.append(MeterHole(v.key))
            elif v.what == "InternalVariable":
                continue
            else:
                raise RuntimeError("Unreachable")

        return out


