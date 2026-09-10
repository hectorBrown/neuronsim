from dataclasses import dataclass

import numpy as np
from numpy import bool, float32
from numpy.random import Generator
from numpy.typing import NDArray
from tqdm import tqdm

from latham import sampler
from latham.params import *


class State:
    def __init__(
        self,
        cell_params: CellParams,
        synaptic_params: SynapticParams,
        network_params: NetworkParams,
        time_step: float,
        random_seed: int,
    ):
        rng = np.random.default_rng(random_seed)
        # initialise voltages at random potentials between V_r and V_t --
        # without this there is no access to the metastable states described in
        # the paper without endogenously active cells
        self.potns: NDArray[float32] = cell_params.v_r + (
            cell_params.v_t - cell_params.v_r
        ) * rng.random(network_params.n, dtype=float32)
        # initialise all conductances at rest (0)
        self.conductances: NDArray[float32] = np.zeros(
            (2, network_params.n), dtype=float32
        )
        # initalise all synaptic currents at rest (0)
        self.synaptic_currents: NDArray[float32] = np.zeros(
            (2, network_params.n), dtype=float32
        )
        # initialise the applied current vector as a boxcar between 0 and I_max
        self.applied_currents: NDArray[float32] = (cell_params.i_max) * rng.random(
            network_params.n, dtype=float32
        )
        # distribute neurons in space
        self.positions: NDArray[float32] = State._get_neuron_positions(
            rng, network_params
        )

        # initialise a neuron_types array of booleans (False -> Inhibitory, True -> Excitatory)
        number_inhib = int(
            network_params.inhib_fraction * network_params.n
        )  # number of inhibitory neurons
        self.neuron_types: NDArray[bool] = np.concat(
            [np.zeros(number_inhib), np.ones(network_params.n - number_inhib)]
        ).astype(bool)  # excitatory represented by 1, inhibitory by 0

        # initialise an array of reversal potentials based on neuron type
        self.rev_potentials: NDArray[float32] = (
            self.neuron_types * network_params.excit_rev_pot
            + (1 - self.neuron_types) * network_params.inhib_rev_pot
        ).astype(float32)

        # find the connection matrix
        self.connectivity: NDArray[float32] = State._get_connection_matrix(
            number_inhib,
            self.positions,
            cell_params,
            synaptic_params,
            network_params,
            rng,
        )

        self.time_step = time_step

        # precompute decay steps
        self.conductance_decay_half_step: NDArray[float32] = np.vstack(
            (
                np.ones(network_params.n) * np.exp(-time_step / 2 / cell_params.tau_k),
                np.ones(network_params.n)
                * np.exp(-time_step / 2 / cell_params.tau_k_ca),
            ),
            dtype=float32,
        )
        self.conductance_decay_full_step: NDArray[float32] = np.pow(
            self.conductance_decay_half_step, 2, dtype=float32
        )
        self.synaptic_current_decay_half_step: NDArray[float32] = np.ones(
            (2, network_params.n), dtype=float32
        ) * np.exp(-time_step / 2 / synaptic_params.tau_s, dtype=float32)
        self.synaptic_current_decay_full_step: NDArray[float32] = np.pow(
            self.synaptic_current_decay_half_step, 2, dtype=float32
        )

        # precompute update to conductance on spike
        self.spike_conductance_update: NDArray[float32] = np.vstack(
            (
                np.ones(network_params.n) * cell_params.delta_g_k,
                np.ones(network_params.n) * cell_params.delta_g_k_ca,
            ),
            dtype=float32,
        )

    def _get_neuron_positions(
        rng: Generator, network_params: NetworkParams
    ) -> NDArray[float32]:
        def radial_dist(r_sample, delta_r):
            return (
                (
                    2 * np.pi * r_sample
                )  # Jacobian normalisation (strictly 2\pi isnt necessary)
                * (1 - np.tanh((np.pow(r_sample, 2) - 1) / delta_r))
                / (np.pi * delta_r * np.log(1 + np.exp(2 / delta_r)))
            )

        # distribute uniformly azimuthally
        theta: NDArray[float32] = (
            2 * np.pi * rng.random(network_params.n, dtype=float32)
        )

        # use a rejection sampler to find radial positions in line with the distribution function
        r = sampler.rejection(
            rng,
            lambda r_sample: radial_dist(r_sample, network_params.delta_r),
            network_params.n,
            0,
            2,  # chosen as the distribution ~ 0 here, could go larger -- don't think its necessary
        )
        return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=float32).T

    def _get_connection_matrix(
        number_inhib: int,
        positions: NDArray[float32],
        cell_params: CellParams,
        synaptic_params: SynapticParams,
        network_params: NetworkParams,
        rng: Generator,
    ) -> NDArray[float32]:
        def get_connection_weight(
            v_psp: float,
            rev_pot: float,
            cell_params: CellParams,
            synaptic_params: SynapticParams,
        ):
            return (
                v_psp
                / (rev_pot - cell_params.v_r)
                / synaptic_params.r_s
                * cell_params.tau_cell
                / synaptic_params.tau_s
                * np.exp(
                    np.log(cell_params.tau_cell / synaptic_params.tau_s)
                    / (cell_params.tau_cell / synaptic_params.tau_s - 1)
                )
            )

        def get_Z(spread) -> float32:
            return (
                np.float32(1)
                if spread == np.inf
                else 2 * np.pow(spread, 2) * (1 - np.exp(-1 / 2 / np.pow(spread, 2)))
            )

        number_excit = network_params.n - number_inhib

        # create four submatrices corresponding to the probability of
        # connection between any two neurons at infinite range according only
        # to their types and compile them into P
        probs_inhib_inhib: NDArray[float32] = (
            network_params.k_i
            * network_params.b_i
            / (number_excit + number_inhib * network_params.b_i)
            * np.ones((number_inhib, number_inhib), dtype=float32)
        )
        probs_inhib_excit: NDArray[float32] = (
            network_params.k_e
            * network_params.b_e
            / (number_excit + number_inhib * network_params.b_e)
            * np.ones((number_inhib, number_excit), dtype=float32)
        )
        probs_excit_inhib: NDArray[float32] = (
            network_params.k_i
            / (number_excit + number_inhib * network_params.b_i)
            * np.ones((number_excit, number_inhib), dtype=float32)
        )
        probs_excit_excit: NDArray[float32] = (
            network_params.k_e
            / (number_excit + number_inhib * network_params.b_e)
            * np.ones((number_excit, number_excit), dtype=float32)
        )
        probs: NDArray[float32] = np.vstack(
            (
                np.hstack((probs_inhib_inhib, probs_inhib_excit)),
                np.hstack((probs_excit_inhib, probs_excit_excit)),
            )
        )
        # don't allow autapses
        np.fill_diagonal(probs, 0)

        # modulate P by axonal spread distribution
        # first normalise by Z array
        probs /= np.hstack(
            (
                np.ones((network_params.n, number_inhib))
                * get_Z(network_params.sigma_i),
                np.ones((network_params.n, number_excit))
                * get_Z(network_params.sigma_e),
            )
        )
        # find square euclidean norms between neurons
        dists: NDArray[float32] = positions[:, None] - positions[None, :]
        norms: NDArray[float32] = (dists * dists).sum(axis=2)
        variances: NDArray[float32] = np.hstack(
            (
                np.ones((network_params.n, number_inhib))
                * np.pow(network_params.sigma_i, 2),
                np.ones((network_params.n, number_excit))
                * np.pow(network_params.sigma_e, 2),
            ),
            dtype=float32,
        )

        # decrease P according to axonal spread
        probs *= np.exp(-norms / 2 / variances)

        connections = rng.random((network_params.n, network_params.n)) < probs

        weights_inhib = get_connection_weight(
            network_params.v_ipsp,
            network_params.inhib_rev_pot,
            cell_params,
            synaptic_params,
        )
        weights_excit = get_connection_weight(
            network_params.v_epsp,
            network_params.excit_rev_pot,
            cell_params,
            synaptic_params,
        )
        weights: NDArray[float32] = np.hstack(
            (
                weights_inhib * np.ones((network_params.n, number_inhib)),
                weights_excit * np.ones((network_params.n, number_excit)),
            ),
            dtype=float32,
        )
        return connections * weights

    def integrate_voltage(self, cell_params: CellParams) -> NDArray[float32]:
        """
        Performs RK4 integration for the membrane potential.

        Returns:
            float: The step in membrane potential across all neurons.
        """

        def potential_step(
            potns: np.ndarray,
            conductances: np.ndarray,
            synaptic_currents: np.ndarray,
            applied_currents: np.ndarray,
            cell_params: CellParams,
        ) -> NDArray[float32]:
            i = synaptic_currents[0]
            i_epsilon = synaptic_currents[1]
            g_k = conductances[0]
            g_k_ca = conductances[1]
            res = (
                (potns - cell_params.v_r)
                * (potns - cell_params.v_t)
                / (cell_params.v_t - cell_params.v_r)
            )
            res += applied_currents
            res += -(g_k + g_k_ca) * (potns - cell_params.epsilon_k)
            res += -(potns * i - i_epsilon)
            return res / cell_params.tau_cell

        k_1 = potential_step(
            self.potns,
            self.conductances,
            self.synaptic_currents,
            self.applied_currents,
            cell_params,
        )
        k_2 = potential_step(
            self.potns + k_1 * self.time_step / 2,
            self.conductances * self.conductance_decay_half_step,
            self.synaptic_currents * self.synaptic_current_decay_half_step,
            self.applied_currents,
            cell_params,
        )
        k_3 = potential_step(
            self.potns + k_2 * self.time_step / 2,
            self.conductances * self.conductance_decay_half_step,
            self.synaptic_currents * self.synaptic_current_decay_half_step,
            self.applied_currents,
            cell_params,
        )
        k_4 = potential_step(
            self.potns + k_3 * self.time_step,
            self.conductances * self.conductance_decay_full_step,
            self.synaptic_currents * self.synaptic_current_decay_full_step,
            self.applied_currents,
            cell_params,
        )
        return self.time_step / float32(6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

    def update_conductances(self, steps=1):
        if steps == 1:
            self.conductances *= self.conductance_decay_full_step
        else:
            self.conductances *= np.pow(self.conductance_decay_full_step, steps)

    def update_synaptic_currents(self, steps=1):
        if steps == 1:
            self.synaptic_currents *= self.synaptic_current_decay_full_step
        else:
            self.synaptic_currents *= np.pow(
                self.synaptic_current_decay_full_step, steps
            )

    def process_spikes(self, cell_params: CellParams, synaptic_params: SynapticParams):
        # boolean array of neurons for which a spike has occured
        spikes: NDArray[bool] = self.potns > cell_params.v_apex
        if spikes.sum() == 0:
            return spikes

        # reset membrane potentials for those
        self.potns = np.where(spikes, cell_params.v_repol, self.potns).astype(float32)

        # adjust conductances
        self.conductances += spikes * self.spike_conductance_update

        # find a reduced connectivity matrix for only js where spike has occured
        spike_connectivity: NDArray[float32] = self.connectivity[:, spikes]

        # this is effectively dotting the spike_connectivity matrix with the
        # spike vector (W_ij s^j) and doing the same multiplying the spike
        # vector piecewise with the corresponding reversal potentials for the
        # second synaptic current in parallel
        #
        # This is much faster than doing np.dot, as the spiking is sparse
        self.synaptic_currents += (
            np.array(
                [
                    np.sum(spike_connectivity, axis=1),
                    np.sum(
                        spike_connectivity * self.rev_potentials[spikes],
                        axis=1,
                    ),
                ],
                dtype=float32,
            )
            * synaptic_params.r_s
        )

        return spikes


@dataclass(frozen=True)
class IndexedTraceType:
    index: int


@dataclass(frozen=True)
class FullAverageTraceType:
    pass


@dataclass(frozen=True)
class ExcitatoryAverageTraceType:
    pass


@dataclass(frozen=True)
class InhibitoryAverageTraceType:
    pass


@dataclass(frozen=True)
class NoTraceType:
    pass


NO_TRACE_TYPE = NoTraceType()


type TraceType = (
    IndexedTraceType
    | FullAverageTraceType
    | ExcitatoryAverageTraceType
    | InhibitoryAverageTraceType
    | NoTraceType
)


type SimulationResult = (
    tuple[NDArray[float32], NDArray[bool], State]
    | tuple[NDArray[float32], NDArray[bool], State, NDArray[float32]]
    | tuple[NDArray[float32], NDArray[bool], State, NDArray[float32], NDArray[float32]]
)


def run_sim(
    cell_params: CellParams,
    synaptic_params: SynapticParams,
    network_params: NetworkParams,
    time_step: float = 1,  # ms
    total_steps: int = 100 * 1000,  # 100s
    trace_v: TraceType = NO_TRACE_TYPE,
    trace_g_k_ca: TraceType = NO_TRACE_TYPE,
    random_seed: int = 0,
) -> SimulationResult:
    """
    Run neuronal simulation.

    Args:
        cell_params (CellParams): The cell parameters for the simulation.
        synaptic_params (SynapticParams): The synaptic parameters for the simulation.
        network_params (NetworkParams): The network parameters for the simulation.
        time_step (float): The time step per-update of the simulation (ms). Defaults to 1.
        total_steps (int): The total number of steps to perform. Defaults to 100,000 -> 100s with a 1ms step.
        trace_v (TraceType): Sets whether a trace is required for the membrane potential.
        trace_g_k_ca (TraceType): Sets whether a trace is required for the slow after-hyperpolarization conductance.
        random_seed (int): A seed for the random number generator used for simulation values. Defaults to 0.

    Returns:
        NDArray[float32]: Array of shape (total_steps,) with the cumulative time at each step.
        NDArray[bool]: Array of shape (total_steps,N), where N is the number of
            neurons, which maps which neurons spiked at every step (spikes -> True).
        State: The final state of the simulation.
        (Optional) NDArray[float32]: A voltage trace if `trace_V_index` is set.
        (Optional) NDArray[float32]: A slow after-hyperpolarization conductance trace if `trace_g_K_Ca` if set.
    """
    state = State(cell_params, synaptic_params, network_params, time_step, random_seed)
    print("Successfully initialised simulation.")
    trace_v_acc = np.zeros(
        total_steps + 1,
        dtype=float32,
    )
    trace_g_k_ca_acc = np.zeros(
        total_steps + 1,
        dtype=float32,
    )

    spikes_s: NDArray[bool] = np.zeros((total_steps + 1, network_params.n)).astype(bool)
    t = np.arange(0, time_step * total_steps + time_step, time_step, dtype=float32)
    for i, _ in tqdm(
        enumerate(t),
        total=total_steps,
    ):
        # RK4 voltage step
        state.potns += state.integrate_voltage(cell_params)
        state.update_conductances()
        state.update_synaptic_currents()
        # get spikes
        spikes: NDArray[bool] = state.process_spikes(cell_params, synaptic_params)

        spikes_s[i] = spikes
        _update_trace(trace_v_acc, trace_v, state.potns, network_params, i)
        _update_trace(
            trace_g_k_ca_acc, trace_g_k_ca, state.conductances[1], network_params, i
        )

    res: tuple[NDArray[float32], NDArray[bool], State] = t, spikes_s, state

    match trace_v:
        case NoTraceType():
            pass
        case _:
            res: tuple[NDArray[float32], NDArray[bool], State, NDArray[float32]] = (
                res + (trace_v_acc,)
            )  # ty: ignore
    match trace_g_k_ca:
        case NoTraceType():
            pass
        case _:
            res: (
                tuple[NDArray[float32], NDArray[bool], State, NDArray[float32]]
                | tuple[
                    NDArray[float32],
                    NDArray[bool],
                    State,
                    NDArray[float32],
                    NDArray[float32],
                ]
            ) = res + (trace_g_k_ca_acc,)  # ty: ignore
    return res


def _update_trace(
    accumulator: NDArray[float32],
    type: TraceType,
    state: NDArray[float32],
    network_params: NetworkParams,
    step: int,
):
    match type:
        case FullAverageTraceType():
            accumulator[step] = state.sum() / network_params.n
        case ExcitatoryAverageTraceType():
            accumulator[step] = state.sum() / (
                network_params.n - int(network_params.inhib_fraction * network_params.n)
            )
        case InhibitoryAverageTraceType():
            accumulator[step] = state.sum() / int(
                network_params.inhib_fraction * network_params.n
            )
        case IndexedTraceType(index):
            accumulator[step] = state[index]
