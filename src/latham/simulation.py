import numpy as np
from numba import njit
from numba_progress import ProgressBar
from numpy import bool, float32
from numpy.random import Generator
from numpy.typing import NDArray

from latham import sampler
from latham.params import *


class State:
    def __init__(
        self,
        cell_params: CellParams,
        synaptic_params: SynapticParams,
        network_params: NetworkParams,
        time_step: float,
    ):
        rng = np.random.default_rng()
        # initialise voltages at random potentials between V_r and V_t --
        # without this there is no access to the metastable states described in
        # the paper without endogenously active cells
        # self.V: NDArray[float32] = (
        #     np.ones(network_params.N, dtype=float32) * cell_params.V_r
        # )
        self.V: NDArray[float32] = cell_params.V_r + (
            cell_params.V_t - cell_params.V_r
        ) * rng.random(network_params.N, dtype=float32)
        # initialise all conductances at rest (0)
        self.conductances: NDArray[float32] = np.zeros(
            (2, network_params.N), dtype=float32
        )
        # initalise all synaptic currents at rest (0)
        self.synaptic_currents: NDArray[float32] = np.zeros(
            (2, network_params.N), dtype=float32
        )
        # initialise the applied current vector as a boxcar between 0 and I_max
        self.I_a: NDArray[float32] = (cell_params.I_max) * rng.random(
            network_params.N, dtype=float32
        )
        # distribute neurons in space
        self.positions: NDArray[float32] = State._get_neuron_positions(
            rng, network_params
        )

        # initialise a neuron_types array of booleans (False -> Inhibitory, True -> Excitatory)
        N_I = int(
            network_params.inhib_fraction * network_params.N
        )  # number of inhibitory neurons
        self.neuron_types: NDArray[bool] = np.concat(
            [np.zeros(N_I), np.ones(network_params.N - N_I)]
        ).astype(bool)  # excitatory represented by 1, inhibitory by 0

        # initialise an array of reversal potentials based on neuron type
        self.rev_potentials: NDArray[float32] = (
            self.neuron_types * network_params.excit_rev_pot
            + (1 - self.neuron_types) * network_params.inhib_rev_pot
        ).astype(float32)

        # find the connection matrix
        self.connectivity: NDArray[float32] = State._get_connection_matrix(
            N_I,
            self.positions,
            cell_params,
            synaptic_params,
            network_params,
        )

        self.time_step = time_step

        # precompute decay steps
        self.conductance_decay_half_step: NDArray[float32] = np.vstack(
            (
                np.ones(network_params.N) * np.exp(-time_step / 2 / cell_params.tau_K),
                np.ones(network_params.N)
                * np.exp(-time_step / 2 / cell_params.tau_K_Ca),
            ),
            dtype=float32,
        )
        self.conductance_decay_full_step: NDArray[float32] = np.pow(
            self.conductance_decay_half_step, 2, dtype=float32
        )
        self.synaptic_current_decay_half_step: NDArray[float32] = np.ones(
            (2, network_params.N), dtype=float32
        ) * np.exp(-time_step / 2 / synaptic_params.tau_s, dtype=float32)
        self.synaptic_current_decay_full_step: NDArray[float32] = np.pow(
            self.synaptic_current_decay_half_step, 2, dtype=float32
        )

        # precompute update to conductance on spike
        self.spike_conductance_update: NDArray[float32] = np.vstack(
            (
                np.ones(network_params.N) * cell_params.delta_g_K,
                np.ones(network_params.N) * cell_params.delta_g_K_Ca,
            ),
            dtype=float32,
        )

    def _get_neuron_positions(
        rng: Generator, network_params: NetworkParams
    ) -> NDArray[float32]:
        def radial_dist(r_sample, Delta_r):
            return (1 - np.tanh((np.pow(r_sample, 2) - 1) / Delta_r)) / (
                np.pi * Delta_r * np.log(1 + np.exp(2 / Delta_r))
            )

        # distribute uniformly azimuthally
        theta: NDArray[float32] = (
            2 * np.pi * rng.random(network_params.N, dtype=float32)
        )

        # use a rejection sampler to find radial positions in line with the distribution function
        r = sampler.rejection(
            rng,
            lambda r_sample: radial_dist(r_sample, network_params.Delta_r),
            network_params.N,
            0,
            2,  # chosen as the distribution ~ 0 here, could go larger -- don't think its necessary
        )
        return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=float32).T

    def _get_connection_matrix(
        N_I: int,
        positions: NDArray[float32],
        cell_params: CellParams,
        synaptic_params: SynapticParams,
        network_params: NetworkParams,
    ) -> NDArray[float32]:
        def get_connection_weight(
            V_PSP: float,
            rev_pot: float,
            cell_params: CellParams,
            synaptic_params: SynapticParams,
        ):
            return (
                V_PSP
                / (rev_pot - cell_params.V_r)
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

        N_E = network_params.N - N_I

        # create four submatrices corresponding to the probability of
        # connection between any two neurons at infinite range according only
        # to their types and compile them into P
        P_II: NDArray[float32] = (
            network_params.K_I
            * network_params.B_I
            / (N_E + N_I * network_params.B_I)
            * np.ones((N_I, N_I), dtype=float32)
        )
        P_IE: NDArray[float32] = (
            network_params.K_E
            * network_params.B_E
            / (N_E + N_I * network_params.B_E)
            * np.ones((N_I, N_E), dtype=float32)
        )
        P_EI: NDArray[float32] = (
            network_params.K_I
            / (N_E + N_I * network_params.B_I)
            * np.ones((N_E, N_I), dtype=float32)
        )
        P_EE: NDArray[float32] = (
            network_params.K_E
            / (N_E + N_I * network_params.B_E)
            * np.ones((N_E, N_E), dtype=float32)
        )
        P: NDArray[float32] = np.vstack(
            (np.hstack((P_II, P_IE)), np.hstack((P_EI, P_EE)))
        )
        # don't allow autapses
        np.fill_diagonal(P, 0)

        # modulate P by axonal spread distribution
        # first normalise by Z array
        P /= np.hstack(
            (
                np.ones((network_params.N, N_I)) * get_Z(network_params.sigma_I),
                np.ones((network_params.N, N_E)) * get_Z(network_params.sigma_E),
            )
        )
        # find square euclidean norms between neurons
        dists: NDArray[float32] = positions[:, None] - positions[None, :]
        norms: NDArray[float32] = (dists * dists).sum(axis=2)
        variances: NDArray[float32] = np.hstack(
            (
                np.ones((network_params.N, N_I)) * np.pow(network_params.sigma_I, 2),
                np.ones((network_params.N, N_E)) * np.pow(network_params.sigma_E, 2),
            ),
            dtype=float32,
        )

        # decrease P according to axonal spread
        P *= np.exp(-norms / 2 / variances)

        connections = np.random.uniform(0, 1, (network_params.N, network_params.N)) < P

        W_I = get_connection_weight(
            network_params.V_IPSP,
            network_params.inhib_rev_pot,
            cell_params,
            synaptic_params,
        )
        W_E = get_connection_weight(
            network_params.V_EPSP,
            network_params.excit_rev_pot,
            cell_params,
            synaptic_params,
        )
        weights: NDArray[float32] = np.hstack(
            (
                W_I * np.ones((network_params.N, N_I)),
                W_E * np.ones((network_params.N, N_E)),
            ),
            dtype=float32,
        )
        return connections * weights


@njit
def integrate_voltage(
    V: NDArray[float32],
    conductances: NDArray[float32],
    synaptic_currents: NDArray[float32],
    I_a: NDArray[float32],
    time_step: np.float32,
    conductance_decay_half_step: NDArray[float32],
    synaptic_current_decay_half_step: NDArray[float32],
    conductance_decay_full_step: NDArray[float32],
    synaptic_current_decay_full_step: NDArray[float32],
    V_r: float,
    V_t: float,
    epsilon_K: float,
    tau_cell: float,
) -> NDArray[float32]:
    """
    Performs RK4 integration for the membrane potential.

    Returns:
        NDArray[float32]: The step in membrane potential across all neurons.
    """

    def potential_step(
        V: NDArray[float32],
        conductances: NDArray[float32],
        synaptic_currents: NDArray[float32],
        I_a: NDArray[float32],
        V_r: float,
        V_t: float,
        epsilon_K: float,
        tau_cell: float,
    ) -> NDArray[float32]:
        I = synaptic_currents[0]
        I_epsilon = synaptic_currents[1]
        g_K = conductances[0]
        g_K_Ca = conductances[1]
        res: NDArray[float32] = (
            (V - float32(V_r)) * (V - float32(V_t)) / float32(V_t - V_r)
        )
        res += I_a
        res += -(g_K + g_K_Ca) * (V - epsilon_K)
        res += -(V * I - I_epsilon)
        return res / float32(tau_cell)

    k1 = potential_step(
        V,
        conductances,
        synaptic_currents,
        I_a,
        V_r,
        V_t,
        epsilon_K,
        tau_cell,
    )
    k2 = potential_step(
        V + k1 * time_step / 2,
        conductances * conductance_decay_half_step,
        synaptic_currents * synaptic_current_decay_half_step,
        I_a,
        V_r,
        V_t,
        epsilon_K,
        tau_cell,
    )
    k3 = potential_step(
        V + k2 * time_step / 2,
        conductances * conductance_decay_half_step,
        synaptic_currents * synaptic_current_decay_half_step,
        I_a,
        V_r,
        V_t,
        epsilon_K,
        tau_cell,
    )
    k4 = potential_step(
        V + k3 * time_step,
        conductances * conductance_decay_full_step,
        synaptic_currents * synaptic_current_decay_full_step,
        I_a,
        V_r,
        V_t,
        epsilon_K,
        tau_cell,
    )
    return V + time_step / float32(6) * (k1 + 2 * k2 + 2 * k3 + k4)


@njit
def update_conductances(
    conductances: NDArray[float32],
    conductance_decay_full_step: NDArray[float32],
    steps=1,
) -> NDArray[float32]:
    if steps == 1:
        return conductances * conductance_decay_full_step
    else:
        return conductances * np.pow(conductance_decay_full_step, steps)


@njit
def update_synaptic_currents(
    synaptic_currents: NDArray[float32],
    synaptic_current_decay_full_step: NDArray[float32],
    steps=1,
) -> NDArray[float32]:
    if steps == 1:
        return synaptic_currents * synaptic_current_decay_full_step
    else:
        return synaptic_currents * np.pow(synaptic_current_decay_full_step, steps)


@njit
def process_spikes(
    V: NDArray[float32],
    conductances: NDArray[float32],
    synaptic_currents: NDArray[float32],
    spike_conductance_update: NDArray[float32],
    connectivity: NDArray[float32],
    rev_potentials: NDArray[float32],
    V_apex: float,
    V_repol: float,
    r_s: float,
) -> tuple[NDArray[float32], NDArray[float32], NDArray[float32], NDArray[bool]]:
    # boolean array of neurons for which a spike has occured
    spikes: NDArray[bool] = V > V_apex
    if spikes.sum() == 0:
        return V, conductances, synaptic_currents, spikes
    # print(f"Spikes: {spikes.sum()}")

    # reset membrane potentials for those
    V *= (1 - spikes).astype(float32)
    V += (spikes * V_repol).astype(float32)

    # adjust conductances
    conductances += spikes * spike_conductance_update

    # find a reduced connectivity matrix for only js where spike has occured
    spike_connectivity: NDArray[float32] = connectivity[:, spikes]

    # this is effectively dotting the spike_connectivity matrix with the
    # spike vector (W_ij s^j) and doing the same multiplying the spike
    # vector piecewise with the corresponding reversal potentials for the
    # second synaptic current in parallel
    #
    # This is much faster than doing np.dot, as the spiking is sparse
    synaptic_currents += (
        np.vstack(
            (
                np.sum(spike_connectivity, axis=1),
                np.sum(
                    spike_connectivity * rev_potentials[spikes],
                    axis=1,
                ),
            ),
        ).astype(float32)
        * r_s
    )

    return V, conductances, synaptic_currents, spikes


def run_sim(
    cell_params: CellParams,
    synaptic_params: SynapticParams,
    network_params: NetworkParams,
    time_step: float = 1,  # ms
    total_steps=100 * 1000,  # 100s
):
    state = State(cell_params, synaptic_params, network_params, time_step)
    print("Successfully initialised simulation.")

    t = np.arange(0, time_step * total_steps + time_step, time_step)
    spikes_s = None
    with ProgressBar(total=total_steps) as progress:
        spikes_s = _run_sim_loop(
            state.V,
            state.conductances,
            state.synaptic_currents,
            state.I_a,
            time_step,
            state.conductance_decay_half_step,
            state.conductance_decay_full_step,
            state.synaptic_current_decay_half_step,
            state.synaptic_current_decay_full_step,
            state.spike_conductance_update,
            state.connectivity,
            state.rev_potentials,
            cell_params.V_r,
            cell_params.V_t,
            cell_params.V_apex,
            cell_params.V_repol,
            cell_params.epsilon_K,
            cell_params.tau_cell,
            synaptic_params.r_s,
            network_params.N,
            total_steps,
            progress,
        )

    return t, spikes_s, state


@njit
def _run_sim_loop(
    V,
    conductances,
    synaptic_currents,
    I_a,
    time_step,
    conductance_decay_half_step,
    conductance_decay_full_step,
    synaptic_current_decay_half_step,
    synaptic_current_decay_full_step,
    spike_conductance_update,
    connectivity,
    rev_potentials,
    V_r,
    V_t,
    V_apex,
    V_repol,
    epsilon_K,
    tau_cell,
    r_s,
    N,
    total_steps,
    progress_hook,
) -> NDArray[bool]:
    spikes_s: NDArray[bool] = np.zeros((total_steps + 1, N)).astype(bool)
    for i in range(total_steps):
        # RK4 voltage step
        V = integrate_voltage(
            V,
            conductances,
            synaptic_currents,
            I_a,
            np.float32(time_step),
            conductance_decay_half_step,
            synaptic_current_decay_half_step,
            conductance_decay_full_step,
            synaptic_current_decay_full_step,
            V_r,
            V_t,
            epsilon_K,
            tau_cell,
        )
        conductances = update_conductances(conductances, conductance_decay_full_step)
        synaptic_currents = update_synaptic_currents(
            synaptic_currents, synaptic_current_decay_full_step
        )
        # get spikes
        V, conductances, synaptic_currents, spikes = process_spikes(
            V,
            conductances,
            synaptic_currents,
            spike_conductance_update,
            connectivity,
            rev_potentials,
            V_apex,
            V_repol,
            r_s,
        )
        spikes_s[i] = spikes
        progress_hook.update(1)
    return spikes_s
