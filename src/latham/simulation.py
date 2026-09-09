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
    ):
        rng = np.random.default_rng(0)
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
            rng,
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
            return (
                (
                    2 * np.pi * r_sample
                )  # Jacobian normalisation (strictly 2\pi isnt necessary)
                * (1 - np.tanh((np.pow(r_sample, 2) - 1) / Delta_r))
                / (np.pi * Delta_r * np.log(1 + np.exp(2 / Delta_r)))
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
        rng: Generator,
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

        connections = rng.random((network_params.N, network_params.N)) < P

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

    def integrate_voltage(self, cell_params: CellParams) -> NDArray[float32]:
        """
        Performs RK4 integration for the membrane potential.

        Returns:
            float: The step in membrane potential across all neurons.
        """

        def potential_step(
            V: np.ndarray,
            conductances: np.ndarray,
            synaptic_currents: np.ndarray,
            I_a: np.ndarray,
            cell_params: CellParams,
        ) -> NDArray[float32]:
            I = synaptic_currents[0]
            I_epsilon = synaptic_currents[1]
            g_K = conductances[0]
            g_K_Ca = conductances[1]
            res = (
                (V - cell_params.V_r)
                * (V - cell_params.V_t)
                / (cell_params.V_t - cell_params.V_r)
            )
            res += I_a
            res += -(g_K + g_K_Ca) * (V - cell_params.epsilon_K)
            res += -(V * I - I_epsilon)
            return res / cell_params.tau_cell

        k1 = potential_step(
            self.V,
            self.conductances,
            self.synaptic_currents,
            self.I_a,
            cell_params,
        )
        k2 = potential_step(
            self.V + k1 * self.time_step / 2,
            self.conductances * self.conductance_decay_half_step,
            self.synaptic_currents * self.synaptic_current_decay_half_step,
            self.I_a,
            cell_params,
        )
        k3 = potential_step(
            self.V + k2 * self.time_step / 2,
            self.conductances * self.conductance_decay_half_step,
            self.synaptic_currents * self.synaptic_current_decay_half_step,
            self.I_a,
            cell_params,
        )
        k4 = potential_step(
            self.V + k3 * self.time_step,
            self.conductances * self.conductance_decay_full_step,
            self.synaptic_currents * self.synaptic_current_decay_full_step,
            self.I_a,
            cell_params,
        )
        return self.time_step / float32(6) * (k1 + 2 * k2 + 2 * k3 + k4)

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

    def process_spikes(self, cell_params, synaptic_params):
        # boolean array of neurons for which a spike has occured
        spikes: NDArray[bool] = self.V > cell_params.V_apex
        if spikes.sum() == 0:
            return spikes

        # reset membrane potentials for those
        self.V = np.where(spikes, cell_params.V_repol, self.V).astype(float32)

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


def run_sim(
    cell_params: CellParams,
    synaptic_params: SynapticParams,
    network_params: NetworkParams,
    time_step: float = 1,  # ms
    total_steps=100 * 1000,  # 100s
):
    state = State(cell_params, synaptic_params, network_params, time_step)
    print("Successfully initialised simulation.")

    spikes_s: NDArray[bool] = np.zeros((total_steps + 1, network_params.N)).astype(bool)
    t = np.arange(0, time_step * total_steps + time_step, time_step)
    for i, _ in tqdm(
        enumerate(t),
        total=total_steps,
    ):
        # RK4 voltage step
        state.V += state.integrate_voltage(cell_params)
        state.update_conductances()
        state.update_synaptic_currents()
        # get spikes
        spikes: NDArray[bool] = state.process_spikes(cell_params, synaptic_params)

        spikes_s[i] = spikes
    return t, spikes_s, state
