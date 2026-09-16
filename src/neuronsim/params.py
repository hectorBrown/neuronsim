import numpy as np


class CellParams:
    """
    Holds parameters associated with individual neurons.

    Attributes:
        tau_cell (float): The relaxation time constant of the neuron membrane (ms).
        v_r (float): The resting potential of the neuron membrane (mV).
        v_t (float): The threshold potential of the neuron membrane (mV).
        v_apex (float): The potential at which the membrane potential is reset to `V_repol`, after a spike has occured (mV).
        v_repol (float): The potential which the membrane potential is reset to, after a spike has occured (mV).
        epsilon_k (float): The potassium reversal potential (mV).
        delta_g_k (float): The change in fast hyperpolarization current, after a spike has occured.
        delta_g_k_ca (float): The change in slow hyperpolarization current, after a spike has occured.
        tau_k (float): The relaxation time constant of the fast hyperpolarization current.
        tau_k_ca (float): The relaxation time constant of the slow hyperpolarization current.
        i_max (float): The maximum value that the random depolarizing current can take on for any given neuron.
    """

    def __repr__(self):
        res = "CellParams {"
        res += f"tau_cell: {self.tau_cell}, "
        res += f"v_r: {self.v_r}, "
        res += f"v_t: {self.v_t}, "
        res += f"v_apex: {self.v_apex}, "
        res += f"v_repol: {self.v_repol}, "
        res += f"epsilon_k: {self.epsilon_k}, "
        res += f"delta_g_k: {self.delta_g_k}, "
        res += f"delta_g_k_ca: {self.delta_g_k_ca}, "
        res += f"tau_k: {self.tau_k}, "
        res += f"tau_k_ca: {self.tau_k_ca}, "
        res += f"i_max: {self.i_max}" + "}"
        return res

    tau_cell: float = 10  # ms
    v_r: float = -65  # mV
    v_t: float = -50  # mV
    v_apex: float = 20  # mV
    v_repol: float = -80  # mV
    epsilon_k: float = -80
    delta_g_k: float = 1
    tau_k: float = 30  # ms
    tau_k_ca: float = 2000  # ms

    def __init__(self, i_max: float, delta_g_k_ca: float):
        self.delta_g_k_ca = delta_g_k_ca
        self.i_max = i_max


class SynapticParams:
    """
    Holds parameters associated with synapses between neurons.

    Attributes:
        r_s (float): The fraction of closed channels that open each time a presynaptic neuron fires.
        tau_s (float): The relaxation time constant of the fraction of open channels on a synapse (ms).
    """

    def __repr__(self):
        res = "SynapticParams {"
        res += f"r_s: {self.r_s}, "
        res += f"tau_s: {self.tau_s}" + "}"
        return res

    r_s = 0.1
    tau_s = 3  # ms


class ClassicNetworkParams:
    """
    Holds parameters associated with the neural network.

    Attributes:
        n (int): The total number of neurons in the network.
        delta_r (float): Half the radial distance over which the distribution of neurons drops almost to 0.
        excit_rev_pot (float): The exictatory reversal potential at synapse (mV).
        inhib_rev_pot (float): The inhibitory reversal potential at synapse (mV).
        inhib_fraction (float): The fraction of neurons which are inhibitory.
        k_e (int): The mean number of postsynaptic neurons an excitatory neuron connects to.
        k_i (int): The mean number of postsynaptic neurons an inhibitory neuron connects to.
        b_e (float): The connectivity bias for excitatory neurons (>1 -> towards inhibitory neurons).
        b_i (float): The connectivity bias for inhibitory neurons (>1 -> towards inhibitory neurons).
        sigma_e (float): The axonal spread of excitatory neurons (as a fraction of cortical radius).
        sigma_i (float): The axonal spread of inhibitory neurons (as a fraction of cortical radius).
        v_epsp (float): The excitatory postsynaptic potential (mV).
        v_ipsp (float): The inhibitory postsynaptic potential (mV).
    """

    def __repr__(self):
        res = "ClassicNetworkParams {"
        res += f"n: {self.n}, "
        res += f"delta_r: {self.delta_r}, "
        res += f"excit_rev_pot: {self.excit_rev_pot}, "
        res += f"inhib_rev_pot: {self.inhib_rev_pot}, "
        res += f"b_e: {self.b_e}, "
        res += f"b_i: {self.b_i}, "
        res += f"inhib_fraction: {self.inhib_fraction}, "
        res += f"k_e: {self.k_e}, "
        res += f"k_i: {self.k_i}, "
        res += f"sigma_e: {self.sigma_e}, "
        res += f"sigma_i: {self.sigma_i}, "
        res += f"v_epsp: {self.v_epsp}, "
        res += f"v_ipsp: {self.v_ipsp}" + "}"
        return res

    n: int = 10000
    delta_r: float = 0.1
    excit_rev_pot: float = 0  # mV
    inhib_rev_pot = -80  # mV

    def __init__(
        self,
        b_e: float,
        b_i: float,
        inhib_fraction: float,
        k_e: int,
        k_i: int,
        sigma_e: float,
        sigma_i: float,
        v_epsp: float,
        v_ipsp: float,
    ):
        self.inhib_fraction = inhib_fraction
        self.k_e = k_e
        self.k_i = k_i
        self.b_e = b_e
        self.b_i = b_i
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.v_epsp = v_epsp
        self.v_ipsp = v_ipsp


class WattsStrogatzNetworkParams:
    """
    Holds parameters associated with the neural network, when connectivity is
    determined by the Watts-Strogatz network.

    Attributes:
        n (int): The total number of neurons in the network.
        excit_rev_pot (float): The exictatory reversal potential at synapse (mV).
        inhib_rev_pot (float): The inhibitory reversal potential at synapse (mV).
        inhib_fraction (float): The fraction of neurons which are inhibitory.
        k (int): The mean number of postsynaptic neurons a neuron connects to.
        beta (int): The rewiring probability parameter for the Watts-Strogatz network.
        v_epsp (float): The excitatory postsynaptic potential (mV).
        v_ipsp (float): The inhibitory postsynaptic potential (mV).
    """

    def __repr__(self):
        res = "WattsStrogatzNetworkParams {"
        res += f"n: {self.n}, "
        res += f"excit_rev_pot: {self.excit_rev_pot}, "
        res += f"inhib_rev_pot: {self.inhib_rev_pot}, "
        res += f"inhib_fraction: {self.inhib_fraction}, "
        res += f"k: {self.k}, "
        res += f"beta: {self.beta}, "
        res += f"v_epsp: {self.v_epsp}, "
        res += f"v_ipsp: {self.v_ipsp}" + "}"
        return res

    n: int = 10000
    excit_rev_pot: float = 0  # mV
    inhib_rev_pot = -80  # mV

    def __init__(
        self,
        inhib_fraction: float,
        k: int,
        beta: float,
        v_epsp: float,
        v_ipsp: float,
    ):
        self.inhib_fraction = inhib_fraction
        self.k = k
        self.beta = beta
        self.v_epsp = v_epsp
        self.v_ipsp = v_ipsp


type NetworkParams = WattsStrogatzNetworkParams | ClassicNetworkParams


def network_A(
    i_max: float, b_e: float, b_i: float, delta_g_k_ca: float
) -> tuple[CellParams, SynapticParams, ClassicNetworkParams]:
    """
    Creates parameters for a network of type A as specified in Latham et al. (2000).

    Args:
        i_max (float): The maximum value that the random depolarizing current can take on for any given neuron.
        b_e (float): The connectivity bias for excitatory neurons (>1 -> towards inhibitory neurons).
        b_i (float): The connectivity bias for inhibitory neurons (>1 -> towards inhibitory neurons).
        delta_g_k_ca (float): The change in slow hyperpolarization current, after a spike has occured.

    Returns:
        CellParams: The cell parameters.
        SynapticParams: The synaptic parameters.
        ClassicNetworkParams: The network parameters.
    """
    return (
        CellParams(i_max, delta_g_k_ca),
        SynapticParams(),
        ClassicNetworkParams(
            b_e=b_e,
            b_i=b_i,
            k_e=1000,
            k_i=1000,
            inhib_fraction=0.2,
            sigma_e=np.inf,
            sigma_i=np.inf,
            v_epsp=1,
            v_ipsp=-1.5,
        ),
    )


def network_B(
    i_max: float, b_e: float, b_i: float, delta_g_k_ca: float
) -> tuple[CellParams, SynapticParams, ClassicNetworkParams]:
    """
    Creates parameters for a network of type B as specified in Latham et al. (2000).

    Args:
        i_max (float): The maximum value that the random depolarizing current can take on for any given neuron.
        b_e (float): The connectivity bias for excitatory neurons (>1 -> towards inhibitory neurons).
        b_i (float): The connectivity bias for inhibitory neurons (>1 -> towards inhibitory neurons).
        delta_g_k_ca (float): The change in slow hyperpolarization current, after a spike has occured.

    Returns:
        CellParams: The cell parameters.
        SynapticParams: The synaptic parameters.
        ClassicNetworkParams: The network parameters.
    """
    return (
        CellParams(i_max, delta_g_k_ca),
        SynapticParams(),
        ClassicNetworkParams(
            b_e=b_e,
            b_i=b_i,
            k_e=200,
            k_i=200,
            inhib_fraction=0.3,
            sigma_e=0.12,
            sigma_i=0.12,
            v_epsp=4,
            v_ipsp=-6,
        ),
    )


def network_C(
    i_max: float,
    k: int,
    beta: float,
    delta_g_k_ca: float,
    inhib_fraction: float,
    v_epsp: float,
    v_ipsp: float,
) -> tuple[CellParams, SynapticParams, WattsStrogatzNetworkParams]:
    """
    Creates parameters for a network of type C (Watts-Strogatz connectivity).

    Args:
        i_max (float): The maximum value that the random depolarizing current can take on for any given neuron.
        k (int): The number of postsynaptic neurons each presynaptic neuron connects to.
        beta (float): The Watts-Strogatz rewiring probability parameter.
        delta_g_k_ca (float): The change in slow hyperpolarization current, after a spike has occured.
        inhib_fraction (float): The fraction of neurons which are inhibitory.
        v_epsp (float): The excitatory postsynaptic potential.
        v_ipsp (float): The inhibitory postsynaptic potential.

    Returns:
        CellParams: The cell parameters.
        SynapticParams: The synaptic parameters.
        NetworkParams: The network parameters.
    """
    return (
        CellParams(i_max, delta_g_k_ca),
        SynapticParams(),
        WattsStrogatzNetworkParams(
            k=k,
            beta=beta,
            inhib_fraction=inhib_fraction,
            v_epsp=v_epsp,
            v_ipsp=v_ipsp,
        ),
    )
