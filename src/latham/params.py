import numpy as np


class CellParams:
    """
    Holds parameters associated with individual neurons.

    Attributes:
        tau_cell (float): The relaxation time constant of the neuron membrane (ms).
        V_r (float): The resting potential of the neuron membrane (mV).
        V_t (float): The threshold potential of the neuron membrane (mV).
        V_apex (float): The potential at which the membrane potential is reset to `V_repol`, after a spike has occured (mV).
        V_repol (float): The potential which the membrane potential is reset to, after a spike has occured (mV).
        epsilon_K (float): The potassium reversal potential (mV).
        delta_g_K (float): The change in fast hyperpolarization current, after a spike has occured.
        delta_g_K_Ca (float): The change in slow hyperpolarization current, after a spike has occured.
        tau_K (float): The relaxation time constant of the fast hyperpolarization current.
        tau_K_Ca (float): The relaxation time constant of the slow hyperpolarization current.
        I_max (float): The maximum value that the random depolarizing current can take on for any given neuron.
    """

    def __repr__(self):
        res = "CellParams {"
        res += f"tau_cell: {self.tau_cell}, "
        res += f"V_r: {self.V_r}, "
        res += f"V_t: {self.V_t}, "
        res += f"V_apex: {self.V_apex}, "
        res += f"V_repol: {self.V_repol}, "
        res += f"epsilon_K: {self.epsilon_K}, "
        res += f"delta_g_K: {self.delta_g_K}, "
        res += f"delta_g_K_Ca: {self.delta_g_K_Ca}, "
        res += f"tau_K: {self.tau_K}, "
        res += f"tau_K_Ca: {self.tau_K_Ca}, "
        res += f"I_max: {self.I_max}" + "}"
        return res

    tau_cell: float = 10  # ms
    V_r: float = -65  # mV
    V_t: float = -50  # mV
    V_apex: float = 20  # mV
    V_repol: float = -80  # mV
    epsilon_K: float = -80
    delta_g_K: float = 1
    tau_K: float = 30  # ms
    tau_K_Ca: float = 2000  # ms

    def __init__(self, I_max: float, delta_g_K_Ca: float):
        self.delta_g_K_Ca = delta_g_K_Ca
        self.I_max = I_max


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


class NetworkParams:
    """
    Holds parameters associated with the neural network.

    Attributes:
        N (int): The total number of neurons in the network.
        Delta_r (float): Half the radial distance over which the distribution of neurons drops almost to 0.
        excit_rev_pot (float): The exictatory reversal potential at synapse (mV).
        inhib_rev_pot (float): The inhibitory reversal potential at synapse (mV).
        inhib_fraction (float): The fraction of neurons which are inhibitory.
        K_E (int): The mean number of postsynaptic neurons an excitatory neuron connects to.
        K_I (int): The mean number of postsynaptic neurons an inhibitory neuron connects to.
        B_E (float): The connectivity bias for excitatory neurons (>1 -> towards inhibitory neurons).
        B_I (float): The connectivity bias for inhibitory neurons (>1 -> towards inhibitory neurons).
        sigma_E (float): The axonal spread of excitatory neurons (as a fraction of cortical radius).
        sigma_I (float): The axonal spread of inhibitory neurons (as a fraction of cortical radius).
        V_EPSP (float): The excitatory postsynaptic potential (mV).
        V_IPSP (float): The inhibitory postsynaptic potential (mV).
    """

    def __repr__(self):
        res = "NetworkParams {"
        res += f"N: {self.N}, "
        res += f"Delta_r: {self.Delta_r}, "
        res += f"excit_rev_pot: {self.excit_rev_pot}, "
        res += f"inhib_rev_pot: {self.inhib_rev_pot}, "
        res += f"B_E: {self.B_E}, "
        res += f"B_I: {self.B_I}, "
        res += f"inhib_fraction: {self.inhib_fraction}, "
        res += f"K_E: {self.K_E}, "
        res += f"K_I: {self.K_I}, "
        res += f"sigma_E: {self.sigma_E}, "
        res += f"sigma_I: {self.sigma_I}, "
        res += f"V_EPSP: {self.V_EPSP}, "
        res += f"V_IPSP: {self.V_IPSP}" + "}"
        return res

    N: int = 10000
    Delta_r: float = 0.1
    excit_rev_pot: float = 0  # mV
    inhib_rev_pot = -80  # mV

    def __init__(
        self,
        B_E: float,
        B_I: float,
        inhib_fraction: float,
        K_E: int,
        K_I: int,
        sigma_E: float,
        sigma_I: float,
        V_EPSP: float,
        V_IPSP: float,
    ):
        self.inhib_fraction = inhib_fraction
        self.K_E = K_E
        self.K_I = K_I
        self.B_E = B_E
        self.B_I = B_I
        self.sigma_E = sigma_E
        self.sigma_I = sigma_I
        self.V_EPSP = V_EPSP
        self.V_IPSP = V_IPSP


def network_A(
    I_max: float, B_E: float, B_I: float, delta_g_K_Ca: float
) -> tuple[CellParams, SynapticParams, NetworkParams]:
    """
    Creates parameters for a network of type A as specified in Latham et al. (2000).

    Args:
        I_max (float): The maximum value that the random depolarizing current can take on for any given neuron.
        B_E (float): The connectivity bias for excitatory neurons (>1 -> towards inhibitory neurons).
        B_I (float): The connectivity bias for inhibitory neurons (>1 -> towards inhibitory neurons).
        delta_g_K_Ca (float): The change in slow hyperpolarization current, after a spike has occured.

    Returns:
        CellParams: The cell parameters.
        SynapticParams: The synaptic parameters.
        NetworkParams: The network parameters.
    """
    return (
        CellParams(I_max, delta_g_K_Ca),
        SynapticParams(),
        NetworkParams(
            B_E=B_E,
            B_I=B_I,
            K_E=1000,
            K_I=1000,
            inhib_fraction=0.2,
            sigma_E=np.inf,
            sigma_I=np.inf,
            V_EPSP=1,
            V_IPSP=-1.5,
        ),
    )


def network_B(
    I_max: float, B_E: float, B_I: float, delta_g_K_Ca: float
) -> tuple[CellParams, SynapticParams, NetworkParams]:
    """
    Creates parameters for a network of type B as specified in Latham et al. (2000).

    Args:
        I_max (float): The maximum value that the random depolarizing current can take on for any given neuron.
        B_E (float): The connectivity bias for excitatory neurons (>1 -> towards inhibitory neurons).
        B_I (float): The connectivity bias for inhibitory neurons (>1 -> towards inhibitory neurons).
        delta_g_K_Ca (float): The change in slow hyperpolarization current, after a spike has occured.

    Returns:
        CellParams: The cell parameters.
        SynapticParams: The synaptic parameters.
        NetworkParams: The network parameters.
    """
    return (
        CellParams(I_max, delta_g_K_Ca),
        SynapticParams(),
        NetworkParams(
            B_E=B_E,
            B_I=B_I,
            K_E=200,
            K_I=200,
            inhib_fraction=0.3,
            sigma_E=0.12,
            sigma_I=0.12,
            V_EPSP=4,
            V_IPSP=-6,
        ),
    )
