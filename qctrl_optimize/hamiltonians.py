import numpy as np

# number of signals in various Hamiltonians
SIMPLE_HAM_N_SIG = 5
TRANSMON_QUBIT_N_SIG = 2

def simple_hamiltonian_from_signals(graph, signals, vars=[]):
    """
    Create the Hamiltonian used in Sauvage and Mintert, PRL 129 050507 2022, from a list of Q-CTRL signals.
    List must be of length SIMPLE_HAM_N_SIG.
    """

    assert(len(signals) == SIMPLE_HAM_N_SIG)
    si = graph.pauli_matrix('I')
    sx = graph.pauli_matrix('X')
    sy = graph.pauli_matrix('Y')
    sz = graph.pauli_matrix('Z')

    drives = None

    for s, op in zip(signals, [graph.kron(sx, sx), graph.kron(sy, si), graph.kron(sz, si), graph.kron(si, sy), graph.kron(si, sz)]):
        if drives is None:
            drives = graph.hermitian_part(s * op)
        else:
            drives += graph.hermitian_part(s * op)

    hamiltonian = drives

    return hamiltonian, signals, vars

def simple_hamiltonian_from_signals_1q(graph, signals, vars=[]):
    """
    Create the Hamiltonian used in Sauvage and Mintert, PRL 129 050507 2022, from a list of Q-CTRL signals.
    List must be of length SIMPLE_HAM_N_SIG.
    """

    sx = graph.pauli_matrix('X')
    sy = graph.pauli_matrix('Y')

    drives = None

    for s, op in zip(signals, [sx, sy]):
        if drives is None:
            drives = graph.hermitian_part(s * op)
        else:
            drives += graph.hermitian_part(s * op)

    hamiltonian = drives

    return hamiltonian, signals, vars

def single_transmon_hamiltonian_from_signals(graph, times, signals=[], vars=[], d=2, w=0.1, xi=-0.33e-3):
    """
    Create the Hamiltonian for a single transmon, from a list of Q-CTRL signals.
    List must be of length SIMPLE_HAM_N_SIG.

    times: sample times
    signals: list of Q-CTRL signals of length TRANSMON_QUBIT_N_SIG
    """

    drive_names = []

    a = graph.annihilation_operator(d)
    adag = graph.creation_operator(d)
    
    H = 2*np.pi*(w*adag@a + xi/2*adag@adag@a@a)

    if len(signals) == 0:
        sin_omega = graph.pwc_signal(
            values=np.sin(w*times), duration=times[-1]
        )
    
        p = sin_omega*vars[0]
        q = sin_omega*vars[1]
        signals = [p,q]
    else:
        p = signals[0]
        q = signals[1]
    H += p*(a+adag) + 1j*q*(a-adag)

    opt_var_names = [v.name for v in vars]
    drive_names = [n for n in drive_names]
    return H, signals, vars