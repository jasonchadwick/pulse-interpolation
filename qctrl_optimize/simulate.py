from . import hamiltonians
import matplotlib.pyplot as plt
import numpy as np

def simulate_many(qctrl, values, targets, vals_to_sigs, infid_func):
    """
    Simulate a set of pulses.

    Parameters:
    `qctrl`:            Qctrl() instance
    `values`:           array of pulse values of shape (n, ...) where n is number of distinct simulation problems
    `targets`:          list of targets of length n
    `vals_to_sigs`:     function taking array of pulse values (or None) to the corresponding Q-CTRL signal objects
    `infid_func`:         function (target, signals) => a measure of infidelity (cost) for the problem
    """
    graph_sim = qctrl.create_graph()
    
    node_names = []
    sig_list = []
    for i in range(values.shape[0]):
        signals,_ = vals_to_sigs(graph_sim, values[i], optimizable=False)

        # change signal names to avoid conflicts
        for sig in signals:
            sig.name += f'_{i}'
        infidelity = infid_func(graph_sim, targets[i], signals)
        infidelity.name = f'infidelity_{i}'
        node_names.append(infidelity.name)
        sig_list.append(signals)

    # Run simulation.
    sim_result = qctrl.functions.calculate_graph(
        graph=graph_sim, output_node_names=node_names
    )

    infids = []
    for name in node_names:
        infids.append(sim_result.output[name]['value'])
        print(name + f": \t{(sim_result.output[name]['value']):.3e}")

    return sim_result, infids, sig_list