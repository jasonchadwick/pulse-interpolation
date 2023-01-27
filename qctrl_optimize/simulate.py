from . import hamiltonians
import matplotlib.pyplot as plt
import numpy as np

def simulate_result_from_signals(qctrl, graph, signals, target, times, ham_func='simple_pwc'):
    if ham_func == 'simple_pwc':
        hamiltonian, signals, _,_,_ = hamiltonians.simple_hamiltonian_from_signals(graph, signals)
    elif ham_func == 'single_transmon':
        hamiltonian, signals, _,_,_ = hamiltonians.single_transmon_hamiltonian_from_signals(graph, times, vars=signals)
    else:
        raise Exception(f'Unknown Hamiltonian {ham_func}')

    unitaries = graph.time_evolution_operators_pwc(
        hamiltonian=hamiltonian, sample_times=times, name="unitaries"
    )

    target_operator = graph.target(
        target
    )
    # Create infidelity.
    infidelity = graph.infidelity_pwc(
        hamiltonian=hamiltonian,
        target=target_operator,
        name='infidelity'
    )

    # Run simulation.
    sim_result = qctrl.functions.calculate_graph(
        graph=graph, output_node_names=['unitaries', 'infidelity']
    )

    states = []
    for j in range(target.shape[0]):
        state = np.array([[1 if i==j else 0 for i in range(target.shape[0])]]).T
        state_evolutions = sim_result.output['unitaries']['value'] @ state
        states.append(state_evolutions)

    print(f"Infidelity: \t{(sim_result.output['infidelity']['value']):.3e}")

    return sim_result, states

def simulate_result(qctrl, optimization_result, target, times, ham_func='simple_pwc'):
    duration = times[-1]

    graph_sim = qctrl.create_graph()

    signals = []
    for drive_name in optimization_result.keys():
        _, values, _ = qctrl.utils.pwc_pairs_to_arrays(
            optimization_result[drive_name]
        )

        signals.append(graph_sim.pwc_signal(values=values, duration=duration, name=drive_name))
    return simulate_result_from_signals(qctrl, graph_sim, signals, target, times, ham_func=ham_func)

# simulate from 2d np array of dimensions (num_controls, num_pwc_segments)
def simulate_from_values(qctrl, values, target, times, ham_func='simple_pwc'):
    graph_sim = qctrl.create_graph()
    
    signals = []
    for row in range(values.shape[0]):
        signals.append(graph_sim.pwc_signal(values=values[row,:], duration=times[-1]))

    return simulate_result_from_signals(qctrl, graph_sim, signals, target, times, ham_func=ham_func)

# simulate from 2d np array of dimensions (num_controls, num_pwc_segments)
def simulate_many_from_values(qctrl, values, targets, duration, ham_func):
    graph_sim = qctrl.create_graph()
    
    node_names = []
    for i in range(values.shape[0]):
        signals = []
        for row in range(values.shape[1]):
            signals.append(graph_sim.pwc_signal(values=values[i,row,:], duration=duration))

        hamiltonian = ham_func(graph_sim, signals)

        target_operator = graph_sim.target(
            targets[i]
        )
        # Create infidelity.
        infidelity = graph_sim.infidelity_pwc(
            hamiltonian=hamiltonian,
            target=target_operator,
            name=f'infidelity_{i}'
        )
        node_names.append(infidelity.name)

    # Run simulation.
    sim_result = qctrl.functions.calculate_graph(
        graph=graph_sim, output_node_names=node_names
    )

    infids = []
    for name in node_names:
        infids.append(sim_result.output[name]['value'])
        print(name + f": \t{(sim_result.output[name]['value']):.3e}")

    return sim_result, infids

# simulate from 2d np array of dimensions (num_controls, num_pwc_segments)
def simulate_many(qctrl, values, targets, vals_to_sigs, infid_func):
    graph_sim = qctrl.create_graph()
    
    node_names = []
    for i in range(values.shape[0]):
        signals,_ = vals_to_sigs(graph_sim, values[i], optimizable=False)

        # change signal names to avoid conflicts
        for sig in signals:
            sig.name += f'_{i}'

        infidelity = infid_func(graph_sim, targets[i], signals)
        infidelity.name = f'infidelity_{i}'
        node_names.append(infidelity.name)

    # Run simulation.
    sim_result = qctrl.functions.calculate_graph(
        graph=graph_sim, output_node_names=node_names
    )

    infids = []
    for name in node_names:
        infids.append(sim_result.output[name]['value'])
        print(name + f": \t{(sim_result.output[name]['value']):.3e}")

    return sim_result, infids