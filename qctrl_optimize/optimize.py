import numpy as np

def pulse_values(results):
    """Extract array of pulse param values from QCTRL output dict"""
    return np.array([[slice['value'] for slice in opt_var_result] for opt_var_result in results.values()])

def get_ham_infid_func(sigs_to_ham):
    """
    Return a function that calculates the standard Hamiltonian-based infidelity from a target and set of signals
    """
    def infid(graph, target, signals):
        # Create target operator
        target_operator = graph.target(
            target
        )

        hamiltonian = sigs_to_ham(graph, signals)
        
        infidelity = graph.infidelity_pwc(
            hamiltonian=hamiltonian,
            target=target_operator,
            name='infidelity'
        )
        return infidelity
    return infid

def get_vals_to_sigs_pwc(duration, var_shape, amplitude=1):
    """
    Return a function that converts pwc values to Q-CTRL pwc signals
    """
    def values_to_sigs_pwc(graph, values, optimizable=True):
        """converts an (n x m) array of values to n signals of m pwc values each"""
        sigs = []
        vars = []

        if values is None:
            assert(optimizable and var_shape is not None)
            vars = [graph.optimization_variable(
                    var_shape[1], 
                    lower_bound=-1, 
                    upper_bound=1, 
                    initial_values=None, 
                    name=f'ctrl_{i}_var'
                ) 
                for i in range(var_shape[0])
            ]
            sigs = [
                graph.pwc_signal(
                    values=amplitude*vars[i],
                    duration=duration,
                    name=f'ctrl_{i}'
                )
                for i in range(var_shape[0])
            ]
        else:
            for i,vals in enumerate(values):
                if optimizable:
                    var = graph.optimization_variable(
                        len(vals), 
                        lower_bound=-1, 
                        upper_bound=1, 
                        initial_values=vals, 
                        name=f'ctrl_{i}_var'
                    )
                    vars.append(var)
                    sigs.append(graph.pwc_signal(
                        values=amplitude*var,
                        duration=duration,
                        name=f'ctrl_{i}'
                    ))
                else:
                    sigs.append(graph.pwc_signal(
                        values=vals,
                        duration=duration,
                        name=f'ctrl_{i}'
                    ))
                    vars.append(vals)
        return sigs,vars
    return values_to_sigs_pwc

def optimize(qctrl, target, values_to_sigs, infid_func, tik0=0, init_guess=None, seed=None, max_iter=200):
    """
    Optimize a general Q-CTRL problem.

    qctrl:              Qctrl() instance
    target:             target, of whatever form is required by `infid_func`
    values_to_sigs:     function taking array of pulse values (or None) to the corresponding Q-CTRL signal objects
    infid_func:         function (target, signals) => a measure of infidelity (cost) for the problem
    tik0:               Tikhonov regularization coefficient
    init_guess:         None or an array of pulse values to start the optimization from
    seed:               int
    max_iter:           max iterations for each Q-CTRL optimization
    """

    graph = qctrl.create_graph()

    signals,vars = values_to_sigs(graph, init_guess, optimizable=True)

    opt_var_names = [v.name for v in vars]
    sig_names = [s.name for s in signals if s.name not in opt_var_names]
    
    infidelity = infid_func(graph, target, signals)

    if init_guess is None:
        tik_vals = vars
    else:
        tik_vals = [vars[i] - init_guess[i,:] for i,v in enumerate(vars)]

    tikhonov = tik0 * graph.sum([graph.abs(v)**2 for v in tik_vals]) / len(tik_vals)
    tikhonov.name = 'tikhonov'

    cost = infidelity + tikhonov
    cost.name = "cost"

    # Run the optimization and retrieve results.
    optimization_result = qctrl.functions.calculate_optimization(
        cost_node_name="cost", output_node_names=sig_names+opt_var_names+['tikhonov', 'infidelity'], graph=graph,
        max_iteration_count=max_iter, seed=seed, cost_history_scope="ITERATION_VALUES",
        optimization_count=1
    )

    n_iter = len(optimization_result.cost_history.iteration_values[0]) - 1

    print(f"Iterations:\t{n_iter}")
    print(f"Optimized cost:\t{optimization_result.cost:.3e}")
    print(f"Infidelity:\t{optimization_result.output['infidelity']['value']:.3e}")
    print(f"Tikhonov:\t{optimization_result.output['tikhonov']['value']:.3e}")

    drive_results = {k:v for (k,v) in optimization_result.output.items() if k in sig_names}
    opt_var_results = {k:v for (k,v) in optimization_result.output.items() if k in opt_var_names}

    return optimization_result, drive_results, opt_var_results, n_iter