import random
from scipy.spatial import Delaunay
from qctrl_optimize import optimize as opt
from qctrl_optimize import gates
from qctrl_optimize import simulate as sim
from qctrl import Qctrl
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pickle

def neighboring_vertices(simplices, v):
    """find neighboring vertices (i.e. connected by an edge) of a vertex"""

    indptr,indices = simplices.vertex_neighbor_vertices
    return indices[indptr[v]:indptr[v+1]]

def neighbor_avg(target_point, points, pcofs):
    """compute average pulse coefficients between all neighbors of a point"""

    if points.shape[0] == 0 or points.shape[0] < points.shape[1]:
        print('NONE')
        return None

    target_idx_list = (target_point == points).all(axis=1).nonzero()[0]
    if len(target_idx_list) > 0:
        target_idx = target_idx_list[0]
    else:
        points = np.concatenate([points, [target_point]])
        target_idx = points.shape[0] - 1

    if points.shape[0] <= points.shape[1]:
        print('NONE')
        return None
    
    try:
        simplices = Delaunay(points)
        neighbors = neighboring_vertices(simplices, target_idx)
        print(neighbors)
        neighbor_pcofs = pcofs[neighbors]
        return np.mean(neighbor_pcofs, axis=0)
    except:
        return None

def neighbor_tik(target_point, points, pcofs):
    """calculate tikhonov penalty (without tik0 coefficient!) between a pulse and its neighbor_avg target pulse"""

    ref_pcof = neighbor_avg(target_point, points, pcofs)
    target_idx = (target_point == points).all(axis=1).nonzero()[0][0]
    tik_vals = [pcofs[target_idx,v,:] - ref_pcof[v,:] for v in range(ref_pcof.shape[0])]
    tik = np.sum([np.abs(v)**2 for v in tik_vals])
    return tik

def sort_points_by_tik(points, all_opt_var_results):
    """sort points by neighbor_tik penalty (in decreasing order)"""

    points = np.copy(points)
    tiks = [neighbor_tik(ref_pt, points, np.array(all_opt_var_results)) for ref_pt in points]
    p = np.flip(np.argsort(tiks))
    points = points[p,:]
    all_opt_var_results = all_opt_var_results[p]
    
    return points, all_opt_var_results

class Interpolater:
    def __init__(self, qctrl, target_func, tik0, opt_init_guess_func, seed, infid_func, values_to_sigs):
        """
        Create a new Interpolater object.

        Parameters:
        `qctrl`:                  qctrl.Qctrl instance
        `target_func`:            function (param1,param2,...) => target
        `tik0`:                   float
        `opt_init_guess_func`:    function (point, all_points, pulse_results) => init_pulse_values
        `seed`:                   int
        `infid_func`:             function (graph, target, signals) => infidelity_cost_term
        `values_to_sigs`:         function (graph, values, is_optimizable) => list{signal}
        """
        self.qctrl = qctrl
        self.target_func = target_func
        self.tik0 = tik0
        self.opt_init_guess_func = opt_init_guess_func
        self.seed = seed
        self.infid_func = infid_func
        self.values_to_sigs = values_to_sigs
    
    def train(self, train_points, first_run=False, all_opt_var_results=None, max_iter=200):
        """
        Run a single iteration of re-optimization (or an initial optimization of first_run=True) and update all reference pulses.

        Parameters:
        `train_points`:         np.array of reference points in parameter space, shape (d, n) where d is dimension and n is number of points
        `first_run`:            if true, do not use self.opt_init_guess_func and instead start each pulse optimization from a random guess
        `all_opt_var_results`:  previously-optimized pulse variables for use in opt_init_guess_func, shape (n, ...) where n is number of train points
        `max_iter`:             maximum number of iterations allowed for the pulse optimizer
        """

        n_iters = []

        # optimize each reference pulse 
        for i,point in enumerate(train_points):
            print(point)
            target = self.target_func(*point)
            if first_run or self.opt_init_guess_func is None:
                init_guess = None
            else:
                init_guess = self.opt_init_guess_func(point, train_points, all_opt_var_results)

            optimization_result, drive_results, opt_var_results, n_iter = opt.optimize(self.qctrl, target, self.values_to_sigs, self.infid_func, self.tik0, init_guess=init_guess, seed=self.seed, max_iter=max_iter)
            n_iters.append(n_iter)
            pulse_values = np.array([o['value'] for o in opt_var_results.values()])#opt.pulse_values(opt_var_results)
            if i == 0:
                print(1, pulse_values)
            if all_opt_var_results is None:
                all_opt_var_results = np.zeros((train_points.shape[0], pulse_values.shape[0], pulse_values.shape[1]))
            all_opt_var_results[i,:,:] = pulse_values

        return all_opt_var_results, n_iters

    def interpolate(self, train_points, opt_var_results, test_point):
        """
        Return 
        """
        raise Exception('Not implemented')

    def test_interpolations(self, train_points, opt_var_results, test_points):
        """
        Test linear interpolation given a set of optimized pulses at specific points in parameter space.

        Parameters:
        `train_points`:     np.array of reference points in parameter space, shape (d, n) where d is dimension and n is number of points
        `opt_var_results`:  previously-optimized pulse variables, shape (n, ...) where n is number of train points
        `test_points`:      np.array of test points in parameter space, shape (d, m) where d is dimension and m is number of points
        """
        
        pulses = []
        target_list = []
        dim = train_points.shape[1]
        if dim == 1:
            # cannot use simplex approach when dim == 1

            # create interpolation for each test point
            for i,point in enumerate(test_points):
                # find the two adjacent points
                diffs = train_points - point
                idx_smaller = np.where(diffs == np.max(diffs[diffs <= 0]))[0][0]
                idx_larger = np.where(diffs == np.min(diffs[diffs >= 0]))[0][0]

                if idx_smaller == idx_larger:
                    result = opt_var_results[idx_smaller]
                    print(result.shape)
                    pulses.append(opt_var_results[idx_smaller])
                else:
                    val_smaller = train_points[idx_smaller]
                    val_larger = train_points[idx_larger]

                    # construct linear combination of reference pulses
                    weight_smaller = (point - val_smaller)/(val_larger - val_smaller)
                    weight_larger = 1 - weight_smaller
                    result = weight_smaller*opt_var_results[idx_smaller] + weight_larger*opt_var_results[idx_larger]
                    print(result.shape)
                    pulses.append(result)

                # the target operation to compare against
                target = self.target_func(*point)

                target_list.append(target)
                simplices = None
        else:
            # calculate barycentric coords (from https://stackoverflow.com/questions/57863618/how-to-vectorize-calculation-of-barycentric-coordinates-in-python)
            simplices = Delaunay(train_points)
            containing_simplices = simplices.find_simplex(test_points)
            assert(np.sum(containing_simplices == -1) == 0)
            b0 = np.zeros((test_points.shape[0], dim))
            for i in range(test_points.shape[0]):
                b0[i,:] = simplices.transform[containing_simplices[i],:dim].dot((test_points[i] - simplices.transform[containing_simplices[i],dim]).transpose())
            bary_coords = np.c_[b0, 1 - b0.sum(axis=1)]

            # create interpolation for each test point
            for i,point in enumerate(test_points):
                containing_simplex = containing_simplices[i]
                simplex_vertices = simplices.simplices[containing_simplex]
                coords = bary_coords[i]

                # construct linear combination of reference pulses
                result = np.zeros(opt_var_results[0].shape)
                for j,v in enumerate(simplex_vertices):
                    result += opt_var_results[v] * coords[j]

                # the target operation to compare against
                target = self.target_func(*point)
                if i == 0:
                    print(1.1, result)
                pulses.append(result)
                target_list.append(target)
        sim_results, infids, sig_list = sim.simulate_many(self.qctrl, np.array(pulses), target_list, self.values_to_sigs, self.infid_func)
        return infids, simplices, sim_results, sig_list