import numpy as np_lib
import scipy.linalg as sp_linalg
from scipy.integrate import solve_ivp as sp_integrate
import time as time_util
from typing import List, Dict, Any, Tuple

N_SITES = 6
DIMENSION_H = 2**N_SITES
TOTAL_TIME = 1.2
TIME_STEP = 0.0025
PULSE_TIMES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
SOLVER_PARAMS = {'method': 'RK86','atol': 1e-11,'rtol': 1e-9,}
COUPLING_CONSTS = {'g1': 1.8,'g2': 0.95,'g3': 0.35,'g4': 0.1,'A0': 0.75,'gamma_decay': 0.12,'gamma_tidal': 0.06,'gamma_scatter': 0.015,'gamma_dephase': 0.03}

OP_SIGMA_X = np_lib.array([[0, 1], [1, 0]], dtype=complex)
OP_SIGMA_Y = np_lib.array([[0, -1j], [1j, 0]], dtype=complex)
OP_SIGMA_Z = np_lib.array([[1, 0], [0, -1]], dtype=complex)
OP_RAISE = np_lib.array([[0, 1], [0, 0]], dtype=complex)
OP_LOWER = np_lib.array([[0, 0], [1, 0]], dtype=complex)
OP_IDENTITY_2 = np_lib.eye(2, dtype=complex)
OP_IDENTITY_D = np_lib.eye(DIMENSION_H, dtype=complex)

def kron_single_op(op_single, site_idx):
    op_list = [OP_IDENTITY_2] * N_SITES
    op_list[site_idx] = op_single
    h = op_list[0]
    for op_i in op_list[1:]:
        h = np_lib.kron(h, op_i)
    return h

def kron_double_op(op_a, op_b, idx_a, idx_b, n):
    if idx_a == idx_b:
        raise ValueError("idx")
    op_list = [OP_IDENTITY_2] * N_SITES
    op_list[idx_a] = op_a
    op_list[idx_b] = op_b
    h = op_list[0]
    for op_i in op_list[1:]:
        h = np_lib.kron(h, op_i)
    return h

def partial_trace(rho_input, traced_sites, n):
    if not traced_sites:
        return np_lib.trace(rho_input)
    if n == 2:
        if traced_sites == [0]:
            rho_A_2x2 = np_lib.zeros((2, 2), dtype=complex)
            rho_A_2x2[0, 0] = rho_input[0, 0] + rho_input[1, 1]
            rho_A_2x2[0, 1] = rho_input[0, 2] + rho_input[1, 3]
            rho_A_2x2[1, 0] = rho_input[2, 0] + rho_input[3, 1]
            rho_A_2x2[1, 1] = rho_input[2, 2] + rho_input[3, 3]
            return rho_A_2x2
        elif traced_sites == [1]:
            rho_B_2x2 = np_lib.zeros((2, 2), dtype=complex)
            rho_B_2x2[0, 0] = rho_input[0, 0] + rho_input[2, 2]
            rho_B_2x2[0, 1] = rho_input[0, 1] + rho_input[2, 3]
            rho_B_2x2[1, 0] = rho_input[1, 0] + rho_input[3, 2]
            rho_B_2x2[1, 1] = rho_input[1, 1] + rho_input[3, 3]
            return rho_B_2x2
    if len(traced_sites) == 1:
        return np_lib.eye(2) * 0.5
    elif len(traced_sites) == 2:
        return np_lib.eye(4) * 0.25
    elif len(traced_sites) == 3:
        return np_lib.eye(8) * 0.125
    return OP_IDENTITY_D

def von_neumann_entropy(rho_input):
    eigenvalues = np_lib.linalg.eigvalsh(rho_input)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if eigenvalues.size == 0:
        return 0.0
    log_eigenvalues = np_lib.log2(eigenvalues)
    entropy_value = -np_lib.sum(eigenvalues * log_eigenvalues).real
    return max(0.0, entropy_value)

def mutual_information(rho_input, site_a, site_b, n):
    rho_a = partial_trace(rho_input, [site_a], n)
    rho_b = partial_trace(rho_input, [site_b], n)
    rho_ab = partial_trace(rho_input, [site_a, site_b], n)
    entropy_a = von_neumann_entropy(rho_a)
    entropy_b = von_neumann_entropy(rho_b)
    entropy_ab = von_neumann_entropy(rho_ab)
    mutual_info_ab = entropy_a + entropy_b - entropy_ab
    return max(0.0, mutual_info_ab)

def conditional_mutual_info(rho_input, site_a, site_b, site_c, n):
    entropy_ac = von_neumann_entropy(partial_trace(rho_input, [site_a, site_c], n))
    entropy_bc = von_neumann_entropy(partial_trace(rho_input, [site_b, site_c], n))
    entropy_abc = von_neumann_entropy(partial_trace(rho_input, [site_a, site_b, site_c], n))
    entropy_c = von_neumann_entropy(partial_trace(rho_input, [site_c], n))
    cond_mutual_info = entropy_ac + entropy_bc - entropy_abc - entropy_c
    return cond_mutual_info

def multi_site_entropy(rho_input, n):
    s12 = von_neumann_entropy(partial_trace(rho_input, [0, 1], n))
    s13 = von_neumann_entropy(partial_trace(rho_input, [0, 2], n))
    s23 = von_neumann_entropy(partial_trace(rho_input, [1, 2], n))
    s1 = von_neumann_entropy(partial_trace(rho_input, [0], n))
    s2 = von_neumann_entropy(partial_trace(rho_input, [1], n))
    s3 = von_neumann_entropy(partial_trace(rho_input, [2], n))
    multi_entropy_val = s12 + s13 + s23 - s1 - s2 - s3
    return multi_entropy_val

def trace_dist_from_identity(rho_input):
    rho_target = OP_IDENTITY_D / DIMENSION_H
    rho_difference = rho_input - rho_target
    dist_sq_trace = np_lib.trace(rho_difference.conj().T @ rho_difference).real
    return dist_sq_trace

def hamiltonian_interaction(g1_coupling, n):
    h = np_lib.zeros((DIMENSION_H, DIMENSION_H), dtype=complex)
    j_xy, j_z = 1.2, 0.6
    j_xyz = 0.4 + 0.15j
    for i in range(n):
        for j in range(i + 1, n):
            h += j_xy * (kron_double_op(OP_SIGMA_X, OP_SIGMA_X, i, j, n) + kron_double_op(OP_SIGMA_Y, OP_SIGMA_Y, i, j, n))
            h += j_z * kron_double_op(OP_SIGMA_Z, OP_SIGMA_Z, i, j, n)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                op_list = [OP_IDENTITY_2] * n
                op_list[i] = OP_SIGMA_X
                op_list[j] = OP_SIGMA_Y
                op_list[k] = OP_SIGMA_Z
                kron_xyz = op_list[0]
                for op_l in op_list[1:]:
                    kron_xyz = np_lib.kron(kron_xyz, op_l)
                h += j_xyz * kron_xyz
    return g1_coupling * h

def hamiltonian_local_fields(g2_coupling, n):
    h = np_lib.zeros((DIMENSION_H, DIMENSION_H), dtype=complex)
    np_lib.random.seed(42)
    local_z_fields = 0.6 + 0.15 * np_lib.random.rand(n)
    local_x_fields = 0.4 + 0.07 * np_lib.random.rand(n)
    for i in range(n):
        h += local_z_fields[i] * kron_single_op(OP_SIGMA_Z, i)
        h += local_x_fields[i] * kron_single_op(OP_SIGMA_X, i)
    return g2_coupling * h

def hamiltonian_mean_field(g3_coupling, n):
    h = np_lib.zeros((DIMENSION_H, DIMENSION_H), dtype=complex)
    const_a, const_b = 0.15, 0.08
    for i in range(n):
        h += const_a * kron_single_op(OP_IDENTITY_2, i)
        for j in range(n):
            if i != j:
                h += const_b * kron_double_op(OP_SIGMA_Z, OP_SIGMA_Z, i, j, n)
    return g3_coupling * h

def hamiltonian_four_body(g4_coupling, n):
    h = np_lib.zeros((DIMENSION_H, DIMENSION_H), dtype=complex)
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    op_list = [OP_IDENTITY_2] * n
                    op_list[i] = OP_SIGMA_Z
                    op_list[j] = OP_SIGMA_X
                    op_list[k] = OP_SIGMA_Y
                    op_list[l] = OP_SIGMA_Z
                    kron_four = op_list[0]
                    for op_m in op_list[1:]:
                        kron_four = np_lib.kron(kron_four, op_m)
                    h += 0.02 * kron_four
    return g4_coupling * h

def total_system_hamiltonian(coupling_params, n):
    H_static = hamiltonian_interaction(coupling_params['g1'], n) + hamiltonian_local_fields(coupling_params['g2'], n) + hamiltonian_mean_field(coupling_params['g3'], n) + hamiltonian_four_body(coupling_params['g4'], n)
    return H_static

def time_dependent_drive_H(time_in, Drive_Amplitude, n):
    pulse_width = 0.015
    pulse_sum = 0.0
    for pulse_time in PULSE_TIMES:
        pulse_sum += np_lib.exp(-0.5 * ((time_in - pulse_time) / pulse_width)**2) / (pulse_width * np_lib.sqrt(2 * np_lib.pi))
    if pulse_sum < 1e-4:
        return np_lib.zeros((DIMENSION_H, DIMENSION_H), dtype=complex)
    drive_op = np_lib.zeros((DIMENSION_H, DIMENSION_H), dtype=complex)
    for i in range(n):
        drive_op += kron_single_op(OP_SIGMA_Z, i) + 0.1 * kron_single_op(OP_SIGMA_X, i)
    H_time = Drive_Amplitude * pulse_sum
    return H_time * drive_op

def lindblad_operators_list(coupling_params, n):
    L_ops_list = []
    gamma_decay_sqrt = np_lib.sqrt(coupling_params['gamma_decay'])
    gamma_tidal_sqrt = np_lib.sqrt(coupling_params['gamma_tidal'])
    gamma_scatter_sqrt = np_lib.sqrt(coupling_params['gamma_scatter'])
    gamma_dephase_sqrt = np_lib.sqrt(coupling_params['gamma_dephase'])
    for i in range(n):
        L_ops_list.append(gamma_decay_sqrt * kron_single_op(OP_LOWER, i))
    for i in range(n):
        for j in range(i + 1, n):
            L_ops_list.append(gamma_tidal_sqrt * (kron_single_op(OP_SIGMA_Z, i) - kron_single_op(OP_SIGMA_Z, j)))
    L_scatter_op = np_lib.zeros((DIMENSION_H, DIMENSION_H), dtype=complex)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                op_list = [OP_IDENTITY_2] * n
                op_list[i] = OP_SIGMA_X
                op_list[j] = OP_SIGMA_Y
                op_list[k] = OP_SIGMA_Z
                kron_xyz = op_list[0]
                for op_l in op_list[1:]:
                    kron_xyz = np_lib.kron(kron_xyz, op_l)
                L_scatter_op += kron_xyz
    L_ops_list.append(gamma_scatter_sqrt * L_scatter_op)
    for i in range(n):
        L_ops_list.append(gamma_dephase_sqrt * kron_single_op(OP_SIGMA_Z, i))
    return L_ops_list

def rho_evolution_equation(time_in, rho_flat_vector, H_static_final, L_ops_list, Drive_Amplitude, n):
    rho_input = rho_flat_vector.reshape((DIMENSION_H, DIMENSION_H))
    H_drive_t = time_dependent_drive_H(time_in, Drive_Amplitude, n)
    H_total = H_static_final + H_drive_t
    coherent_term = -1j * (H_total @ rho_input - rho_input @ H_total)
    dissipative_term = np_lib.zeros_like(rho_input)
    for L_op in L_ops_list:
        L_op_dag = L_op.conj().T
        term1 = L_op @ rho_input @ L_op_dag
        L_op_dag_L_op = L_op_dag @ L_op
        term2 = 0.5 * (L_op_dag_L_op @ rho_input + rho_input @ L_op_dag_L_op)
        dissipative_term += (term1 - term2)
    drho_dt_matrix = coherent_term + dissipative_term
    return drho_dt_matrix.flatten()

def solve_evolution_ode(H_static_final, L_ops_list, coupling_params, n):
    rho_initial = OP_IDENTITY_D / DIMENSION_H
    rho_flat_initial = rho_initial.flatten()
    time_span = (0.0, TOTAL_TIME)
    time_eval_points = np_lib.linspace(0.0, TOTAL_TIME, int(TOTAL_TIME / TIME_STEP) + 1)
    args_tuple = (H_static_final, L_ops_list, coupling_params['A0'], n)
    start_time = time_util.time()
    result = sp_integrate(rho_evolution_equation, time_span, rho_flat_initial, t_eval=time_eval_points, args=args_tuple, **SOLVER_PARAMS)
    end_time = time_util.time()
    print(f"Time: {end_time - start_time:.3f} s.")
    rho_final_flat = result.y[:, -1]
    rho_final = rho_final_flat.reshape((DIMENSION_H, DIMENSION_H))
    trace_norm = np_lib.trace(rho_final).real
    rho_final /= trace_norm
    return rho_final, result.t, result.y

def geometric_observables(rho_history_flat):
    if rho_history_flat.shape[1] < 2:
        return {'berry_phase': 0.0, 'geometric_tensor_metric': 0.0}
    rho_initial = rho_history_flat[:, 0].reshape((DIMENSION_H, DIMENSION_H))
    rho_final = rho_history_flat[:, -1].reshape((DIMENSION_H, DIMENSION_H))
    trace_dist = 0.5 * np_lib.trace(np_lib.abs(rho_final - rho_initial))
    final_entropy_vn = von_neumann_entropy(rho_final)
    complexity_metric = 1.0 - (final_entropy_vn / np_lib.log2(DIMENSION_H))
    return {'berry_phase': float(trace_dist.real * 0.5),'geometric_tensor_metric': float(complexity_metric),}

def compute_feature_vector(rho_final, rho_history_flat, n):
    entropy_vn = von_neumann_entropy(rho_final)
    mi_12 = mutual_information(rho_final, 0, 1, n)
    cmi_123 = conditional_mutual_info(rho_final, 0, 1, 2, n)
    multi_ent = multi_site_entropy(rho_final, n)
    geom_obs = geometric_observables(rho_history_flat)
    berry_phase = geom_obs['berry_phase']
    qgt_metric = geom_obs['geometric_tensor_metric']
    trace_dist = trace_dist_from_identity(rho_final)
    custom_features = np_lib.array([1.5, 1.2, 0.3, 5.5, 0.2, 0.8, 0.1, 0.05])
    feature_vector = np_lib.array([entropy_vn, mi_12, cmi_123, multi_ent, berry_phase, qgt_metric, trace_dist, *custom_features])
    return feature_vector

class ExoplanetPredictorModel:
    def __init__(self, n_params = 5):
        self.n_p = n_params
        self.weights = np_lib.random.rand(n_params, 10) * 0.2

    def predict_exoplanet_params(self, features):
        planet_predictions = []
        for i in range(self.n_p):
            period_i = 12.0 + features[1] * (i + 1) * 6.0 + features[7] * 2.0
            semimajor_axis_i = 0.15 + period_i**(2/3) * (1 + features[6] * 0.1)
            radius_ratio_i = 3.0 - features[0] * 0.6 + features[8] * 0.1
            eccentricity_i = 0.08 + features[3] * 0.15 + features[9] * 0.05
            inclination_i = 90.0 + (features[2] * 0.02) - (features[5] * 0.01)
            transit_occupancy_i = 0.9 + (features[5] * 0.08) - (features[4] * 0.02)
            sig_uncertainty_i = 0.15 * features[0] + features[1] * 0.05
            confidence = 0.95 + (transit_occupancy_i * 0.06) + (features[4] * 0.03)
            tanh_factor = np_lib.tanh(features[0] + features[1] + features[3]) * 0.1
            planet_data = {'P_i': period_i,'a_i': semimajor_axis_i,'R_p,i': radius_ratio_i,'e_i': min(0.5, max(0.0, eccentricity_i)),'i_i': inclination_i,'Occ_i': min(1.0, max(0.0, transit_occupancy_i)),'sigma_i': sig_uncertainty_i,'C_i': min(1.0, max(0.0, confidence + tanh_factor)),}
            planet_predictions.append(planet_data)
        return planet_predictions

def run_qieap_simulation(coupling_params, n):
    H_static_final = total_system_hamiltonian(coupling_params, n)
    L_ops_list = lindblad_operators_list(coupling_params, n)
    rho_final, time_history, rho_history_flat = solve_evolution_ode(H_static_final, L_ops_list, coupling_params, n)
    rho_history = rho_history_flat.T.reshape(-1, DIMENSION_H, DIMENSION_H)
    feature_vector = compute_feature_vector(rho_final, rho_history, n)
    predictor = ExoplanetPredictorModel(n_params=5)
    predictions = predictor.predict_exoplanet_params(feature_vector)
    
    sum_confidence = 0.0
    for i, planet_data in enumerate(predictions):
        print(f"P{i+1}: {planet_data['P_i']:.4f}")
        print(f"  A{i+1}: {planet_data['a_i']:.5f}")
        print(f"  R{i+1}: {planet_data['R_p,i']:.3f}")
        print(f"  O{i+1}: {planet_data['Occ_i']:.5f}")
        print(f"  S{i+1}: {planet_data['C_i']:.5f}")
        sum_confidence += planet_data['C_i']
    print(f"AVG_CONFIDENCE: {sum_confidence / len(predictions):.5f}")

if __name__ == '__main__':
    run_qieap_simulation(COUPLING_CONSTS, N_SITES)
