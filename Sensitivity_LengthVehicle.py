# Loading packages that are used in the code
import numpy as np
import os
import pandas as pd
from gurobipy import Model, GRB
import gurobipy as gp
from math import radians, cos, sin, asin, sqrt
import random
from typing import Dict, List
from itertools import combinations_with_replacement

# Get path to current folder
cwd = os.getcwd()

# Constants that will not change between runs
M = 10000  # Big M for time constraints
tighter_windows_instance = 0  # proportion of nodes with tightened time windows
Delta_GH = 1  # number of docks per GH
Docks = list(range(1, Delta_GH + 1))
K_trucks = [1, 2, 3]  # truck indices
n_uld_total = 6  # total number of ULDs (all standard type)
Weight_std = 1000  # weight of standard ULD [kg]
Length_std = 1.534  # length of standard ULD [m]
Proc_Time = 15  # processing time [minutes]
Horizon = 480  # total time limit [minutes]
Cap_W = 3000  # weight capacity of each truck (uniform) [kg]

# Truck type length capacities
Cap_L_std = 13.6  # standard truck length capacity [m]
Cap_L_med = 4.0   # medium truck length capacity [m]
Cap_L_mini = 2.0  # mini truck length capacity [m]

Speed_kmh = 35
Speed_mpm = Speed_kmh / 60.0  # km per minute

# Helper functions
def dms_to_dd(d, m, s, direction='N'):
    dd = d + m / 60 + s / 3600
    if direction in ['S', 'W']:
        dd *= -1
    return dd

def get_dist(coord1, coord2):
    lat1, lon1, lat2, lon2 = map(radians, [coord1[0], coord1[1], coord2[0], coord2[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def plot_routes(routes: Dict[int, List[int]], node_coords: List[List[float]], ff_nodes: Dict[int, List[int]], gh_nodes: Dict[int, List[int]], output_path: str = "truck_routes.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Run: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    depot = node_coords[0]
    ax.scatter(depot[1], depot[0], c="black", s=120, marker="s", label="Depot")
    for f, nodes in ff_nodes.items():
        for i in nodes:
            ax.scatter(node_coords[i][1], node_coords[i][0], c="#1f77b4", s=100, marker="o")
    for g, nodes in gh_nodes.items():
        for i in nodes:
            ax.scatter(node_coords[i][1], node_coords[i][0], c="#ff7f0e", s=100, marker="^")
    ax.scatter([], [], c="#1f77b4", s=100, marker="o", label="FF")
    ax.scatter([], [], c="#ff7f0e", s=100, marker="^", label="GH")
    
    colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    for idx, (k, route) in enumerate(routes.items()):
        if len(route) < 2:
            continue
        color = colors[idx % len(colors)]
        style = linestyles[idx % len(linestyles)]
        xs = [node_coords[i][1] for i in route]
        ys = [node_coords[i][0] for i in route]
        ax.plot(xs, ys, color=color, linestyle=style, linewidth=2, label=f"Truck {k}")
        ax.scatter(xs, ys, color=color, s=30)
    
    ax.set_title("Truck Routes (FFs/GHs)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"fleet Route plot saved to: {output_path}")

# Build static topology
n_uld_total = 6
Nodes_P = list(range(1, n_uld_total + 1))
Nodes_D = list(range(n_uld_total + 1, 2 * n_uld_total + 1))
All_Nodes = [0] + Nodes_P + Nodes_D

FFs = {
    1: Nodes_P[0:n_uld_total // 2],
    2: Nodes_P[n_uld_total // 2:],
}
GHs = {
    1: Nodes_D[0:n_uld_total // 2],
    2: Nodes_D[n_uld_total // 2:],
}

Edges = []
for i in All_Nodes:
    for j in All_Nodes:
        if i == j:
            continue
        if i == 0 and j in Nodes_D:
            continue
        if i in Nodes_P and j == 0:
            continue
        Edges.append((i, j))

locs = {
    'FF1': (dms_to_dd(52, 17, 46.8), dms_to_dd(4, 46, 10.4)),
    'FF2': (dms_to_dd(52, 18, 6.8), dms_to_dd(4, 45, 3.2)),
    'GH1': (dms_to_dd(52, 17, 0.8), dms_to_dd(4, 46, 7.1)),
    'GH2': (dms_to_dd(52, 16, 32.9), dms_to_dd(4, 44, 30.0))
}

node_loc_maps = [[52.2905, 4.7627]]
for i in Nodes_P:
    if i in FFs[1]:
        node_loc_maps.append(list(locs['FF1']))
    elif i in FFs[2]:
        node_loc_maps.append(list(locs['FF2']))
for i in Nodes_D:
    if i in GHs[1]:
        node_loc_maps.append(list(locs['GH1']))
    elif i in GHs[2]:
        node_loc_maps.append(list(locs['GH2']))

# Precompute travel times
T = {}
for i, j in Edges:
    T[i, j] = get_dist(node_loc_maps[i], node_loc_maps[j]) / Speed_mpm

# Static parameter dictionaries
P = {i: (Proc_Time if i != 0 else 0) for i in All_Nodes}
E_win = {i: 0 for i in All_Nodes}
D_win = {i: Horizon for i in All_Nodes}

# Time windows tightening (runs once)
tightened_P_windows = []
shuffled_Nodes = All_Nodes[1:].copy()
random.shuffle(shuffled_Nodes)
for i in shuffled_Nodes[: int(tighter_windows_instance * len(All_Nodes))]:
    if i in Nodes_P:
        E_win[i] = 50
        D_win[i] = 150
        tightened_P_windows.append(i)
    elif i in Nodes_D:
        E_win[i] = 200
        D_win[i] = 350
        if i - n_uld_total in tightened_P_windows:
            D_win[i] += 30


def solve_case(truck_types: tuple) -> dict:
    """Build & solve model for given fleet composition. 
    truck_types: tuple of 3 strings, e.g., ('std', 'med', 'mini')
    Returns results dict."""
    
    # Assign length capacities based on truck types
    cap_lengths = {
        1: Cap_L_std if truck_types[0] == 'std' else (Cap_L_med if truck_types[0] == 'med' else Cap_L_mini),
        2: Cap_L_std if truck_types[1] == 'std' else (Cap_L_med if truck_types[1] == 'med' else Cap_L_mini),
        3: Cap_L_std if truck_types[2] == 'std' else (Cap_L_med if truck_types[2] == 'med' else Cap_L_mini),
    }
    
    # Create configuration label
    config_label = f"{truck_types[0]}-{truck_types[1]}-{truck_types[2]}"
    n_std = sum(1 for t in truck_types if t == 'std')
    n_med = sum(1 for t in truck_types if t == 'med')
    n_mini = sum(1 for t in truck_types if t == 'mini')
    
    m = gp.Model(f"GHDC-PDPTW_Fleet_{config_label}")
    
    # Weights and lengths for standard ULDs only
    W = {i: 0 for i in All_Nodes}
    L = {i: 0 for i in All_Nodes}
    for node in Nodes_P:
        W[node] = Weight_std
        L[node] = Length_std
    
    # Variables
    x = m.addVars(Edges, K_trucks, vtype=GRB.BINARY, name="x")
    tau = m.addVars(All_Nodes, vtype=GRB.CONTINUOUS, name="tau")
    tau_end = m.addVars(K_trucks, vtype=GRB.CONTINUOUS, name="tau_end")
    a_F = m.addVars(K_trucks, FFs.keys(), vtype=GRB.CONTINUOUS, name="a_F")
    d_F = m.addVars(K_trucks, FFs.keys(), vtype=GRB.CONTINUOUS, name="d_F")
    a_G = m.addVars(K_trucks, GHs.keys(), vtype=GRB.CONTINUOUS, name="a_G")
    d_G = m.addVars(K_trucks, GHs.keys(), vtype=GRB.CONTINUOUS, name="d_G")
    w_D = m.addVars(K_trucks, GHs.keys(), vtype=GRB.CONTINUOUS, name="w_D")
    w_F = m.addVars(K_trucks, FFs.keys(), vtype=GRB.CONTINUOUS, name="w_F")
    w_G = m.addVars(K_trucks, GHs.keys(), vtype=GRB.CONTINUOUS, name="w_G")
    eta = m.addVars(K_trucks, K_trucks, GHs.keys(), vtype=GRB.BINARY, name="eta")
    y = m.addVars(K_trucks, Docks, GHs.keys(), vtype=GRB.BINARY, name="y")
    z = m.addVars(K_trucks, Docks, K_trucks, Docks, GHs.keys(), vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")
    
    # Objective components
    travel_cost = gp.quicksum(T[i, j] * x[i, j, k] for k in K_trucks for i, j in Edges)
    wait_gh_dock_cost = gp.quicksum(w_D[k, g] for k in K_trucks for g in GHs.keys())
    wait_gh_service_cost = gp.quicksum(w_G[k, g] for k in K_trucks for g in GHs.keys())
    wait_ff_cost = gp.quicksum(w_F[k, f] for k in K_trucks for f in FFs.keys())
    m.setObjective(travel_cost + wait_gh_dock_cost + wait_gh_service_cost + wait_ff_cost, GRB.MINIMIZE)
    
    depot = 0
    # [2]
    for i in Nodes_P:
        m.addConstr(gp.quicksum(x[j, i, k] for k in K_trucks for j in All_Nodes if (j, i) in Edges) == 1)
    # [3]
    n = len(Nodes_P)
    for k in K_trucks:
        for i in Nodes_P:
            delivery = i + n
            m.addConstr(gp.quicksum(x[j, i, k] for j in All_Nodes if (j, i) in Edges) - gp.quicksum(x[j, delivery, k] for j in All_Nodes if (j, delivery) in Edges) == 0)
    # [4]
    for k in K_trucks:
        m.addConstr(gp.quicksum(x[depot, i, k] for i in All_Nodes if (depot, i) in Edges) <= 1)
    # [5]
    for k in K_trucks:
        for i in All_Nodes:
            m.addConstr(gp.quicksum(x[j, i, k] for j in All_Nodes if (j, i) in Edges) - gp.quicksum(x[i, j, k] for j in All_Nodes if (i, j) in Edges) == 0)
    # [6]
    for k in K_trucks:
        for i in Nodes_P:
            m.addConstr(x[depot, i, k] - x[i + n, depot, k] == 0)
    # [7]
    for k in K_trucks:
        for (i, j) in Edges:
            if i in Nodes_P and j in Nodes_P:
                m.addConstr(x[i, j, k] - x[j + n, i + n, k] == 0)
    # [8–9]
    for k in K_trucks:
        for f in FFs:
            m.addConstr(gp.quicksum(x[j, i, k] for i in FFs[f] for j in All_Nodes if (j, i) in Edges and (j not in FFs[f]) and j != 0) + gp.quicksum(x[0, i, k] for i in FFs[f] if (0, i) in Edges) <= 1)
        for g in GHs:
            m.addConstr(gp.quicksum(x[j, i, k] for i in GHs[g] for j in All_Nodes if (j, i) in Edges and (j not in GHs[g]) and j != 0) + gp.quicksum(x[0, i, k] for i in GHs[g] if (0, i) in Edges) <= 1)
    # [10–14]
    for (i, j) in Edges:
        if j in Nodes_P:
            m.addConstr(tau[j] >= tau[i] + P[i] + T[i, j] - M * (1 - gp.quicksum(x[i, j, k] for k in K_trucks)))
    for g in GHs:
        nodes_g = GHs[g]
        for k in K_trucks:
            for (i, j) in Edges:
                if j in nodes_g and i not in nodes_g and j in Nodes_D:
                    m.addConstr(tau[j] >= tau[i] + P[i] + T[i, j] - M * (1 - x[i, j, k]) + w_D[k, g])
    for g, nodes in GHs.items():
        for i in nodes:
            for j in nodes:
                if i != j and (i, j) in Edges:
                    expr = gp.quicksum(x[i, j, k] for k in K_trucks)
                    m.addConstr(tau[j] >= tau[i] + P[i] + T[i, j] - (1 - expr) * M)
    for k in K_trucks:
        for i in All_Nodes:
            if i != 0 and (i, 0) in Edges:
                m.addConstr(tau_end[k] >= tau[i] + P[i] + T[i, 0] - (1 - x[i, 0, k]) * M)
    for i in Nodes_P + Nodes_D:
        m.addConstr(tau[i] >= E_win[i])
        m.addConstr(tau[i] <= D_win[i])
    # [15–16] capacity constraints - HETEROGENEOUS LENGTH CAPACITIES
    for k in K_trucks:
        load_expr = gp.quicksum(W[i] * x[j, i, k] for i in All_Nodes for j in All_Nodes if (j, i) in Edges)
        m.addConstr(load_expr <= Cap_W)
        # Use truck-specific length capacity
        len_expr = gp.quicksum(L[i] * x[j, i, k] for i in All_Nodes for j in All_Nodes if (j, i) in Edges)
        m.addConstr(len_expr <= cap_lengths[k])
    # [17–33] remaining constraints
    for f, f_nodes in FFs.items():
        for k in K_trucks:
            for i, j in Edges:
                if j in f_nodes and i not in f_nodes:
                    m.addConstr(a_F[k,f] >= tau[i] + P[i] + T[i,j] - (1 - x[i,j,k])*M)
                    m.addConstr(a_F[k,f] <= tau[i] + P[i] + T[i,j] + (1 - x[i,j,k])*M)
    for f, f_nodes in FFs.items():
        for k in K_trucks:
            for i, j in Edges:
                if i in f_nodes and j not in f_nodes:
                    m.addConstr(d_F[k,f] >= tau[i] + P[i] - (1 - x[i,j,k])*M)
                    m.addConstr(d_F[k,f] <= tau[i] + P[i] + (1 - x[i,j,k])*M)
    for g, g_nodes in GHs.items():
        for k in K_trucks:
            for i, j in Edges:
                if j in g_nodes and i not in g_nodes:
                    m.addConstr(a_G[k,g] >= tau[i] + P[i] + T[i,j] - (1 - x[i,j,k])*M)
                    m.addConstr(a_G[k,g] <= tau[i] + P[i] + T[i,j] + (1 - x[i,j,k])*M)
    for g in GHs:
        for k in K_trucks:
            for i in GHs[g]:
                for j in All_Nodes:
                    if j not in GHs[g] and (i,j) in Edges:
                        m.addConstr(d_G[k, g] >= tau[i] + P[i] - (1 - x[i, j, k]) * M)
                        m.addConstr(d_G[k, g] <= tau[i] + P[i] + (1 - x[i, j, k]) * M)
    for f in FFs:
        for k in K_trucks:
            proc_sum = gp.quicksum(P[i] * x[j, i, k] for i in FFs[f] for j in All_Nodes if (j, i) in Edges)
            m.addConstr(w_F[k, f] >= d_F[k, f] - a_F[k, f] - proc_sum)
    for g in GHs:
        for k in K_trucks:
            proc_sum = gp.quicksum(P[i] * x[j, i, k] for i in GHs[g] for j in All_Nodes if (j, i) in Edges)
            m.addConstr(w_G[k, g] >= d_G[k, g] - a_G[k, g] - w_D[k, g] - proc_sum)
    for g in GHs:
        for k1 in K_trucks:
            for k2 in K_trucks:
                m.addConstr(d_G[k1, g] - a_G[k2, g] - w_D[k2, g] + M * eta[k1, k2, g] <= M)
                m.addConstr(-d_G[k1, g] + a_G[k2, g] + w_D[k2, g] - M * eta[k1, k2, g] <= 0)
    for g in GHs:
        for k in K_trucks:
            visit_from_outside = gp.quicksum(x[j, i, k] for i in GHs[g] for j in All_Nodes if (j, i) in Edges and (j not in GHs[g]))
            m.addConstr(gp.quicksum(y[k, d, g] for d in Docks) == visit_from_outside)
    for g in GHs:
        for k1 in K_trucks:
            for k2 in K_trucks:
                if k1 < k2:
                    for d1 in Docks:
                        for d2 in Docks:
                            m.addConstr(z[k1, d1, k2, d2, g] <= y[k1, d1, g])
                            m.addConstr(z[k1, d1, k2, d2, g] <= y[k2, d2, g])
                            m.addConstr(y[k1, d1, g] + y[k2, d2, g] - 1 <= z[k1, d1, k2, d2, g])
    for g in GHs:
        for k1 in K_trucks:
            for k2 in K_trucks:
                if k1 < k2:
                    for d1 in Docks:
                        m.addConstr(z[k1, d1, k2, d1, g] <= eta[k1, k2, g] + eta[k2, k1, g])
    
    m.update()
    m.optimize()
    
    def val(v):
        try:
            return float(v.X)
        except:
            try:
                return float(v)
            except:
                return 0.0
    
    result = {
        'config': config_label,
        'n_std': n_std,
        'n_med': n_med,
        'n_mini': n_mini,
        'obj': m.objVal if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else None,
        'travel': travel_cost.getValue() if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else None,
        'wait_gh_dock': wait_gh_dock_cost.getValue() if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else None,
        'wait_gh_service': wait_gh_service_cost.getValue() if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else None,
        'wait_ff': wait_ff_cost.getValue() if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else None,
        'feasible': m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL],
    }
    
    routes = {}
    cap_usage = {}
    
    # Only extract routes and capacity if feasible
    if result['feasible']:
        for k in K_trucks:
            succ = {i: j for (i, j) in Edges if val(x[i, j, k]) > 0.5}
            route = [0]
            cur = 0
            visited = set([0])
            for _ in range(len(All_Nodes) + 2):
                if cur in succ:
                    nxt = succ[cur]
                    route.append(nxt)
                    if nxt in visited:
                        break
                    visited.add(nxt)
                    cur = nxt
                else:
                    break
            routes[k] = route
        
        plot_routes(routes, node_loc_maps, FFs, GHs, output_path=f"fleet_routes_{config_label}.png")
        
        trucks_used = 0
        for k in K_trucks:
            used_arcs = [(i, j) for (i, j) in Edges if val(x[i, j, k]) > 0.5]
            used_flag = 1 if used_arcs else 0
            trucks_used += used_flag
            weight = 0.0
            length = 0.0
            for i in Nodes_P:
                if any(val(x[j, i, k]) > 0.5 for j in All_Nodes if (j, i) in Edges):
                    weight += W[i]
                    length += L[i]
            cap_usage[k] = {'weight': weight, 'length': length, 'cap_length': cap_lengths[k]}
        result['trucks_used'] = trucks_used
    else:
        result['trucks_used'] = None
        for k in K_trucks:
            cap_usage[k] = {'weight': 0, 'length': 0, 'cap_length': cap_lengths[k]}
    
    result['capacity_usage'] = cap_usage
    
    # Extract tau values only if feasible
    if result['feasible']:
        result['tau'] = {i: val(tau[i]) for i in All_Nodes}
        # include routes as well (computed above)
        result['routes'] = routes
    else:
        result['tau'] = {i: 0 for i in All_Nodes}
        result['routes'] = {}
    
    return result


def main():
    """Run fleet composition sensitivity analysis."""
    
    # Generate all fleet compositions: (n_std, n_med, n_mini) where sum = 3
    fleet_configs = []
    for n_std in range(4):
        for n_med in range(4 - n_std):
            n_mini = 3 - n_std - n_med
            truck_types = tuple(['std']*n_std + ['med']*n_med + ['mini']*n_mini)
            fleet_configs.append(truck_types)
    
    results = []
    tau_records = []
    cap_records = []
    
    print(f"\nfleet === Starting fleet composition sensitivity analysis ===")
    print(f"fleet Testing {len(fleet_configs)} fleet configurations")
    
    for config in fleet_configs:
        config_label = f"{config[0]}-{config[1]}-{config[2]}"
        print(f"\nfleet === solving for fleet: {config_label} ===")
        res = solve_case(config)
        
        if not res['feasible']:
            print(f"fleet INFEASIBLE for config {config_label}")
        
        results.append(res)
        for i, t in res['tau'].items():
            tau_records.append({'config': res['config'], 'node': i, 'tau': t})
        for k, usage in res['capacity_usage'].items():
            cap_records.append({'config': res['config'], 'truck': k, 'weight': usage['weight'], 
                               'length': usage['length'], 'cap_length': usage['cap_length']})
    
    df_res = pd.DataFrame(results)
    df_tau = pd.DataFrame(tau_records)
    df_cap = pd.DataFrame(cap_records)
    
    # Separate feasible and infeasible
    df_feasible = df_res[df_res['feasible']].copy()
    df_infeasible = df_res[~df_res['feasible']].copy()
    
    # Sort feasible by objective (descending order as user requested)
    df_feasible = df_feasible.sort_values('obj', ascending=False).reset_index(drop=True)
    
    # Generate comprehensive sensitivity report
    report = []
    for idx, row in df_feasible.iterrows():
        config = row['config']
        cap_data = df_cap[df_cap['config'] == config]
        
        avg_weight_usage = cap_data['weight'].sum() / len(K_trucks) if len(cap_data) > 0 else 0
        weight_util_pct = (avg_weight_usage / Cap_W) * 100 if Cap_W > 0 else 0
        
        avg_length_usage = cap_data['length'].sum() / len(K_trucks) if len(cap_data) > 0 else 0
        avg_cap_length = cap_data['cap_length'].sum() / len(K_trucks) if len(cap_data) > 0 else Cap_L_std
        length_util_pct = (avg_length_usage / avg_cap_length) * 100 if avg_cap_length > 0 else 0
        
        tau_data = df_tau[df_tau['config'] == config]
        tau_vals = tau_data['tau'].values
        max_tau = tau_vals.max() if len(tau_vals) > 0 and tau_vals.max() > 0 else 0
        # adjust completion by subtracting depot departure time if delayed
        depot_tau = tau_data[tau_data['node'] == 0]['tau'].values
        if len(depot_tau) > 0:
            max_tau = max_tau - depot_tau[0]
        
        report.append({
            'Fleet': config,
            'Std': int(row['n_std']),
            'Med': int(row['n_med']),
            'Mini': int(row['n_mini']),
            'Trucks': int(row['trucks_used']) if row['trucks_used'] is not None else 'N/A',
            'Obj (min)': f"{row['obj']:.2f}",
            'Travel (min)': f"{row['travel']:.2f}",
            'GH Pre-Dock (min)': f"{row['wait_gh_dock']:.2f}",
            'GH Service (min)': f"{row['wait_gh_service']:.2f}",
            'FF Dock (min)': f"{row['wait_ff']:.2f}",
            'Avg Weight (kg)': f"{avg_weight_usage:.1f}",
            'Weight Util (%)': f"{weight_util_pct:.1f}",
            'Avg Length (m)': f"{avg_length_usage:.2f}",
            'Length Util (%)': f"{length_util_pct:.1f}",
            'Completion (min)': f"{max_tau:.2f}",
        })
    
    # Add infeasible cases
    for idx, row in df_infeasible.iterrows():
        report.append({
            'Fleet': row['config'],
            'Std': int(row['n_std']),
            'Med': int(row['n_med']),
            'Mini': int(row['n_mini']),
            'Trucks': 'N/A',
            'Obj (min)': 'INFEASIBLE',
            'Travel (min)': '-',
            'GH Pre-Dock (min)': '-',
            'GH Service (min)': '-',
            'FF Dock (min)': '-',
            'Avg Weight (kg)': '-',
            'Weight Util (%)': '-',
            'Avg Length (m)': '-',
            'Length Util (%)': '-',
            'Completion (min)': '-',
        })
    
    df_report = pd.DataFrame(report)
    df_report.to_csv('fleet_sensitivity_report.csv', index=False)
    print('\nfleet Saved fleet_sensitivity_report.csv')
    print('\nfleet FLEET COMPOSITION SENSITIVITY REPORT (sorted by objective, decreasing):')
    print(df_report.to_string(index=False))
    
    # Create simplified table for feasible configurations only
    print('\n\nfleet FLEET COMPOSITION TABLE (FEASIBLE ONLY):')
    simple_table = []
    for idx, row in df_feasible.iterrows():
        config = row['config']
        cap_data = df_cap[df_cap['config'] == config]
        tau_data = df_tau[df_tau['config'] == config]
        tau_vals = tau_data['tau'].values
        max_tau = tau_vals.max() if len(tau_vals) > 0 else 0
        depot_tau = tau_data[tau_data['node'] == 0]['tau'].values
        if len(depot_tau) > 0:
            max_tau = max_tau - depot_tau[0]
        
        simple_table.append({
            'Fleet Config': config,
            'Trucks Used': int(row['trucks_used']) if row['trucks_used'] is not None else 'N/A',
            'Obj. (min)': f"{row['obj']:.2f}",
            'Travel (min)': f"{row['travel']:.2f}",
            'GH Pre-Dock Wait (min)': f"{row['wait_gh_dock']:.2f}",
            'GH Dock Wait (min)': f"{row['wait_gh_service']:.2f}",
            'FF Dock Wait (min)': f"{row['wait_ff']:.2f}",
            'Avg Weight (kg)': '2000.0',
            'Max Service Time (min)': f"{max_tau:.2f}",
        })
    
    df_simple = pd.DataFrame(simple_table)
    print(df_simple.to_string(index=False))
    
    # Generate graphs
    import matplotlib.pyplot as plt
    
    # Plot 1: Objective value and completion time (sorted by decreasing objective)
    configs_labels = [row['config'] for _, row in df_feasible.iterrows()]
    configs_labels.extend([row['config'] for _, row in df_infeasible.iterrows()])
    
    objs_feas = df_feasible['obj'].values
    compl_feas = []
    for config in df_feasible['config']:
        tau_data = df_tau[df_tau['config'] == config]
        if len(tau_data) > 0:
            max_tau = tau_data['tau'].max()
            depot_tau = tau_data[tau_data['node'] == 0]['tau'].values
            if len(depot_tau) > 0:
                max_tau = max_tau - depot_tau[0]
        else:
            max_tau = 0
        compl_feas.append(max_tau)
    compl_feas = np.array(compl_feas)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot feasible points
    if len(df_feasible) > 0:
        ax.plot(range(len(df_feasible)), objs_feas, marker='o', color='#00bfff', linewidth=2.5, 
                markersize=8, label='Objective Value (Feasible)')
    
    # Plot infeasible as red X markers at top of plot
    if len(df_infeasible) > 0:
        infeas_indices = range(len(df_feasible), len(df_feasible) + len(df_infeasible))
        ax.scatter(infeas_indices, [38]*len(df_infeasible), marker='x', color='red', s=300, 
                  linewidth=3, label='INFEASIBLE', zorder=5)
    
    ax.set_xlabel('Fleet Configuration (sorted by decreasing objective)', fontsize=12)
    ax.set_ylabel('Objective value (min)', color='#00bfff', fontsize=12)
    ax.tick_params(axis='y', labelcolor='#00bfff')
    ax.set_ylim(10, 40)
    ax.set_xticks(range(len(configs_labels)))
    ax.set_xticklabels(configs_labels, rotation=45, ha='right')
    
    # Secondary axis for completion time
    ax2 = ax.twinx()
    if len(df_feasible) > 0:
        ax2.plot(range(len(df_feasible)), compl_feas, color='#8B0000', linestyle=':', linewidth=2.5, 
                marker='s', markersize=8, label='Completion Time (Feasible)')
    ax2.set_ylabel('Completion time (min)', color='#8B0000', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#8B0000')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    fig.tight_layout()
    fig.savefig('fleet_obj_vs_composition.png', dpi=150)
    print('fleet Saved fleet_obj_vs_composition.png')
    
    # Plot 2: Trucks used bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors_bar = ['#2ca02c' if feas else '#d62728' for feas in 
                  [True]*len(df_feasible) + [False]*len(df_infeasible)]
    trucks_vals = list(df_feasible['trucks_used'].fillna(0).values) + [0]*len(df_infeasible)
    
    bars = ax.bar(range(len(configs_labels)), trucks_vals, color=colors_bar, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Fleet Configuration', fontsize=12)
    ax.set_ylabel('Trucks Used', fontsize=12)
    ax.set_xticks(range(len(configs_labels)))
    ax.set_xticklabels(configs_labels, rotation=45, ha='right')
    ax.set_ylim(0, 3.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend for infeasible
    if len(df_infeasible) > 0:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ca02c', alpha=0.7, edgecolor='black', label='Feasible'),
                          Patch(facecolor='#d62728', alpha=0.7, edgecolor='black', label='Infeasible')]
        ax.legend(handles=legend_elements, loc='upper right')
    
    fig.tight_layout()
    fig.savefig('fleet_trucks_vs_composition.png', dpi=150)
    print('fleet Saved fleet_trucks_vs_composition.png')
    
    # Save detailed CSVs
    df_tau.to_csv('fleet_taus_per_node.csv', index=False)
    print('fleet Saved fleet_taus_per_node.csv')
    
    df_cap.to_csv('fleet_capacity_usage.csv', index=False)
    print('fleet Saved fleet_capacity_usage.csv')
    
    print('\nfleet Finished fleet composition sensitivity analysis')


if __name__ == '__main__':
    main()


