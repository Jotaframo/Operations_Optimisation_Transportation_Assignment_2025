# Loading packages that are used in the code
import numpy as np
import os
import pandas as pd
from gurobipy import Model,GRB
import gurobipy as gp
from math import radians, cos, sin, asin, sqrt
import random
from typing import Dict, List, Tuple

# Get path to current folder
cwd = os.getcwd()
RESULTS_DIR = os.path.join(cwd, "results_tightwin_sensitivity")
os.makedirs(RESULTS_DIR, exist_ok=True)


def out_path(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)


def annotate_heatmap_cells(ax, data, fmt: str = "{:.1f}"):
    max_val = np.nanmax(data) if data.size > 0 else 0
    thresh = max_val / 2 if max_val and not np.isnan(max_val) else 0
    n_rows, n_cols = data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                label = "-"
                color = "black"
            else:
                label = fmt.format(v)
                color = "white" if v > thresh else "black"
            ax.text(j, i, label, ha='center', va='center', fontsize=9, color=color)

# Constants that will not change between runs
M = 10000  # Big M for time constraints
Delta_GH = 1  # number of docks per GH
Docks = list(range(1, Delta_GH + 1))
n_uld = 6  # number of ULDs
K_trucks = [1, 2, 3]  # truck indices
Weight_u = 1000  # weight of each ULD [kg]
Length_u = 1.534  # length of each ULD [m]
Proc_Time = 15  # processing time [minutes]
Horizon = 480  # total time limit [minutes]
Cap_W = 3000  # weight capacity of each truck [kg]
Cap_L = 13.6  # length capacity of each truck [m]
Speed_kmh = 35
Speed_mpm = Speed_kmh / 60.0  # km per minute

# base windows used when a node is selected for tightening
BASE_PICKUP_START = 50
BASE_DELIVERY_START = 200

# helper functions

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
    plt.close(fig)
    print(f"Route plot saved to: {output_path}")

# build static topology
Nodes_P = list(range(1, n_uld + 1))
Nodes_D = list(range(n_uld + 1, 2 * n_uld + 1))
All_Nodes = [0] + Nodes_P + Nodes_D

FFs = {
    1: Nodes_P[0:n_uld // 2],
    2: Nodes_P[n_uld // 2:],
}
GHs = {
    1: Nodes_D[0:n_uld // 2],
    2: Nodes_D[n_uld // 2:],
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
        if j in Nodes_D:
            pickup_node = j - n_uld
            if pickup_node in Nodes_P:
                if i != pickup_node and i not in Nodes_D:
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

# precompute travel times
T = {}
for i, j in Edges:
    T[i, j] = get_dist(node_loc_maps[i], node_loc_maps[j]) / Speed_mpm

# other static parameter dictionaries
P = {i: (Proc_Time if i != 0 else 0) for i in All_Nodes}
W = {i: (Weight_u if i in Nodes_P else 0) for i in All_Nodes}
L = {i: (Length_u if i in Nodes_P else 0) for i in All_Nodes}

def generate_time_windows(tight_proportion: float, window_width: float, seed: int = 42) -> Tuple[Dict[int, float], Dict[int, float], List[int]]:
    """Generate E/D windows for one scenario.
    - Nodes not selected keep [0, Horizon].
    - Selected pickups get [BASE_PICKUP_START, BASE_PICKUP_START + width].
    - Selected deliveries get [BASE_DELIVERY_START, BASE_DELIVERY_START + width].
    - If pickup i is tightened and delivery i+n is tightened, delivery upper bound gets +30 min.
    """
    width = max(1.0, float(window_width))
    proportion = min(1.0, max(0.0, float(tight_proportion)))

    E_win = {i: 0 for i in All_Nodes}
    D_win = {i: Horizon for i in All_Nodes}

    rng = random.Random(seed)
    candidate_nodes = All_Nodes[1:].copy()
    rng.shuffle(candidate_nodes)

    n_tight = int(round(proportion * len(candidate_nodes)))
    selected = candidate_nodes[:n_tight]

    tightened_pickups = []
    for i in selected:
        if i in Nodes_P:
            E_win[i] = BASE_PICKUP_START
            D_win[i] = min(Horizon, BASE_PICKUP_START + width)
            tightened_pickups.append(i)
        elif i in Nodes_D:
            E_win[i] = BASE_DELIVERY_START
            D_win[i] = min(Horizon, BASE_DELIVERY_START + width)
            if (i - n_uld) in tightened_pickups:
                D_win[i] = min(Horizon, D_win[i] + 30)

    return E_win, D_win, selected


def solve_case(tight_proportion: float, window_width: float, case_label: str, seed: int = 42) -> dict:
    """Build & solve model for given tightening settings. Returns results."""
    E_win, D_win, selected_nodes = generate_time_windows(tight_proportion, window_width, seed=seed)
    m = gp.Model(f"GHDC-PDPTW_Tight_{case_label}")

    # variables
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

    # objective components
    travel_cost = gp.quicksum(T[i, j] * x[i, j, k]
                              for k in K_trucks
                              for i, j in Edges)
    wait_gh_dock_cost = gp.quicksum(w_D[k, g]
                                    for k in K_trucks
                                    for g in GHs.keys())
    wait_gh_service_cost = gp.quicksum(w_G[k, g]
                                       for k in K_trucks
                                       for g in GHs.keys())
    wait_ff_cost = gp.quicksum(w_F[k, f]
                               for k in K_trucks
                               for f in FFs.keys())
    m.setObjective(travel_cost + wait_gh_dock_cost + wait_gh_service_cost + wait_ff_cost, GRB.MINIMIZE)

    depot = 0
    # [2]
    for i in Nodes_P:
        m.addConstr(
            gp.quicksum(x[j, i, k]
                     for k in K_trucks
                     for j in All_Nodes
                     if (j, i) in Edges) == 1)
    # [3]
    n = len(Nodes_P)
    for k in K_trucks:
        for i in Nodes_P:
            delivery = i + n
            m.addConstr(
                gp.quicksum(x[j, i, k] for j in All_Nodes if (j, i) in Edges) -
                gp.quicksum(x[j, delivery, k] for j in All_Nodes if (j, delivery) in Edges)
                == 0)
    # [4]
    for k in K_trucks:
        m.addConstr(
            gp.quicksum(x[depot, i, k] for i in All_Nodes if (depot, i) in Edges) <= 1)
    # [5]
    for k in K_trucks:
        for i in All_Nodes:
            m.addConstr(
                gp.quicksum(x[j, i, k] for j in All_Nodes if (j, i) in Edges) -
                gp.quicksum(x[i, j, k] for j in All_Nodes if (i, j) in Edges) == 0)
    # [6]
    for k in K_trucks:
        for i in Nodes_P:
            m.addConstr(x[depot, i, k] - x[i + n, depot, k] == 0)
    # [7]
    for k in K_trucks:
        for (i, j) in Edges:
            if i in Nodes_P and j in Nodes_P:
                m.addConstr(x[i, j, k] - x[j + n, i + n, k] == 0)
    # [8] and [9]
    for k in K_trucks:
        for f in FFs:
            m.addConstr(
                gp.quicksum(
                    x[j, i, k]
                    for i in FFs[f]
                    for j in All_Nodes
                    if (j, i) in Edges and (j not in FFs[f]) and j != 0
                )
                + gp.quicksum(
                    x[0, i, k] for i in FFs[f] if (0, i) in Edges
                )
                <= 1)
        for g in GHs:
            m.addConstr(
                gp.quicksum(
                    x[j, i, k]
                    for i in GHs[g]
                    for j in All_Nodes
                    if (j, i) in Edges and (j not in GHs[g]) and j != 0
                )
                + gp.quicksum(
                    x[0, i, k] for i in GHs[g] if (0, i) in Edges
                )
                <= 1)
    # [10–14]
    for (i, j) in Edges:
        if j in Nodes_P:
            m.addConstr(
                tau[j] >= tau[i] + P[i] + T[i, j]
                          - M * (1 - gp.quicksum(x[i, j, k] for k in K_trucks)))
    for g in GHs:
        nodes_g = GHs[g]
        for k in K_trucks:
            for (i, j) in Edges:
                if j in nodes_g and i not in nodes_g and j in Nodes_D:
                    m.addConstr(
                        tau[j] >= tau[i] + P[i] + T[i, j] - M * (1 - x[i, j, k]) + w_D[k, g])
    for g, nodes in GHs.items():
        for i in nodes:
            for j in nodes:
                if i != j and (i, j) in Edges:
                    expr = gp.quicksum(x[i, j, k] for k in K_trucks)
                    m.addConstr(tau[j] >= tau[i] + P[i] + T[i, j] - (1 - expr) * M)
    for k in K_trucks:
        for i in All_Nodes:
            if i != 0 and (i, 0) in Edges:
                m.addConstr(
                    tau_end[k] >= tau[i] + P[i] + T[i, 0] - (1 - x[i, 0, k]) * M)
    for i in Nodes_P + Nodes_D:
        m.addConstr(tau[i] >= E_win[i])
        m.addConstr(tau[i] <= D_win[i])
    # [15–16] capacity constraints (fixed truck capacities)
    for k in K_trucks:
        load_expr = gp.quicksum(W[i] * x[j, i, k]
                                for i in All_Nodes
                                for j in All_Nodes if (j, i) in Edges)
        m.addConstr(load_expr <= Cap_W)
        len_expr = gp.quicksum(L[i] * x[j, i, k]
                               for i in All_Nodes
                               for j in All_Nodes if (j, i) in Edges)
        m.addConstr(len_expr <= Cap_L)
    # (17 & 18) Facility Arrival Time (a_F) (If truck k travels i -> j, and j is in Facility f but i is NOT, record arrival time)
    for f, f_nodes in FFs.items():
        for k in K_trucks:
            for i, j in Edges:
                if j in f_nodes and i not in f_nodes:
                    # [17] Lower Bound
                    m.addConstr(a_F[k,f] >= tau[i] + P[i] + T[i,j] - (1 - x[i,j,k])*M,
                                name=f"Eq17_ArrF_LB_{k}_{f}")
                    # [18] Upper Bound
                    m.addConstr(a_F[k,f] <= tau[i] + P[i] + T[i,j] + (1 - x[i,j,k])*M,
                                name=f"Eq18_ArrF_UB_{k}_{f}")

    # (19 & 20) Facility Departure Time (d_F) (If truck k travels i -> j, and i is in Facility f but j is NOT, record departure time)
    for f, f_nodes in FFs.items():
        for k in K_trucks:
            for i, j in Edges:
                if i in f_nodes and j not in f_nodes:
                    # [19] Lower Bound
                    m.addConstr(d_F[k,f] >= tau[i] + P[i] - (1 - x[i,j,k])*M,
                                name=f"Eq19_DepF_LB_{k}_{f}")
                    # [20] Upper Bound
                    m.addConstr(d_F[k,f] <= tau[i] + P[i] + (1 - x[i,j,k])*M,
                                name=f"Eq20_DepF_UB_{k}_{f}")

    # (21 & 22) Group Arrival Time (a_G) (If truck k travels i -> j, and j is in Group g but i is NOT, record arrival time)
    for g, g_nodes in GHs.items():
        for k in K_trucks:
            for i, j in Edges:
                if j in g_nodes and i not in g_nodes:
                    # [21] Lower Bound
                    m.addConstr(a_G[k,g] >= tau[i] + P[i] + T[i,j] - (1 - x[i,j,k])*M,
                                name=f"Eq21_ArrG_LB_{k}_{g}")
                    # [22] Upper Bound
                    m.addConstr(a_G[k,g] <= tau[i] + P[i] + T[i,j] + (1 - x[i,j,k])*M,
                                name=f"Eq22_ArrG_UB_{k}_{g}")

    # (23 & 24) Departure time from GH 
    for g in GHs:
        for k in K_trucks:
            for i in GHs[g]: # Node in GH
                for j in  All_Nodes: # Possible next node
                    if j not in GHs[g] and (i,j) in Edges:
                        # [23] Lower Bound
                        m.addConstr(d_G[k, g] >= tau[i] + P[i] - (1 - x[i, j, k]) * M, 
                                    name=f"C23_DepGH_LB_k{k}_g{g}_i{i}")
                            # [24] Upper Bound
                        m.addConstr(d_G[k, g] <= tau[i] + P[i] + (1 - x[i, j, k]) * M, 
                                    name=f"C24_DepGH_UB_k{k}_g{g}_i{i}")

    # [25] Waiting time at FF while docked (Waiting >= Departure - Arrival - Total Processing at that FF)
    for f in FFs:
        for k in K_trucks:
            # Sum of processing times for all nodes visited by k in FF f
            proc_sum =  gp.quicksum(P[i] * x[j, i, k] for i in FFs[f] for j in  All_Nodes if (j, i) in Edges)
            m.addConstr(w_F[k, f] >= d_F[k, f] - a_F[k, f] - proc_sum, 
                        name=f"C25_WaitFF_k{k}_f{f}")

    # [26] Waiting time at GH while docked (Waiting >= Departure - Arrival - Waiting before docking - Total Processing at that GH)
    for g in GHs:
        for k in K_trucks:
            proc_sum =  gp.quicksum(P[i] * x[j, i, k] for i in GHs[g] for j in  All_Nodes if (j, i) in Edges)
            m.addConstr(w_G[k, g] >= d_G[k, g] - a_G[k, g] - w_D[k, g] - proc_sum, 
                        name=f"C26_WaitGH_k{k}_g{g}")

    # [27] Overlap variable definition (Lower Bound logic) (If eta=1, then Departure of k1 <= Arrival of k2 + Waiting of k2 (k1 leaves before k2 uses dock))
    for g in GHs:
        for k1 in K_trucks:
            for k2 in K_trucks:
                m.addConstr(d_G[k1, g] - a_G[k2, g] - w_D[k2, g] + M * eta[k1, k2, g] <= M,
                            name=f"C27_OverlapLB_k{k1}_k{k2}_g{g}")

    # [28] Overlap variable definition (Upper Bound logic) (Enforces the binary nature relative to the time difference)
    for g in GHs:
        for k1 in K_trucks:
            for k2 in K_trucks:
                m.addConstr(-d_G[k1, g] + a_G[k2, g] + w_D[k2, g] - M * eta[k1, k2, g] <= 0,
                            name=f"C28_OverlapUB_k{k1}_k{k2}_g{g}")

    # [29] Dock Assignment Constraint (Each truck visiting GH g must be assigned exactly one dock there)
    for g in GHs:
        for k in K_trucks:
            visit_from_outside = gp.quicksum(
                x[j, i, k] for i in GHs[g] for j in All_Nodes
                if (j, i) in Edges and (j not in GHs[g])
            )
            m.addConstr(
                gp.quicksum(y[k, d, g] for d in Docks) == visit_from_outside,
                name=f"C29_AssignDock_k{k}_g{g}"
            )

    # [30] Linearization of z (Part 1) (z <= y1)
    for g in GHs:
        for k1 in K_trucks:
            for k2 in K_trucks:
                if k1 < k2:
                    for d1 in Docks:
                        for d2 in Docks:
                            m.addConstr(z[k1, d1, k2, d2, g] <= y[k1, d1, g],
                                        name=f"C30_LinZ1_k{k1}_k{k2}_d{d1}_d{d2}_g{g}")

    # [31] Linearization of z (Part 2) (z <= y2)
    for g in GHs:
        for k1 in K_trucks:
            for k2 in K_trucks:
                if k1 < k2:
                    for d1 in Docks:
                        for d2 in Docks:
                            m.addConstr(z[k1, d1, k2, d2, g] <= y[k2, d2, g],
                                        name=f"C31_LinZ2_k{k1}_k{k2}_d{d1}_d{d2}_g{g}")

    # [32] Linearization of z (Part 3) (y1 + y2 - 1 <= z (Forces z=1 if both y1 and y2 are 1))
    for g in GHs:
        for k1 in K_trucks:
            for k2 in K_trucks:
                if k1 < k2:
                    for d1 in Docks:
                        for d2 in Docks:
                            m.addConstr(y[k1, d1, g] + y[k2, d2, g] - 1 <= z[k1, d1, k2, d2, g],
                                        name=f"C32_LinZ3_k{k1}_k{k2}_d{d1}_d{d2}_g{g}")

    # [33] Dock Capacity / Non-overlap constraint (Two trucks can be assigned to the SAME dock (d1) only if they do not overlap in time)

    for g in GHs:
        for k1 in K_trucks:
            for k2 in K_trucks:
                if k1 < k2:
                    for d1 in Docks:
                        # Note: Indices for z are k1, d1, k2, d1 (same dock)
                        m.addConstr(z[k1, d1, k2, d1, g] <= eta[k1, k2, g] + eta[k2, k1, g],
                                    name=f"C33_NoOverlap_k{k1}_k{k2}_d{d1}_g{g}")


    m.update()
    m.optimize()

    def val(v):
        try:
            return float(v.X)
        except:
            return float(v)

    has_solution = m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and m.SolCount > 0
    result = {
        'tight_proportion': float(tight_proportion),
        'window_width': float(window_width),
        'scenario': case_label,
        'tightened_nodes_count': len(selected_nodes),
        'feasible': bool(has_solution),
        'obj': m.objVal if has_solution else None,
        'travel': travel_cost.getValue() if has_solution else None,
        'wait_gh_dock': wait_gh_dock_cost.getValue() if has_solution else None,
        'wait_gh_service': wait_gh_service_cost.getValue() if has_solution else None,
        'wait_ff': wait_ff_cost.getValue() if has_solution else None,
    }

    cap_usage = {}
    if has_solution:
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
            cap_usage[k] = {'weight': weight, 'length': length}

        result['trucks_used'] = trucks_used
        result['tau'] = {i: val(tau[i]) for i in All_Nodes}
    else:
        for k in K_trucks:
            cap_usage[k] = {'weight': 0.0, 'length': 0.0}
        result['trucks_used'] = None
        result['tau'] = {i: 0.0 for i in All_Nodes}

    result['capacity_usage'] = cap_usage
    result['selected_nodes'] = selected_nodes

    return result


def main():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    widths_case_1 = [30, 50, 70, 90, 110, 140]
    props_case_2 = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]

    fixed_prop = 0.5
    fixed_width = 90

    all_results = []
    tau_records = []
    cap_records = []

    print("tightwin\n=== CASE 1: fixed proportion=0.5, varying width ===")
    for width in widths_case_1:
        scenario = f"case1_prop{fixed_prop}_width{width}"
        print(f"tightwin solving {scenario}")
        res = solve_case(fixed_prop, width, scenario, seed=42)
        all_results.append(res)
        for i, t in res['tau'].items():
            tau_records.append({'scenario': scenario, 'node': i, 'tau': t, 'window_width': width, 'tight_proportion': fixed_prop})
        for k, usage in res['capacity_usage'].items():
            cap_records.append({'scenario': scenario, 'truck': k, 'weight': usage['weight'], 'length': usage['length'], 'window_width': width, 'tight_proportion': fixed_prop})

    print("\ntightwin=== CASE 2: fixed width, varying proportion ===")
    for prop in props_case_2:
        scenario = f"case2_prop{prop}_width{fixed_width}"
        print(f"tightwin solving {scenario}")
        res = solve_case(prop, fixed_width, scenario, seed=42)
        all_results.append(res)
        for i, t in res['tau'].items():
            tau_records.append({'scenario': scenario, 'node': i, 'tau': t, 'window_width': fixed_width, 'tight_proportion': prop})
        for k, usage in res['capacity_usage'].items():
            cap_records.append({'scenario': scenario, 'truck': k, 'weight': usage['weight'], 'length': usage['length'], 'window_width': fixed_width, 'tight_proportion': prop})

    # Extra analysis: compact interaction grid to see combined effect of both dimensions
    extra_props = [0.2, 0.5, 0.8]
    extra_widths = [50, 90, 130]
    print("\ntightwin=== EXTRA: interaction mini-grid ===")
    for prop in extra_props:
        for width in extra_widths:
            scenario = f"extra_prop{prop}_width{width}"
            print(f"tightwin solving {scenario}")
            res = solve_case(prop, width, scenario, seed=99)
            all_results.append(res)
            for i, t in res['tau'].items():
                tau_records.append({'scenario': scenario, 'node': i, 'tau': t, 'window_width': width, 'tight_proportion': prop})
            for k, usage in res['capacity_usage'].items():
                cap_records.append({'scenario': scenario, 'truck': k, 'weight': usage['weight'], 'length': usage['length'], 'window_width': width, 'tight_proportion': prop})

    df_res = pd.DataFrame(all_results)
    df_tau = pd.DataFrame(tau_records)
    df_cap = pd.DataFrame(cap_records)

    # Completion time (max tau - depot tau)
    completion_map = {}
    for scenario in df_tau['scenario'].unique():
        tau_s = df_tau[df_tau['scenario'] == scenario]
        max_tau = tau_s['tau'].max()
        depot_tau = tau_s[tau_s['node'] == 0]['tau']
        start_tau = float(depot_tau.iloc[0]) if len(depot_tau) > 0 else 0.0
        completion_map[scenario] = max_tau - start_tau

    df_res['completion'] = df_res['scenario'].map(completion_map)
    df_res['total_wait'] = df_res[['wait_gh_dock', 'wait_gh_service', 'wait_ff']].sum(axis=1, min_count=1)
    df_res['wait_share_pct'] = (df_res['total_wait'] / df_res['obj']) * 100

    # Save all tables
    df_res.to_csv(out_path('tightwin_all_results.csv'), index=False)
    df_tau.to_csv(out_path('tightwin_tau_records.csv'), index=False)
    df_cap.to_csv(out_path('tightwin_capacity_usage.csv'), index=False)

    df_case1 = df_res[df_res['scenario'].str.startswith('case1_')].sort_values('window_width')
    df_case2 = df_res[df_res['scenario'].str.startswith('case2_')].sort_values('tight_proportion')
    df_extra = df_res[df_res['scenario'].str.startswith('extra_')]

    df_case1.to_csv(out_path('tightwin_case1_width_sensitivity.csv'), index=False)
    df_case2.to_csv(out_path('tightwin_case2_proportion_sensitivity.csv'), index=False)
    df_extra.to_csv(out_path('tightwin_extra_interaction_grid.csv'), index=False)

    # Combined time breakdown (WeightCap-style): travel/waits + completion on secondary axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        df_case1['window_width'],
        df_case1['travel'],
        df_case1['wait_gh_dock'],
        df_case1['wait_gh_service'],
        df_case1['wait_ff'],
        labels=['Travel', 'GH pre-dock wait', 'GH dock wait', 'FF wait'],
        colors=['#00bfff', '#ff7f0e', '#d946ef', '#2ca02c']
    )
    ax.set_xlabel('Window width (min)')
    ax.set_ylabel('Travel and waiting times (min)')
    ax.grid(True, linestyle='--', alpha=0.30)
    ax2 = ax.twinx()
    ax2.plot(
        df_case1['window_width'],
        df_case1['completion'],
        color='#8B0000',
        linestyle=':',
        linewidth=2.5,
        marker='o',
        label='Completion time'
    )
    ax2.set_ylabel('Completion time (min)', color='#8B0000')
    ax2.tick_params(axis='y', labelcolor='#8B0000')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path('tightwin_case1_time_breakdown_vs_width.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        df_case2['tight_proportion'],
        df_case2['travel'],
        df_case2['wait_gh_dock'],
        df_case2['wait_gh_service'],
        df_case2['wait_ff'],
        labels=['Travel', 'GH pre-dock wait', 'GH dock wait', 'FF wait'],
        colors=['#00bfff', '#ff7f0e', '#d946ef', '#2ca02c']
    )
    ax.set_xlabel('Tightened-node proportion')
    ax.set_ylabel('Travel and waiting times (min)')
    ax.grid(True, linestyle='--', alpha=0.30)
    ax2 = ax.twinx()
    ax2.plot(
        df_case2['tight_proportion'],
        df_case2['completion'],
        color='#8B0000',
        linestyle=':',
        linewidth=2.5,
        marker='o',
        label='Completion time'
    )
    ax2.set_ylabel('Completion time (min)', color='#8B0000')
    ax2.tick_params(axis='y', labelcolor='#8B0000')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path('tightwin_case2_time_breakdown_vs_proportion.png'))
    plt.close(fig)

    # Plot case 1: trucks used vs width
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_case1['window_width'], df_case1['trucks_used'], marker='o', color='#1f77b4')
    ax.set_xlabel('Window width (min)')
    ax.set_ylabel('Trucks used')
    ax.set_title('Case 1 - Trucks used vs window width')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path('tightwin_case1_trucks_vs_width.png'))
    plt.close(fig)

    # Plot case 2: trucks used vs proportion
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_case2['tight_proportion'], df_case2['trucks_used'], marker='o', color='#1f77b4')
    ax.set_xlabel('Tightened-node proportion')
    ax.set_ylabel('Trucks used')
    ax.set_title('Case 2 - Trucks used vs tightened proportion')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path('tightwin_case2_trucks_vs_proportion.png'))
    plt.close(fig)


    # Extra plot: heatmap of waiting share for interaction grid
    if len(df_extra) > 0 and df_extra['feasible'].any():
        feasible_extra = df_extra[df_extra['feasible']].copy()

        # Trucks used heatmap (extra case)
        heat_trucks = feasible_extra.pivot_table(index='tight_proportion', columns='window_width', values='trucks_used', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(heat_trucks.values, aspect='auto', cmap='Blues')
        ax.set_xticks(range(len(heat_trucks.columns)))
        ax.set_xticklabels([str(int(c)) for c in heat_trucks.columns])
        ax.set_yticks(range(len(heat_trucks.index)))
        ax.set_yticklabels([f"{p:.1f}" for p in heat_trucks.index])
        ax.set_xlabel('Window width (min)')
        ax.set_ylabel('Tightened-node proportion')
        annotate_heatmap_cells(ax, heat_trucks.values, fmt="{:.0f}")
        cbar = fig.colorbar(im, ax=ax)
        truck_min = int(np.nanmin(heat_trucks.values))
        truck_max = int(np.nanmax(heat_trucks.values))
        cbar.set_ticks(list(range(truck_min, truck_max + 1)))
        fig.tight_layout()
        fig.savefig(out_path('tightwin_extra_trucks_heatmap.png'))
        plt.close(fig)

        # Waiting-time component heatmaps (extra case)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        wait_components = [
            ('wait_gh_dock', 'GH pre-dock wait'),
            ('wait_gh_service', 'GH dock wait'),
            ('wait_ff', 'FF wait'),
        ]
        for ax, (col, title) in zip(axes, wait_components):
            heat_comp = feasible_extra.pivot_table(index='tight_proportion', columns='window_width', values=col, aggfunc='mean')
            im = ax.imshow(heat_comp.values, aspect='auto', cmap='YlGnBu')
            ax.set_xticks(range(len(heat_comp.columns)))
            ax.set_xticklabels([str(int(c)) for c in heat_comp.columns])
            ax.set_yticks(range(len(heat_comp.index)))
            ax.set_yticklabels([f"{p:.1f}" for p in heat_comp.index])
            ax.set_xlabel('Width (min)')
            ax.set_ylabel('Proportion')
            annotate_heatmap_cells(ax, heat_comp.values, fmt="{:.1f}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path('tightwin_extra_wait_components_heatmaps.png'))
        plt.close(fig)

        # Objective composition heatmaps (travel, total wait, objective)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        comp_cols = [
            ('travel', 'Travel'),
            ('total_wait', 'Total wait'),
            ('obj', 'Objective (travel + wait)'),
        ]
        for ax, (col, title) in zip(axes, comp_cols):
            heat_comp = feasible_extra.pivot_table(index='tight_proportion', columns='window_width', values=col, aggfunc='mean')
            im = ax.imshow(heat_comp.values, aspect='auto', cmap='OrRd')
            ax.set_xticks(range(len(heat_comp.columns)))
            ax.set_xticklabels([str(int(c)) for c in heat_comp.columns])
            ax.set_yticks(range(len(heat_comp.index)))
            ax.set_yticklabels([f"{p:.1f}" for p in heat_comp.index])
            ax.set_xlabel('Width (min)')
            ax.set_ylabel('Proportion')
            annotate_heatmap_cells(ax, heat_comp.values, fmt="{:.1f}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path('tightwin_extra_obj_composition_heatmaps.png'))
        plt.close(fig)

        heat = feasible_extra.pivot_table(index='tight_proportion', columns='window_width', values='wait_share_pct', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(heat.values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels([str(int(c)) for c in heat.columns])
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels([f"{p:.1f}" for p in heat.index])
        ax.set_xlabel('Window width (min)')
        ax.set_ylabel('Tightened-node proportion')
        annotate_heatmap_cells(ax, heat.values, fmt="{:.1f}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_path('tightwin_extra_waitshare_heatmap.png'))
        plt.close(fig)

    # Final compact report
    report_rows = []
    for _, row in df_res.sort_values(['scenario']).iterrows():
        feasible = bool(row['feasible'])
        report_rows.append({
            'Scenario': row['scenario'],
            'Proportion': f"{row['tight_proportion']:.2f}",
            'Width (min)': int(row['window_width']),
            'Tightened Nodes': int(row['tightened_nodes_count']),
            'Feasible': feasible,
            'Objective (min)': f"{row['obj']:.2f}" if feasible and pd.notna(row['obj']) else 'INFEASIBLE',
            'Travel (min)': f"{row['travel']:.2f}" if feasible and pd.notna(row['travel']) else '-',
            'Total wait (min)': f"{row['total_wait']:.2f}" if feasible and pd.notna(row['total_wait']) else '-',
            'Wait share (%)': f"{row['wait_share_pct']:.1f}" if feasible and pd.notna(row['wait_share_pct']) else '-',
            'Completion (min)': f"{row['completion']:.2f}" if feasible and pd.notna(row['completion']) else '-',
            'Trucks used': int(row['trucks_used']) if feasible and pd.notna(row['trucks_used']) else '-',
        })

    df_report = pd.DataFrame(report_rows)
    df_report.to_csv(out_path('tightwin_sensitivity_report.csv'), index=False)

    print("\ntightwin CASE 1 summary:")
    print(df_case1[['scenario', 'window_width', 'feasible', 'obj', 'completion', 'wait_share_pct']].to_string(index=False))
    print("\ntightwin CASE 2 summary:")
    print(df_case2[['scenario', 'tight_proportion', 'feasible', 'obj', 'completion', 'wait_share_pct']].to_string(index=False))
    print(f"\ntightwin Saved all outputs to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()


