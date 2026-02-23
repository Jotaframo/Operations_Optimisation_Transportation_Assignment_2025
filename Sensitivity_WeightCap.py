# Loading packages that are used in the code
import numpy as np
import os
import pandas as pd
from gurobipy import Model,GRB
import gurobipy as gp
from math import radians, cos, sin, asin, sqrt
import random
from typing import Dict, List

# Get path to current folder
cwd = os.getcwd()

# Constants that will not change between runs
M = 10000  # Big M for time constraints

tighter_windows_instance = 0  # proportion of nodes with tightened time windows
Delta_GH = 1  # number of docks per GH
Docks = list(range(1, Delta_GH + 1))
n_uld = 6  # number of ULDs
K_trucks = [1, 2, 3]  # truck indices
Weight_u = 1000  # weight of each ULD [kg]
Length_u = 1.534  # length of each ULD [m]
Proc_Time = 15  # processing time [minutes]
Horizon = 480  # total time limit [minutes]
Cap_L = 13.6  # length capacity of each truck [m]
Speed_kmh = 35
Speed_mpm = Speed_kmh / 60.0  # km per minute

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
E_win = {i: 0 for i in All_Nodes}
D_win = {i: Horizon for i in All_Nodes}
W = {i: (Weight_u if i in Nodes_P else 0) for i in All_Nodes}
L = {i: (Length_u if i in Nodes_P else 0) for i in All_Nodes}

# time windows tightening (runs once)
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
        if i - n_uld in tightened_P_windows:
            D_win[i] += 30


def solve_case(cap_w: float) -> dict:
    """Build & solve model for a given weight capacity. Returns results."""
    m = gp.Model(f"GHDC-PDPTW_Cap{cap_w}")

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
    # [15–16] capacity constraints use cap_w
    for k in K_trucks:
        load_expr = gp.quicksum(W[i] * x[j, i, k]
                                for i in All_Nodes
                                for j in All_Nodes if (j, i) in Edges)
        m.addConstr(load_expr <= cap_w)
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

    result = {
        'Cap_W': cap_w,
        'obj': m.objVal if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else None,
        'travel': travel_cost.getValue() if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else None,
        'wait_gh_dock': wait_gh_dock_cost.getValue() if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else None,
        'wait_gh_service': wait_gh_service_cost.getValue() if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else None,
        'wait_ff': wait_ff_cost.getValue() if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else None,
    }

    routes = {}
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
    plot_routes(routes, node_loc_maps, FFs, GHs, output_path=f"captruck_routes_cap{cap_w}.png")

    trucks_used = 0
    cap_usage = {}
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
    result['capacity_usage'] = cap_usage

    result['tau'] = {i: val(tau[i]) for i in All_Nodes}

    return result


def main():
    cap_values = [2000, 3000, 4000, 5000, 6000]
    results = []
    tau_records = []
    cap_records = []
    for cap in cap_values:
        print(f"captruck \n=== solving for Cap_W={cap} ===")
        res = solve_case(cap)
        results.append(res)
        for i, t in res['tau'].items():
            tau_records.append({'Cap_W': cap, 'node': i, 'tau': t})
        for k, usage in res['capacity_usage'].items():
            cap_records.append({'Cap_W': cap, 'truck': k, 'weight': usage['weight'], 'length': usage['length']})

    df_res = pd.DataFrame(results)
    df_tau = pd.DataFrame(tau_records)
    df_cap = pd.DataFrame(cap_records)

    print("\nSummary results:")
    print(df_res[['Cap_W', 'obj', 'travel', 'wait_gh_dock', 'wait_gh_service', 'wait_ff', 'trucks_used']])

    import matplotlib.pyplot as plt
    # (objective curve is shown in the breakdown plot, so skip separate graph)
    # fig, ax = plt.subplots()
    # ax.plot(df_res['Cap_W'], df_res['obj'], marker='o')
    # ax.set_xlabel('Weight capacity (kg)')
    # ax.set_ylabel('Objective value')
    # fig.savefig('captruck_obj_vs_cap.png')
    # print('captruck Saved captruck_obj_vs_cap.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(df_res['Cap_W'], df_res['travel'], df_res['wait_gh_dock'], df_res['wait_gh_service'], df_res['wait_ff'], labels=['travel', 'wait_gh_dock', 'wait_gh_service', 'wait_ff'], colors=['#00bfff', '#ff7f0e', '#d946ef', '#2ca02c'])
    ax.set_xlabel('Weight capacity (kg)')
    ax.set_ylabel('Objective value', color='black')
    ax.tick_params(axis='y', labelcolor='black')
    
    # Secondary y-axis for completion time
    max_taus = []
    for cap in df_res['Cap_W']:
        tau_data = df_tau[df_tau['Cap_W'] == cap]
        max_taus.append(tau_data['tau'].max())
    
    ax2 = ax.twinx()
    ax2.plot(df_res['Cap_W'], max_taus, color='#8B0000', linestyle=':', linewidth=2.5, marker='o', label='Completion time')
    ax2.set_ylabel('Completion time (min)', color='#8B0000')
    ax2.tick_params(axis='y', labelcolor='#8B0000')
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    
    fig.tight_layout()
    fig.savefig('captruck_breakdown_vs_cap.png')
    print('captruck Saved captruck_breakdown_vs_cap.png')

    fig, ax = plt.subplots()
    ax.plot(df_res['Cap_W'], df_res['trucks_used'], marker='o')
    ax.set_xlabel('Weight capacity (kg)')
    ax.set_ylabel('Trucks used')
    ax.legend(loc='upper left')
    fig.savefig('captruck_trucks_vs_cap.png')
    print('captruck Saved captruck_trucks_vs_cap.png')

    pivot = df_tau.pivot(index='node', columns='Cap_W', values='tau')
    pivot.to_csv('captruck_taus_per_node.csv')
    print('captruck Saved captruck_taus_per_node.csv')

    df_cap.to_csv('captruck_capacity_usage.csv', index=False)
    print('captruck Saved captruck_capacity_usage.csv')
    
    # Generate comprehensive sensitivity report
    report = []
    for idx, row in df_res.iterrows():
        cap = row['Cap_W']
        cap_data = df_cap[df_cap['Cap_W'] == cap]
        
        # Calculate average capacity usage
        avg_weight_usage = cap_data['weight'].sum() / len(K_trucks) if len(cap_data) > 0 else 0
        weight_util_pct = (avg_weight_usage / cap) * 100 if cap > 0 else 0
        
        # Get tau stats (service times)
        tau_data = df_tau[df_tau['Cap_W'] == cap]
        tau_vals = tau_data['tau'].values
        max_tau = tau_vals.max() if len(tau_vals) > 0 else 0
        
        report.append({
            'Cap_W (kg)': int(cap),
            'Trucks Used': int(row['trucks_used']),
            'Obj Value': f"{row['obj']:.2f}",
            'Travel (min)': f"{row['travel']:.2f}",
            'GH Pre-Dock Wait (min)': f"{row['wait_gh_dock']:.2f}",
            'GH Dock Wait (min)': f"{row['wait_gh_service']:.2f}",
            'FF Dock Wait (min)': f"{row['wait_ff']:.2f}",
            'Avg Weight Usage (kg)': f"{avg_weight_usage:.1f}",
            'Weight Util (%)': f"{weight_util_pct:.1f}",
            'Max Service Time (min)': f"{max_tau:.2f}",
        })
    
    df_report = pd.DataFrame(report)
    df_report.to_csv('captruck_sensitivity_report.csv', index=False)
    print('captruck Saved captruck_sensitivity_report.csv')
    
    print('\ncaptruck SENSITIVITY REPORT:')
    print(df_report.to_string(index=False))
    print('\ncaptruck Finished sensitivity analysis')


if __name__ == '__main__':
    main()


