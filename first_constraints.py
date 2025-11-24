from gurobipy import Model, GRB, quicksum
import math

# -------------------------
# BASIC SETS
# -------------------------

# Trucks
K = [0, 1]                     # two trucks

# ULDs and nodes
n = 4                          # 4 ULDs
NP = [1, 2, 3, 4]              # pickup nodes
ND = [5, 6, 7, 8]              # delivery nodes (i+n)
depot = 0
N1 = [depot] + NP + ND         # all nodes = [0..8]

# Feasible arcs: full directed graph except self-loops
E1 = [(i, j) for i in N1 for j in N1 if i != j]

# -------------------------
# FREIGHT FORWARDERS & GROUND HANDLERS
# -------------------------

# 2 freight forwarders, 2 ground handlers
F = [0, 1]                     # FF indices
G = [0, 1]                     # GH indices

# Each FF "owns" 2 pickup nodes
# FF 0: pickups 1, 2
# FF 1: pickups 3, 4
F_nodes = {
    0: {1, 2},
    1: {3, 4},
}

# Each GH "owns" 2 delivery nodes
# GH 0: deliveries 5, 6 (ULDs 1 & 2)
# GH 1: deliveries 7, 8 (ULDs 3 & 4)
G_nodes = {
    0: {5, 6},
    1: {7, 8},
}

# Convenience unions
F_all_nodes = set().union(*F_nodes.values())   # {1,2,3,4}
G_all_nodes = set().union(*G_nodes.values())   # {5,6,7,8}

# -------------------------
# PARAMETERS: travel times, service times, big-M
# -------------------------

# Simple coordinates on a line so distances are easy
coords = {
    0: (0.0, 0.0),   # depot
    1: (1.0, 0.0),
    2: (2.0, 0.0),
    3: (3.0, 0.0),
    4: (4.0, 0.0),
    5: (1.5, 0.0),
    6: (2.5, 0.0),
    7: (3.5, 0.0),
    8: (4.5, 0.0),
}

# Travel time T[i,j] = 10 * Euclidean distance
T = {}
for (i, j) in E1:
    xi, yi = coords[i]
    xj, yj = coords[j]
    dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
    T[(i, j)] = 10.0 * dist

# Service time at each pickup/delivery node
P = {i: 2.0 for i in NP + ND}

# Big-M for time constraints
BIG_M = 1e4

# First eleven constraints

# (1) each pickup node is visited exactly once
for i in NP:
    model.addConstr(
        quicksum(x[k, j, i]
                 for k in K
                 for j in N1
                 if (j, i) in E1) == 1,
        name=f"pickup_node_visted_once_{i}"
    )

# (2) pickup and delivery of each ULD are done by the same truck
n = len(NP)  # NP = [1,...,n]

for k in K:
    for i in NP:
        delivery = i + n
        model.addConstr(
            quicksum(x[k, j, i]       for j in N1 if (j, i)       in E1) -
            quicksum(x[k, j, delivery] for j in N1 if (j, delivery) in E1)
            == 0,
            name=f"same_truck_pick_deliv_i{i}_k{k}"
        )


# (3) each truck leaves the depot at most once
depot = 0

for k in K:
    model.addConstr(
        quicksum(x[k, depot, i] for i in N1 if (depot, i) in E1) <= 1,
        name=f"use_truck_at_most_once_k{k}"
    )

# (4) what enters a node equals what leaves it (except depot, where this just allows start/end).
for k in K:
    for i in N1:
        model.addConstr(
            quicksum(x[k, j, i] for j in N1 if (j, i) in E1) -
            quicksum(x[k, i, j] for j in N1 if (i, j) in E1)
            == 0,
            name=f"flow_conservation_at_each_node_k{k}_i{i}"
        )


# (5) LIFO strategy for first and last nodes visisted by each truck
for k in K:
    for i in NP:
        delivery = i + n
        model.addConstr(
            x[k, depot, i] - x[k, delivery, depot] == 0,
            name=f"LIFO_first_last_i{i}_k{k}"
        )

# (6) LIFO strategy for intermediate nodes
for k in K:
    for (i, j) in E1:
        if i in NP and j in NP:  # both are pickup nodes
            delivery_i = i + n
            delivery_j = j + n
            model.addConstr(
                x[k, i, j] - x[k, delivery_j, delivery_i] == 0,
                name=f"LIFO_reverse_i{i}_j{j}_k{k}"
            )

# (7) each freight forwarder's nodes are visited at most once by each truck
for k in K:
    for f in F:
        nodes_f   = F_nodes[f]                 # math: 𝓕_f
        outside_f = F_all_nodes - nodes_f      # math: 𝓕 \ {𝓕_f}

        model.addConstr(
             quicksum(x[k, j, i] for i in nodes_f for j in outside_f if (j, i) in E1)
             +quicksum(x[k, depot, i] for i in nodes_f if (depot, i) in E1)<= 1,
            name=f"visit_FF_at_most_once_f{f}_k{k}"
        )

# (8) each ground handler's nodes are visited at most once by each truck
for k in K:
    for g in G:
        nodes_g   = G_nodes[g]                 # math: 𝓖_g
        outside_g = G_all_nodes - nodes_g      # math: 𝓖 \ {𝓖_g}

        model.addConstr(
             quicksum(x[k, j, i] for i in nodes_g for j in outside_g if (j, i) in E1)
             +quicksum(x[k, depot, i] for i in nodes_g if (depot, i) in E1)<= 1,
            name=f"visit_GH_at_most_once_g{g}_k{k}"
        )

# (9) time precedence for pickup nodes

for (i, j) in E1:
    if j in NP:  # j is a pickup node (the one following i)
        model.addConstr(
            tau[j] >= tau[i] + P[i] + T[i, j]
                      - BIG_M * (1 - quicksum(x[k, i, j] for k in K)),
            name=f"time_pickups_i{i}_j{j}"
        )

# (10) time precedence for entering a GH
for g in G:
    nodes_g = G_nodes[g]  # delivery nodes of GH g
    for k in K:
        for (i, j) in E1:
            # j is a delivery node in GH g, and i is outside GH g
            if j in nodes_g and i not in nodes_g and j in ND:
                model.addConstr(
                    tau[j] >= tau[i] + P[i] + T[i, j] \
                              - BIG_M * (1 - x[k, i, j]) \
                              + w_D[k, g],
                    name=f"time_enter_GH_g{g}_k{k}_i{i}_j{j}"
                )
