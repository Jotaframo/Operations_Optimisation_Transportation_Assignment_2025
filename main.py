# Loading packages that are used in the code
import numpy as np
import os
import pandas as pd
from gurobipy import Model,GRB,LinExpr
import gurobipy as gp
from math import radians, cos, sin, asin, sqrt

# Get path to current folder
cwd = os.getcwd()

# Get all instances
full_list           = os.listdir(cwd)
model=Model()
M=86400 #[s] in a day
# --- 1. DATA INPUT & PROCESSING ---

# ULD & Truck Specs
tighter_windows_instance=0.2
Delta_GH = 1 # Number of docks per GH (Assuming 'Very Large' instance setting or standard)
Docks = list(range(1, Delta_GH + 1)) # Set of Docks
n_uld = 8
K_trucks = [1, 2] # Two trucks
Weight_u = 1000   # kg
Length_u = 1.534  # meters (Converted 153.4cm to m)
Proc_Time = 2     # minutes
Horizon = 480     # minutes
Cap_W = 10000     # kg
Cap_L = 13.6      # meters
Speed_kmh = 35    # km/h
Speed_mpm = 35 / 60.0 # km per minute
# Coordinates (DMS to Decimal Degrees)
def dms_to_dd(d, m, s, direction='N'):
    dd = d + m/60 + s/3600
    if direction in ['S', 'W']: dd *= -1
    return dd

# Locations
locs = {
    'FF1': (dms_to_dd(52,17,46.8), dms_to_dd(4,46,10.4)),
    'FF2': (dms_to_dd(52,18,6.8),  dms_to_dd(4,45,3.2)),
    'GH1': (dms_to_dd(52,17,0.8),  dms_to_dd(4,46,7.1)),
    'GH2': (dms_to_dd(52,16,32.9), dms_to_dd(4,44,30.0))
}

# Centroid for Depot (Node 0) - Calculated average of above
locs['Depot'] = (52.2905, 4.7627) 

# Haversine Distance Function
def get_dist(coord1, coord2):
    lat1, lon1, lat2, lon2 = map(radians, [coord1[0], coord1[1], coord2[0], coord2[1]])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

# --- NODE MAPPING ---
# 0: Depot
# 1-4: Pickups at FF1
# 5-8: Pickups at FF2
# 9-12: Deliveries at GH1 (Corresponding to 1-4)
# 13-16: Deliveries at GH2 (Corresponding to 5-8)

Nodes_P = list(range(1, 9))
Nodes_D = list(range(9, 17))
All_Nodes = [0] + Nodes_P + Nodes_D
Edges = [(i, j) for i in  All_Nodes for j in  All_Nodes if i != j]

# Parameter Dictionaries
T = {} # Travel Time
P = {0: 0} # Processing Time
E_win = {0: 0} # Earliest Time
D_win = {0: Horizon} # Latest Time
W = {0: 0} # Weights
L = {0: 0} # Lengths

# Map nodes to physical locations to calc distances
node_loc_map = {0: locs['Depot']}

for i in range(1, 5): 
    node_loc_map[i] = locs['FF1']; node_loc_map[i+8] = locs['GH1']
for i in range(5, 9): 
    node_loc_map[i] = locs['FF2']; node_loc_map[i+8] = locs['GH2']


# Fill Parameters
for i in  All_Nodes:
    # Processing Time
    if i != 0: P[i] = Proc_Time
    
    # Weights & Lengths (Only for Pickups, 0 for deliveries to avoid double counting in cap constraints)
    if i in Nodes_P:
        W[i] = Weight_u
        L[i] = Length_u
    elif i in Nodes_D:
        W[i] = 0
        L[i] = 0
        
    # Time Windows
    tightened_P_windows = []
    if i in Nodes_P[:int(tighter_windows_instance*len(Nodes_P))]:
        E_win[i] = 50; D_win[i] = 150
        tightened_P_windows.append(i)
    elif i in Nodes_D[:int(tighter_windows_instance*len(Nodes_D))]:
        E_win[i] = 200; D_win[i] = 350
        if i-n_uld in tightened_P_windows:
            D_win[i]+= 30 # Includes 30 min buffer if needed per paper????? 
    else:
        E_win[i] = 0; D_win[i] = 480


# Calculate Travel Matrix (T_ij)
for i, j in Edges:
    dist_km = get_dist(node_loc_map[i], node_loc_map[j])
    T[i,j] = dist_km / Speed_mpm

# Facility and Group Sets
# FFs (Pickups)
Facilities = {
    1: [1, 2, 3, 4], # FF1
    2: [5, 6, 7, 8]  # FF2
}
# GHs (Deliveries)
Groups = {
    1: [9, 10, 11, 12],  # GH1
    2: [13, 14, 15, 16]  # GH2
}



m = gp.Model("GHDC-PDPTW")

# --- VARIABLES ---
x = m.addVars(Edges, K_trucks, vtype=GRB.BINARY, name="x")
tau = m.addVars( All_Nodes, vtype=GRB.CONTINUOUS, name="tau")
tau_end = m.addVars(K_trucks, vtype=GRB.CONTINUOUS, name="tau_end")

# Facility/Group Arrival/Departure times (indexed by Truck, Facility/Group ID)
a_F = m.addVars(K_trucks, Facilities.keys(), vtype=GRB.CONTINUOUS, name="a_F")
d_F = m.addVars(K_trucks, Facilities.keys(), vtype=GRB.CONTINUOUS, name="d_F")
a_G = m.addVars(K_trucks, Groups.keys(), vtype=GRB.CONTINUOUS, name="a_G")
# d_G is used in later constraints (23+), but required if we were doing the full model
d_G = m.addVars(K_trucks, Groups.keys(), vtype=GRB.CONTINUOUS, name="d_G") 


# Waiting times
w_D = m.addVars(K_trucks, Groups.keys(), vtype=GRB.CONTINUOUS, name="w_D") # Waiting at GH before docking
w_F = m.addVars(K_trucks, Facilities.keys(), vtype=GRB.CONTINUOUS, name="w_F") # Waiting at FF while docked
w_G = m.addVars(K_trucks, Groups.keys(), vtype=GRB.CONTINUOUS, name="w_G") # Waiting at GH while docked

# Scheduling and Dock Assignment variables
# eta: 1 if k1 leaves g before k2 docks
eta = m.addVars(K_trucks, K_trucks, Groups.keys(), vtype=GRB.BINARY, name="eta") 
# y: 1 if truck k assigned to dock d at GH g
y = m.addVars(K_trucks, Docks, Groups.keys(), vtype=GRB.BINARY, name="y")
# z: Linearization variable for y*y. Relaxed to continuous [0,1] as per paper Section 3.3
z = m.addVars(K_trucks, Docks, K_trucks, Docks, Groups.keys(), vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")

# (1) each pickup node is visited exactly once
for i in Nodes_P:
    m.addConstr(
        gp.quicksum(x[j, i, k]
                 for k in K_trucks
                 for j in All_Nodes
                 if (j, i) in Edges) == 1,
        name=f"pickup_node_visted_once_{i}"
    )

# (2) pickup and delivery of each ULD are done by the same truck
n = len(Nodes_P)  # Nodes_P = [1,...,n]

for k in K_trucks:
    for i in Nodes_P:
        delivery = i + n
        m.addConstr(
            gp.quicksum(x[j, i, k]       for j in All_Nodes if (j, i)       in Edges) -
            gp.quicksum(x[j, delivery, k] for j in All_Nodes if (j, delivery) in Edges)
            == 0,
            name=f"same_truck_pick_deliv_i{i}_k{k}"
        )


# (3) each truck leaves the depot at most once
depot = 0

for k in K_trucks:
    m.addConstr(
        gp.quicksum(x[depot, i, k] for i in All_Nodes if (depot, i) in Edges) <= 1,
        name=f"use_truck_at_most_once_k{k}"
    )

# (4) what enters a node equals what leaves it (except depot, where this just allows start/end).
for k in K_trucks:
    for i in All_Nodes:
        m.addConstr(
            gp.quicksum(x[j, i, k] for j in All_Nodes if (j, i) in Edges) -
            gp.quicksum(x[i, j, k] for j in All_Nodes if (i, j) in Edges)
            == 0,
            name=f"flow_conservation_at_each_node_k{k}_i{i}"
        )


# (5) LIFO strategy for first and last nodes visisted by each truck
for k in K_trucks:
    for i in Nodes_P:
        delivery = i + n
        m.addConstr(
            x[depot, i, k] - x[delivery, depot, k] == 0,
            name=f"LIFO_first_last_i{i}_k{k}"
        )

# (6) LIFO strategy for intermediate nodes
for k in K_trucks:
    for (i, j) in Edges:
        if i in Nodes_P and j in Nodes_P:  # both are pickup nodes
            delivery_i = i + n
            delivery_j = j + n
            m.addConstr(
                x[i, j, k] - x[delivery_j, delivery_i, k] == 0,
                name=f"LIFO_reverse_i{i}_j{j}_k{k}"
            )

# (7) each freight forwarder's nodes are visited at most once by each truck
for k in K_trucks:
    for f in Facilities:
        nodes_f   = Facilities[f]                 # math: 𝓕_f
        outside_f = set(All_Nodes) - set(nodes_f)      # math: 𝓕 \ {𝓕_f}

        m.addConstr(
             gp.quicksum(x[j, i, k] for i in nodes_f for j in outside_f if (j, i) in Edges)
             + gp.quicksum(x[depot, i, k] for i in nodes_f if (depot, i) in Edges) <= 1,
            name=f"visit_FF_at_most_once_f{f}_k{k}"
        )

# (8) each ground handler's nodes are visited at most once by each truck
for k in K_trucks:
    for g in Groups:
        nodes_g   = Groups[g]                 # math: 𝓖_g
        outside_g = set(All_Nodes) - set(nodes_g)      # math: 𝓖 \ {𝓖_g}

        m.addConstr(
             gp.quicksum(x[j, i, k] for i in nodes_g for j in outside_g if (j, i) in Edges)
             + gp.quicksum(x[depot, i, k] for i in nodes_g if (depot, i) in Edges) <= 1,
            name=f"visit_GH_at_most_once_g{g}_k{k}"
        )

# (9) time precedence for pickup nodes

for (i, j) in Edges:
    if j in Nodes_P:  # j is a pickup node (the one following i)
        m.addConstr(
            tau[j] >= tau[i] + P[i] + T[i, j]
                      - M * (1 - gp.quicksum(x[i, j, k] for k in K_trucks)),
            name=f"time_pickups_i{i}_j{j}"
        )

# (10) time precedence for entering a GH
for g in Groups:
    nodes_g = Groups[g]  # delivery nodes of GH g
    for k in K_trucks:
        for (i, j) in Edges:
            # j is a delivery node in GH g, and i is outside GH g
            if j in nodes_g and i not in nodes_g and j in Nodes_D:
                m.addConstr(
                    tau[j] >= tau[i] + P[i] + T[i, j] \
                              - M * (1 - x[i, j, k]) \
                              + w_D[k, g],
                    name=f"time_enter_GH_g{g}_k{k}_i{i}_j{j}"
                )

# --- CONSTRAINTS 12 to 22 ---

# (12) Time consistency within GH Groups
# Ensures valid flow of time when moving between nodes inside the same Ground Handler

for g, nodes in Groups.items(): #i and j belonging to the same GH
    for i in nodes:
        for j in nodes:
            if i != j and (i, j) in Edges:
                # Sum x over all trucks (any truck making this move)
                expr =  gp.quicksum(x[i, j, k] for k in K_trucks)
                m.addConstr(tau[j] >= tau[i] + P[i] + T[i,j] - (1 - expr)*M, 
                            name=f"Eq12_IntraGroupTime_{i}_{j}")

# (13) Vehicle End Time
# Links the last node visited to the depot arrival time
for k in K_trucks:
    for i in  All_Nodes:
        if i != 0: # For all nodes going to depot
            m.addConstr(tau_end[k] >= tau[i] + P[i] + T[i,0] - (1 - x[i, 0, k])*M,
                        name=f"Eq13_EndTime_{k}_{i}")

# (14) Time Windows
# Hard constraints on Earliest and Latest service times
for i in Nodes_P + Nodes_D:
    m.addConstr(tau[i] >= E_win[i], name=f"Eq14_TW_LB_{i}")
    m.addConstr(tau[i] <= D_win[i], name=f"Eq14_TW_UB_{i}")

# (15) Weight Capacity
# Total weight of Pickups visited by truck k must not exceed Q_W
for k in K_trucks:
    # Summing flow into pickup nodes (i <= n)
    load_expr =  gp.quicksum(W[i] * x[j, i, k] 
                            for i in  All_Nodes 
                            for j in  All_Nodes if (j, i) in Edges)
    m.addConstr(load_expr <= Cap_W, name=f"Eq15_WeightCap_{k}")

# (16) Length Capacity
# Total length of Pickups visited by truck k must not exceed Q_L
for k in K_trucks:
    len_expr =  gp.quicksum(L[i] * x[j, i, k] 
                           for i in  All_Nodes 
                           for j in  All_Nodes if (j, i) in Edges)
    m.addConstr(len_expr <= Cap_L, name=f"Eq16_LengthCap_{k}")

# --- Linearization of Arrival/Departure Times (17-22) ---

# (17 & 18) Facility Arrival Time (a_F)
# If truck k travels i -> j, and j is in Facility f but i is NOT, record arrival time.
for f, f_nodes in Facilities.items():
    for k in K_trucks:
        for i, j in Edges:
            if j in f_nodes and i not in f_nodes:
                # (17) Lower Bound
                m.addConstr(a_F[k,f] >= tau[i] + P[i] + T[i,j] - (1 - x[i,j,k])*M,
                            name=f"Eq17_ArrF_LB_{k}_{f}")
                # (18) Upper Bound
                m.addConstr(a_F[k,f] <= tau[i] + P[i] + T[i,j] + (1 - x[i,j,k])*M,
                            name=f"Eq18_ArrF_UB_{k}_{f}")

# (19 & 20) Facility Departure Time (d_F)
# If truck k travels i -> j, and i is in Facility f but j is NOT, record departure time.
for f, f_nodes in Facilities.items():
    for k in K_trucks:
        for i, j in Edges:
            if i in f_nodes and j not in f_nodes:
                # (19) Lower Bound
                m.addConstr(d_F[k,f] >= tau[i] + P[i] - (1 - x[i,j,k])*M,
                            name=f"Eq19_DepF_LB_{k}_{f}")
                # (20) Upper Bound
                m.addConstr(d_F[k,f] <= tau[i] + P[i] + (1 - x[i,j,k])*M,
                            name=f"Eq20_DepF_UB_{k}_{f}")

# (21 & 22) Group Arrival Time (a_G)
# If truck k travels i -> j, and j is in Group g but i is NOT, record arrival time.
for g, g_nodes in Groups.items():
    for k in K_trucks:
        for i, j in Edges:
            if j in g_nodes and i not in g_nodes:
                # (21) Lower Bound
                m.addConstr(a_G[k,g] >= tau[i] + P[i] + T[i,j] - (1 - x[i,j,k])*M,
                            name=f"Eq21_ArrG_LB_{k}_{g}")
                # (22) Upper Bound
                m.addConstr(a_G[k,g] <= tau[i] + P[i] + T[i,j] + (1 - x[i,j,k])*M,
                            name=f"Eq22_ArrG_UB_{k}_{g}")


# --- CONSTRAINTS (23 to 33) ---

# (23) Departure time from GH (Lower Bound)
# Ensures departure time >= start service of the last node served in that GH + processing
for g in Groups:
    for k in K_trucks:
        for i in Groups[g]: # Node in GH
            for j in  All_Nodes: # Possible next node
                if j not in Groups[g] and (i,j) in Edges:
                     m.addConstr(d_G[k, g] >= tau[i] + P[i] - (1 - x[i, j, k]) * M, 
                                 name=f"C23_DepGH_LB_k{k}_g{g}_i{i}")

# (24) Departure time from GH (Upper Bound)
# Tightens the departure time definition
for g in Groups:
    for k in K_trucks:
        for i in Groups[g]: 
            for j in  All_Nodes:
                if j not in Groups[g] and (i,j) in Edges:
                     m.addConstr(d_G[k, g] <= tau[i] + P[i] + (1 - x[i, j, k]) * M, 
                                 name=f"C24_DepGH_UB_k{k}_g{g}_i{i}")

# (25) Waiting time at FF while docked
# Waiting >= Departure - Arrival - Total Processing at that FF
for f in Facilities:
    for k in K_trucks:
        # Sum of processing times for all nodes visited by k in FF f
        proc_sum =  gp.quicksum(P[i] * x[j, i, k] for i in Facilities[f] for j in  All_Nodes if (j, i) in Edges)
        m.addConstr(w_F[k, f] >= d_F[k, f] - a_F[k, f] - proc_sum, 
                    name=f"C25_WaitFF_k{k}_f{f}")

# (26) Waiting time at GH while docked
# Waiting >= Departure - Arrival - Waiting before docking - Total Processing at that GH
for g in Groups:
    for k in K_trucks:
        proc_sum =  gp.quicksum(P[i] * x[j, i, k] for i in Groups[g] for j in  All_Nodes if (j, i) in Edges)
        m.addConstr(w_G[k, g] >= d_G[k, g] - a_G[k, g] - w_D[k, g] - proc_sum, 
                    name=f"C26_WaitGH_k{k}_g{g}")

# (27) Overlap variable definition (Lower Bound logic)
# If eta=1, then Departure of k1 <= Arrival of k2 + Waiting of k2 (k1 leaves before k2 uses dock)
for g in Groups:
    for k1 in K_trucks:
        for k2 in K_trucks:
            m.addConstr(d_G[k1, g] - a_G[k2, g] - w_D[k2, g] + M * eta[k1, k2, g] <= M,
                        name=f"C27_OverlapLB_k{k1}_k{k2}_g{g}")

# (28) Overlap variable definition (Upper Bound logic)
# Enforces the binary nature relative to the time difference
for g in Groups:
    for k1 in K_trucks:
        for k2 in K_trucks:
            m.addConstr(-d_G[k1, g] + a_G[k2, g] + w_D[k2, g] - M * eta[k1, k2, g] <= 0,
                        name=f"C28_OverlapUB_k{k1}_k{k2}_g{g}")

# (29) Dock Assignment
# If a truck visits a GH, it must be assigned to exactly one dock.
# RHS sums all arcs entering the GH g from outside nodes (including pickup nodes i-n)
for g in Groups:
    for k in K_trucks:
        # Calculate if truck k visits GH g: Sum of arcs entering g from outside
        visits_gh =  gp.quicksum(x[j, i, k] for i in Groups[g] for j in  All_Nodes if (j, i) in Edges and j not in Groups[g])
        
        m.addConstr( gp.quicksum(y[k, d, g] for d in Docks) == visits_gh,
                    name=f"C29_AssignDock_k{k}_g{g}")

# (30) Linearization of z (Part 1)
# z <= y1
for g in Groups:
    for k1 in K_trucks:
        for k2 in K_trucks:
            if k1 < k2:
                for d1 in Docks:
                    for d2 in Docks:
                        m.addConstr(z[k1, d1, k2, d2, g] <= y[k1, d1, g],
                                    name=f"C30_LinZ1_k{k1}_k{k2}_d{d1}_d{d2}_g{g}")

# (31) Linearization of z (Part 2)
# z <= y2
for g in Groups:
    for k1 in K_trucks:
        for k2 in K_trucks:
            if k1 < k2:
                for d1 in Docks:
                    for d2 in Docks:
                        m.addConstr(z[k1, d1, k2, d2, g] <= y[k2, d2, g],
                                    name=f"C31_LinZ2_k{k1}_k{k2}_d{d1}_d{d2}_g{g}")

# (32) Linearization of z (Part 3)
# y1 + y2 - 1 <= z (Forces z=1 if both y1 and y2 are 1)
for g in Groups:
    for k1 in K_trucks:
        for k2 in K_trucks:
            if k1 < k2:
                for d1 in Docks:
                    for d2 in Docks:
                        m.addConstr(y[k1, d1, g] + y[k2, d2, g] - 1 <= z[k1, d1, k2, d2, g],
                                    name=f"C32_LinZ3_k{k1}_k{k2}_d{d1}_d{d2}_g{g}")

# (33) Dock Capacity / Non-overlap constraint
# Two trucks can be assigned to the SAME dock (d1) only if they do not overlap in time.
# If z=1 (both on same dock), then eta[k1,k2] + eta[k2,k1] must be >= 1 (one precedes the other)
for g in Groups:
    for k1 in K_trucks:
        for k2 in K_trucks:
            if k1 < k2:
                for d1 in Docks:
                    # Note: Indices for z are k1, d1, k2, d1 (same dock)
                    m.addConstr(z[k1, d1, k2, d1, g] <= eta[k1, k2, g] + eta[k2, k1, g],
                                name=f"C33_NoOverlap_k{k1}_k{k2}_d{d1}_g{g}")

m.update()

print("tabien")