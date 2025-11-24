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
Edges = [(i, j) for i in All_Nodes for j in All_Nodes if i != j]

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
for i in All_Nodes:
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
    if i in Nodes_P:
        E_win[i] = 50; D_win[i] = 150
    elif i in Nodes_D:
        E_win[i] = 200; D_win[i] = 350 # Includes 30 min buffer if needed per paper

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
tau = m.addVars(All_Nodes, vtype=GRB.CONTINUOUS, name="tau")
tau_end = m.addVars(K_trucks, vtype=GRB.CONTINUOUS, name="tau_end")

# Facility/Group Arrival/Departure times (indexed by Truck, Facility/Group ID)
a_F = m.addVars(K_trucks, Facilities.keys(), vtype=GRB.CONTINUOUS, name="a_F")
d_F = m.addVars(K_trucks, Facilities.keys(), vtype=GRB.CONTINUOUS, name="d_F")
a_G = m.addVars(K_trucks, Groups.keys(), vtype=GRB.CONTINUOUS, name="a_G")
# d_G is used in later constraints (23+), but required if we were doing the full model
d_G = m.addVars(K_trucks, Groups.keys(), vtype=GRB.CONTINUOUS, name="d_G") 

# --- CONSTRAINTS 12 to 22 ---

# (12) Time consistency within GH Groups
# Ensures valid flow of time when moving between nodes inside the same Ground Handler

for g, nodes in Groups.items(): #i and j belonging to the same GH
    for i in nodes:
        for j in nodes:
            if i != j and (i, j) in Edges:
                # Sum x over all trucks (any truck making this move)
                expr = gp.quicksum(x[i, j, k] for k in K_trucks)
                m.addConstr(tau[j] >= tau[i] + P[i] + T[i,j] - (1 - expr)*M, 
                            name=f"Eq12_IntraGroupTime_{i}_{j}")

# (13) Vehicle End Time
# Links the last node visited to the depot arrival time
for k in K_trucks:
    for i in All_Nodes:
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
    load_expr = gp.quicksum(W[i] * x[j, i, k] 
                            for i in All_Nodes 
                            for j in All_Nodes if (j, i) in Edges)
    m.addConstr(load_expr <= Cap_W, name=f"Eq15_WeightCap_{k}")

# (16) Length Capacity
# Total length of Pickups visited by truck k must not exceed Q_L
for k in K_trucks:
    len_expr = gp.quicksum(L[i] * x[j, i, k] 
                           for i in All_Nodes 
                           for j in All_Nodes if (j, i) in Edges)
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

m.update()
print("Constraints 12-22 generated successfully.")