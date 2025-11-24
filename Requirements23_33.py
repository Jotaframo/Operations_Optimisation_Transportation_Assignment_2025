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

Delta_GH = 1 # Number of docks per GH (Assuming 'Very Large' instance setting or standard)
Docks = list(range(1, Delta_GH + 1)) # Set of Docks

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


# Missing variables added below for M1 model
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

# --- CONSTRAINTS (23 to 33) ---

# (23) Departure time from GH (Lower Bound)
# Ensures departure time >= start service of the last node served in that GH + processing
for g in Groups:
    for k in K_trucks:
        for i in Groups[g]: # Node in GH
            for j in All_Nodes: # Possible next node
                if j not in Groups[g] and (i,j) in Edges:
                     m.addConstr(d_G[k, g] >= tau[i] + P[i] - (1 - x[i, j, k]) * M, 
                                 name=f"C23_DepGH_LB_k{k}_g{g}_i{i}")

# (24) Departure time from GH (Upper Bound)
# Tightens the departure time definition
for g in Groups:
    for k in K_trucks:
        for i in Groups[g]: 
            for j in All_Nodes:
                if j not in Groups[g] and (i,j) in Edges:
                     m.addConstr(d_G[k, g] <= tau[i] + P[i] + (1 - x[i, j, k]) * M, 
                                 name=f"C24_DepGH_UB_k{k}_g{g}_i{i}")

# (25) Waiting time at FF while docked
# Waiting >= Departure - Arrival - Total Processing at that FF
for f in Facilities:
    for k in K_trucks:
        # Sum of processing times for all nodes visited by k in FF f
        proc_sum = gp.quicksum(P[i] * x[j, i, k] for i in Facilities[f] for j in All_Nodes if (j, i) in Edges)
        m.addConstr(w_F[k, f] >= d_F[k, f] - a_F[k, f] - proc_sum, 
                    name=f"C25_WaitFF_k{k}_f{f}")

# (26) Waiting time at GH while docked
# Waiting >= Departure - Arrival - Waiting before docking - Total Processing at that GH
for g in Groups:
    for k in K_trucks:
        proc_sum = gp.quicksum(P[i] * x[j, i, k] for i in Groups[g] for j in All_Nodes if (j, i) in Edges)
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
        visits_gh = gp.quicksum(x[j, i, k] for i in Groups[g] for j in All_Nodes if (j, i) in Edges and j not in Groups[g])
        
        m.addConstr(gp.quicksum(y[k, d, g] for d in Docks) == visits_gh,
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