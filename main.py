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
Length_u = 1.626  # meters (Converted 162.6cm to m)
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
