#THIS FILE PERFORMS A SENSITIVITY ANALYSIS ON THE NUMBER OF GH FACILITIES AND FF FACILITIES

# Loading packages that are used in the code
import numpy as np
import os
import pandas as pd
from gurobipy import Model,GRB,LinExpr
import gurobipy as gp
from math import radians, cos, sin, asin, sqrt
import random
from typing import Dict, List
import matplotlib.pyplot as plt


# Get path to current folder
cwd = os.getcwd()

# Get all instances
full_list           = os.listdir(cwd)
model=Model()
M=10000 # Big M for time constraints (should be larger than Horizon + max processing time)



### DATA INPUT & PARAMETER DEFINITIONS ###

tighter_windows_instance=0 # proportion of nodes with tightened time windows (0.2 = 20% of nodes have tighter windows)
Delta_GH = 1 # number of docks per GH (Assuming 'Very Large' instance setting or standard)
Docks = list(range(1, Delta_GH + 1)) # Set of Docks
n_uld = 6 # number of ULDs (Pickups = 1..n_uld, Deliveries = n_uld+1..2*n_uld)
K_trucks = [1, 2, 3] #truck instances
Weight_u = 1000   # weight of each ULD in [kg]
Length_u = 1.534  # length of each UL in [m]
Proc_Time = 15     # processing time of each uld at FF/GH in [minutes]
Horizon = 480     # total time limit in [minutes]
Cap_W = 3000     # weight Capacity of each Truck in [kg]
Cap_L = 13.6      # length Capacity of each Truck in [m]
Speed_kmh = 35    # speed in km/h (assumed constant for all edges)
Speed_mpm = Speed_kmh / 60.0 # km per minute

# Coordinates (DMS to Decimal Degrees)
def dms_to_dd(d, m, s, direction='N'):
    dd = d + m/60 + s/3600
    if direction in ['S', 'W']: dd *= -1
    return dd

# Haversine Distance Function to get the distance in [km] between two lat/lon coordinates
def get_dist(coord1, coord2):
    lat1, lon1, lat2, lon2 = map(radians, [coord1[0], coord1[1], coord2[0], coord2[1]])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

# Plotting Routes and Facilities
def plot_routes(routes: Dict[int, List[int]], node_coords: List[List[float]], ff_nodes: Dict[int, List[int]], gh_nodes: Dict[int, List[int]], output_path: str = "truck_routes.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot depot
    depot = node_coords[0]
    ax.scatter(depot[1], depot[0], c="black", s=80, marker="s", label="Depot")

    def _node_xy(node_idx: int):
        return node_coords[node_idx][1], node_coords[node_idx][0]

    def _node_label_offset(node_idx: int, total_nodes: int, base_x: int = 10, base_y: int = 10):
        if total_nodes <= 1:
            return base_x, base_y
        centered_rank = node_idx - (total_nodes - 1) / 2
        return base_x, base_y + int(centered_rank * 12)

    def _plot_facility_nodes(node_groups: Dict[int, List[int]], prefix: str, color: str, marker: str):
        for group_id, nodes in node_groups.items():
            total_nodes = len(nodes)
            for node_idx, node in enumerate(nodes):
                xy = _node_xy(node)
                ax.scatter(xy[0], xy[1], c=color, s=150, marker=marker)
                ax.annotate(
                    f"N{node} ({prefix}{group_id})",
                    xy=xy,
                    xytext=_node_label_offset(node_idx, total_nodes),
                    textcoords="offset points",
                    fontsize=10,
                    color=color,
                    weight="bold",
                )

    _plot_facility_nodes(ff_nodes, "FF", "#1f77b4", "o")
    _plot_facility_nodes(gh_nodes, "GH", "#ff7f0e", "^")

    ax.scatter([], [], c="#1f77b4", s=60, marker="o", label="FF")
    ax.scatter([], [], c="#ff7f0e", s=60, marker="^", label="GH")

    used_truck_ids = sorted(k for k, route in routes.items() if len(route) >= 2)
    min_rad, max_rad = 0.07, 0.13
    precision = 10000
    rad_pool = list(range(int(min_rad * precision), int(max_rad * precision) + 1))
    picked_rads = random.sample(rad_pool, len(used_truck_ids))
    truck_curvature = {
        truck_id: rad_int / precision
        for truck_id, rad_int in zip(used_truck_ids, picked_rads)
    }

    colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for idx, (k, route) in enumerate(routes.items()): #Each Truck K follows a specific route 
        segments = list(zip(route, route[1:]))
        if not segments:
            continue
        color = colors[idx % len(colors)]
        route_distance_km = sum(
            get_dist(node_coords[i], node_coords[j]) for i, j in segments
        )
        route_xy = [_node_xy(node) for node in route]
        xs = [xy[0] for xy in route_xy]
        ys = [xy[1] for xy in route_xy]
        ax.plot([], [], color=color, linewidth=2, label=f"Truck {k} ({route_distance_km:.2f} km)", alpha=0.8)
        ax.scatter(xs, ys, color=color, s=30)
        for i, j in segments:
            x0, y0 = _node_xy(i)
            x1, y1 = _node_xy(j)
            rad = truck_curvature.get(k, 0.09)
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.8,
                    shrinkA=6,
                    shrinkB=6,
                    connectionstyle=f"arc3,rad={rad}",
                    alpha=0.9,
                ),
            )

    ax.set_title("Truck Routes (FFs/GHs)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Route plot saved to: {output_path}")

    plt.show()


def plot_truck_timeline_gantt(
    routes: Dict[int, List[int]],
    trucks: List[int],
    travel_time: Dict,
    tau_vals: Dict[int, float],
    proc_times: Dict[int, float],
    nodes_p: List[int],
    ff_nodes: Dict[int, List[int]],
    a_f_vals: Dict,
    d_f_vals: Dict,
    w_f_vals: Dict,
    gh_nodes: Dict[int, List[int]],
    a_g_vals: Dict,
    d_g_vals: Dict,
    w_d_vals: Dict,
    w_g_vals: Dict,
    tau_end_vals: Dict[int, float],
    output_path: str = "truck_timeline.png",
):
    fig_height = max(4, 1.2 * len(trucks) + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    phase_colors = {
        "travel": "#1f77b4",
        "ff_service": "#2ca02c",
        "ff_wait": "#17becf",
        "gh_queue": "#ff7f0e",
        "gh_wait": "#bcbd22",
        "gh_service": "#9467bd",
        "return": "#7f7f7f",
    }
    phase_labels = {
        "travel": "Travel",
        "ff_service": "Service at FF",
        "ff_wait": "Waiting at FF",
        "gh_queue": "Queue at GH",
        "gh_wait": "Waiting at GH dock",
        "gh_service": "Service at GH dock",
        "return": "Return",
    }

    used_labels = set()
    eps = 1e-6

    def add_phase_bar(y_pos, start, duration, phase):
        if duration <= eps:
            return
        label = phase_labels[phase] if phase not in used_labels else None
        if label is not None:
            used_labels.add(phase)
        ax.barh(
            y=y_pos,
            width=duration,
            left=start,
            height=0.62,
            color=phase_colors[phase],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            label=label,
        )

    for row, k in enumerate(trucks):
        route = routes.get(k, [])

        if len(route) >= 2:
            for pos in range(len(route) - 1):
                i = route[pos]
                j = route[pos + 1]
                travel_start = tau_vals.get(0, 0.0) if i == 0 else tau_vals.get(i, 0.0) + proc_times.get(i, 0.0)
                travel_duration = travel_time.get((i, j), 0.0)
                phase = "return" if j == 0 else "travel"
                add_phase_bar(row, travel_start, travel_duration, phase)

            for node in route:
                if node in nodes_p:
                    add_phase_bar(row, tau_vals.get(node, 0.0), proc_times.get(node, 0.0), "ff_service")

        for f in ff_nodes.keys():
            arr_f = a_f_vals.get((k, f), 0.0)
            dep_f = d_f_vals.get((k, f), 0.0)
            ff_wait = max(0.0, w_f_vals.get((k, f), 0.0))
            if dep_f > arr_f + eps and ff_wait > eps:
                ff_wait_start = max(arr_f, dep_f - ff_wait)
                add_phase_bar(row, ff_wait_start, min(ff_wait, dep_f - ff_wait_start), "ff_wait")

        for g in gh_nodes.keys():
            arr = a_g_vals.get((k, g), 0.0)
            dep = d_g_vals.get((k, g), 0.0)
            queue = max(0.0, w_d_vals.get((k, g), 0.0))
            dock_start = arr + queue
            dock_duration = max(0.0, dep - dock_start)
            gh_wait = min(max(0.0, w_g_vals.get((k, g), 0.0)), dock_duration)
            gh_service = max(0.0, dock_duration - gh_wait)

            add_phase_bar(row, arr, queue, "gh_queue")
            add_phase_bar(row, dock_start, gh_wait, "gh_wait")
            add_phase_bar(row, dock_start + gh_wait, gh_service, "gh_service")

    max_end = max([tau_end_vals.get(k, 0.0) for k in trucks] + [1.0])
    ax.set_xlim(0, max_end * 1.05)
    ax.set_ylim(-0.8, len(trucks) - 0.2)
    ax.set_yticks(range(len(trucks)))
    ax.set_yticklabels([f"Truck {k}" for k in trucks])
    ax.set_xlabel("Time [min]")
    ax.set_title("Truck Timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Truck timeline Gantt saved to: {output_path}")
    plt.show()

def plot_ff_gh_flow_matrix(
    served_pickups: List[int],
    n_pickups: int,
    ff_nodes: Dict[int, List[int]],
    gh_nodes: Dict[int, List[int]],
    output_path: str = "ff_gh_flow_matrix.png",
):
    ff_ids = sorted(ff_nodes.keys())
    gh_ids = sorted(gh_nodes.keys())

    pickup_to_ff = {}
    for f, nodes in ff_nodes.items():
        for node in nodes:
            pickup_to_ff[node] = f

    delivery_to_gh = {}
    for g, nodes in gh_nodes.items():
        for node in nodes:
            delivery_to_gh[node] = g

    ff_pos = {f: idx for idx, f in enumerate(ff_ids)}
    gh_pos = {g: idx for idx, g in enumerate(gh_ids)}
    flow_matrix = np.zeros((len(ff_ids), len(gh_ids)), dtype=int)

    for pickup_node in served_pickups:
        ff_id = pickup_to_ff.get(pickup_node)
        delivery_node = pickup_node + n_pickups
        gh_id = delivery_to_gh.get(delivery_node)
        if ff_id is None or gh_id is None:
            continue
        flow_matrix[ff_pos[ff_id], gh_pos[gh_id]] += 1

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(flow_matrix, cmap="Blues", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of ULDs")

    ax.set_xticks(np.arange(len(gh_ids)))
    ax.set_yticks(np.arange(len(ff_ids)))
    ax.set_xticklabels([f"GH{g}" for g in gh_ids])
    ax.set_yticklabels([f"FF{f}" for f in ff_ids])
    ax.set_xlabel("Ground Handler (GH)")
    ax.set_ylabel("Freight Forwarder (FF)")
    ax.set_title("FF to GH Flow Matrix (ULD Count)")

    for i in range(flow_matrix.shape[0]):
        for j in range(flow_matrix.shape[1]):
            val_ij = flow_matrix[i, j]
            txt_color = "white" if val_ij > flow_matrix.max() / 2 and flow_matrix.max() > 0 else "black"
            ax.text(j, i, f"{val_ij}", ha="center", va="center", color=txt_color, fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"FF-to-GH flow matrix saved to: {output_path}")

#NODE MAPPING
Nodes_P = list(range(1, n_uld+1)) # Pickup nodes (1 to n_uld)
Nodes_D = list(range(n_uld+1, 2*n_uld+1)) # Delivery nodes (n_uld+1 to 2*n_uld)
All_Nodes = [0] + Nodes_P + Nodes_D  # 0 is depot, followed by pickups and deliveries

# FFs/GHs are created dynamically from the selected `locs` set (see below)
FFs = {}
GHs = {}

Edges = []
for i in All_Nodes:
    for j in All_Nodes:
        if i == j: 
            continue
        # 1) Forbid to go from depot to delivery: 0 -> D
        if i == 0 and j in Nodes_D:
            continue
        # 2) Forbid returning to the depot from a pickup: P -> 0
        if i in Nodes_P and j == 0:
            continue
        # 3) Forbid going to a delivery before its corresponding pickup
        if j in Nodes_D:
            pickup_node = j - n_uld
            if pickup_node in Nodes_P:
                # Don't allow edge to delivery unless pickup was already visited
                # This requires checking if i is the pickup or comes after it
                if i != pickup_node and i not in Nodes_D:
                    continue
        Edges.append((i, j))


# Parameter Dictionaries
T = {} # Travel Time
P = {0: 0} # Processing Time
E_win = {0: 0} # Earliest Time
D_win = {0: Horizon} # Latest Time
W = {0: 0} # Weights
L = {0: 0} # Lengths



# ALL_LOCS = {
#     'FF1': (dms_to_dd(52,17,46.8), dms_to_dd(4,46,10.4)),
#     'FF2': (dms_to_dd(52,18,6.8),  dms_to_dd(4,45,3.2)),
#     'FF3': (dms_to_dd(52,17,0.9108), dms_to_dd(4,46,9.0422)),
#     'FF4': (dms_to_dd(52,16,29.579), dms_to_dd(4,44,28.912)),
#     'GH1': (dms_to_dd(52,17,0.8),  dms_to_dd(4,46,7.1)),
#     'GH2': (dms_to_dd(52,16,32.9), dms_to_dd(4,44,30.0)),
#     'GH3': (dms_to_dd(52,17,42.297), dms_to_dd(4,45,57.302)),
#     'GH4': (dms_to_dd(52,17,50.289), dms_to_dd(4,44,46.132))
    
# }

# Locations
locs_0 = {
    'FF1': (dms_to_dd(52,17,46.8), dms_to_dd(4,46,10.4)),
    'FF2': (dms_to_dd(52,18,6.8),  dms_to_dd(4,45,3.2)),
    'GH1': (dms_to_dd(52,17,0.8),  dms_to_dd(4,46,7.1)),
    'GH2': (dms_to_dd(52,16,32.9), dms_to_dd(4,44,30.0))
}



locs_1 = {
    'FF2': (dms_to_dd(52,18,6.8),  dms_to_dd(4,45,3.2)),
    'GH1': (dms_to_dd(52,17,0.8),  dms_to_dd(4,46,7.1)),
    'GH2': (dms_to_dd(52,16,32.9), dms_to_dd(4,44,30.0)),
}


locs_2 = {
    'FF1': (dms_to_dd(52,17,46.8), dms_to_dd(4,46,10.4)),
    'FF2': (dms_to_dd(52,18,6.8),  dms_to_dd(4,45,3.2)),
    'GH2': (dms_to_dd(52,16,32.9), dms_to_dd(4,44,30.0)) 
}



locs_3 = {
    'FF1': (dms_to_dd(52,17,46.8), dms_to_dd(4,46,10.4)),
    'FF2': (dms_to_dd(52,18,6.8),  dms_to_dd(4,45,3.2)),
    'FF3': (dms_to_dd(52,15,0.9108), dms_to_dd(4,46,9.0422)),
    'GH1': (dms_to_dd(52,17,0.8),  dms_to_dd(4,46,7.1)),
    'GH2': (dms_to_dd(52,16,32.9), dms_to_dd(4,44,30.0)),
    'GH3': (dms_to_dd(52,17,42.297), dms_to_dd(4,45,57.302))
}

locs_4 = {
    'FF1': (dms_to_dd(52,17,46.8), dms_to_dd(4,46,10.4)),
    'FF2': (dms_to_dd(52,18,6.8),  dms_to_dd(4,45,3.2)),
    'FF3': (dms_to_dd(52,15,0.9108), dms_to_dd(4,46,9.0422)),
    'FF4': (dms_to_dd(52,18,29.579), dms_to_dd(4,44,28.912)),
    'GH1': (dms_to_dd(52,17,0.8),  dms_to_dd(4,46,7.1)),
    'GH2': (dms_to_dd(52,16,32.9), dms_to_dd(4,44,30.0)),
    'GH3': (dms_to_dd(52,17,42.297), dms_to_dd(4,45,57.302)),
    'GH4': (dms_to_dd(52,17,50.289), dms_to_dd(4,44,46.132))  
}

LOCS_SETS = {
    "locs_1": locs_1,
    "locs_2": locs_2,
    "locs_3": locs_3,
    "locs_4": locs_4,
}

ENABLE_PLOTS = "1"
# SELECT YOU LOCATION SET HERE
selected_locs_name = "locs_4" 
locs = LOCS_SETS[selected_locs_name]


def _facility_sort_key(name: str): #gets the number identifier for each GH and FF
    digits = "".join(ch for ch in name if ch.isdigit())
    suffix_num = int(digits) if digits else 0
    return (name.rstrip("0123456789"), suffix_num, name)

ff_location_keys = sorted([key for key in locs.keys() if key.upper().startswith("FF")], key=_facility_sort_key)
gh_location_keys = sorted([key for key in locs.keys() if key.upper().startswith("GH")], key=_facility_sort_key)

if len(ff_location_keys) == 0 or len(gh_location_keys) == 0:
    raise ValueError("The selected `locs` must include at least one FF* and one GH* key.")

# Use at most one pickup/delivery node per facility to avoid empty groups
ff_location_keys = ff_location_keys[:min(len(ff_location_keys), len(Nodes_P))]
gh_location_keys = gh_location_keys[:min(len(gh_location_keys), len(Nodes_D))]

FFs = {
    idx + 1: list(group)
    for idx, group in enumerate(np.array_split(Nodes_P, len(ff_location_keys)))
}
GHs = {
    idx + 1: list(group)
    for idx, group in enumerate(np.array_split(Nodes_D, len(gh_location_keys)))
}

ff_key_by_id = {idx + 1: loc_key for idx, loc_key in enumerate(ff_location_keys)}
gh_key_by_id = {idx + 1: loc_key for idx, loc_key in enumerate(gh_location_keys)}

pickup_to_ff_id = {
    pickup_node: ff_id
    for ff_id, nodes in FFs.items()
    for pickup_node in nodes
}
delivery_to_gh_id = {
    delivery_node: gh_id
    for gh_id, nodes in GHs.items()
    for delivery_node in nodes
}

node_loc_maps = [[52.2905, 4.7627]] # Depot

for i in Nodes_P:
    ff_id = pickup_to_ff_id[i]
    node_loc_maps.append(list(locs[ff_key_by_id[ff_id]]))

for i in Nodes_D:
    gh_id = delivery_to_gh_id[i]
    node_loc_maps.append(list(locs[gh_key_by_id[gh_id]]))

# Fill Parameters
for i in  All_Nodes:
    # Processing Time
    if i != 0: P[i] = Proc_Time
    
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
    if i in Nodes_P:
        E_win[i] = 0; D_win[i] = Horizon
    elif i in Nodes_D:
        E_win[i] = 0; D_win[i] = Horizon

# Tightened Time Windows
tightened_P_windows = []
shuffled_Nodes=All_Nodes[1:].copy()
random.shuffle(shuffled_Nodes)
print(shuffled_Nodes[:int(tighter_windows_instance*len(All_Nodes))])
for i in shuffled_Nodes[:int(tighter_windows_instance*len(All_Nodes))]:
    if i in Nodes_P:
        E_win[i] = 50; D_win[i] = 150
        tightened_P_windows.append(i)
    elif i in Nodes_D:
        E_win[i] = 200; D_win[i] = 350
        if i-n_uld in tightened_P_windows:
            D_win[i]+= 30 # Includes 30 min buffer if needed per paper????? 

# Calculate Travel Matrix (T_ij)
for i, j in Edges:
    dist_km = get_dist(node_loc_maps[i], node_loc_maps[j])
    T[i,j] = dist_km / Speed_mpm

#Define Model
m = gp.Model("GHDC-PDPTW")



### VARIABLES ###

x = m.addVars(Edges, K_trucks, vtype=GRB.BINARY, name="x")
tau = m.addVars( All_Nodes, vtype=GRB.CONTINUOUS, name="tau")
tau_end = m.addVars(K_trucks, vtype=GRB.CONTINUOUS, name="tau_end")

# Facility/Group Arrival/Departure times (indexed by Truck, Facility/Group ID)
a_F = m.addVars(K_trucks, FFs.keys(), vtype=GRB.CONTINUOUS, name="a_F")
d_F = m.addVars(K_trucks, FFs.keys(), vtype=GRB.CONTINUOUS, name="d_F")
a_G = m.addVars(K_trucks, GHs.keys(), vtype=GRB.CONTINUOUS, name="a_G")
# d_G is used in later constraints (23+), but required if we were doing the full model
d_G = m.addVars(K_trucks, GHs.keys(), vtype=GRB.CONTINUOUS, name="d_G") 

# Waiting times
w_D = m.addVars(K_trucks, GHs.keys(), vtype=GRB.CONTINUOUS, name="w_D") # Waiting at GH before docking
w_F = m.addVars(K_trucks, FFs.keys(), vtype=GRB.CONTINUOUS, name="w_F") # Waiting at FF while docked
w_G = m.addVars(K_trucks, GHs.keys(), vtype=GRB.CONTINUOUS, name="w_G") # Waiting at GH while docked

# Scheduling and Dock Assignment variables:
#   eta: 1 if k1 leaves g before k2 docks
eta = m.addVars(K_trucks, K_trucks, GHs.keys(), vtype=GRB.BINARY, name="eta") 
y = m.addVars(K_trucks, Docks, GHs.keys(), vtype=GRB.BINARY, name="y")
#   z: Linearization variable for y*y
z = m.addVars(K_trucks, Docks, K_trucks, Docks, GHs.keys(), vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")

### OBJECTIVE FUNCTION ###

# Term 1: Total Travel Time (Sum over all trucks k and all edges (i,j))
travel_cost = gp.quicksum(T[i, j] * x[i, j, k] 
                          for k in K_trucks 
                          for i, j in Edges)
# Term 2: Waiting time at GH queues (Before Docking)
wait_gh_dock_cost = gp.quicksum(w_D[k, g] 
                                for k in K_trucks 
                                for g in GHs.keys())
# Term 3: Waiting time at GH docks (During Service/Windows)
wait_gh_service_cost = gp.quicksum(w_G[k, g] 
                                   for k in K_trucks 
                                   for g in GHs.keys())
# Term 4: Waiting time at FF docks (During Service/Windows)
wait_ff_cost = gp.quicksum(w_F[k, f] 
                           for k in K_trucks 
                           for f in FFs.keys())

# [1] Objective Function
m.setObjective(travel_cost + wait_gh_dock_cost + wait_gh_service_cost + wait_ff_cost, 
               GRB.MINIMIZE)

m.update()


### CONSTRAINTS ###
# [2] Each pickup node is visited exactly once
for i in Nodes_P:
    m.addConstr(
        gp.quicksum(x[j, i, k]
                 for k in K_trucks
                 for j in All_Nodes
                 if (j, i) in Edges) == 1,
        name=f"pickup_node_visted_once_{i}"
    )

# [3] Pickup and delivery of each ULD are done by the same truck
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


# [4] Each truck leaves the depot at most once
depot = 0

for k in K_trucks:
    m.addConstr(
        gp.quicksum(x[depot, i, k] for i in All_Nodes if (depot, i) in Edges) <= 1,
        name=f"use_truck_at_most_once_k{k}"
    )

# [5] What enters a node equals what leaves it (except depot, where this just allows start/end).
for k in K_trucks:
    for i in All_Nodes:
        m.addConstr(
            gp.quicksum(x[j, i, k] for j in All_Nodes if (j, i) in Edges) -
            gp.quicksum(x[i, j, k] for j in All_Nodes if (i, j) in Edges)
            == 0,
            name=f"flow_conservation_at_each_node_k{k}_i{i}"
        )


# [6] LIFO strategy for first and last nodes visisted by each truck
for k in K_trucks:
    for i in Nodes_P:
        m.addConstr(
            x[depot, i, k] - x[i + n, depot, k] == 0,
            name=f"LIFO_first_last_i{i}_k{k}"
        )

# [7] LIFO strategy for intermediate nodes
for k in K_trucks:
    for (i, j) in Edges:
        if i in Nodes_P and j in Nodes_P:  # both are pickup nodes
            m.addConstr(
                x[i, j, k] - x[j + n, i + n, k] == 0,
                name=f"LIFO_reverse_i{i}_j{j}_k{k}"
            )


# [8] Each FF can be visited at most once per truck
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
            <= 1,
            name=f"visit_FF_at_most_once_f{f}_k{k}"
        )

# [9] Each GH can be visited at most once per truck
for k in K_trucks:
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
            <= 1,
            name=f"visit_GH_at_most_once_g{g}_k{k}"
        )


# [10] Time precedence for pickup nodes
for (i, j) in Edges:
    if j in Nodes_P:  # j is a pickup node (the one following i)
        m.addConstr(
            tau[j] >= tau[i] + P[i] + T[i, j]
                      - M * (1 - gp.quicksum(x[i, j, k] for k in K_trucks)),
            name=f"time_pickups_i{i}_j{j}"
        )

# [11] Time precedence for entering a GH
for g in GHs:
    nodes_g = GHs[g]  # delivery nodes of GH g
    for k in K_trucks:
        for (i, j) in Edges:
            if j in nodes_g and i not in nodes_g and j in Nodes_D: # j is a delivery node in GH g, and i is outside GH g
                m.addConstr(
                    tau[j] >= tau[i] + P[i] + T[i, j] - M * (1 - x[i, j, k]) + w_D[k, g],
                    name=f"time_enter_GH_g{g}_k{k}_i{i}_j{j}"
                )

# [12] Time consistency within GH GHs (Ensures valid flow of time when moving between nodes inside the same Ground Handler)
for g, nodes in GHs.items(): #i and j belonging to the same GH
    for i in nodes:
        for j in nodes:
            if i != j and (i, j) in Edges:
                expr =  gp.quicksum(x[i, j, k] for k in K_trucks)
                m.addConstr(tau[j] >= tau[i] + P[i] + T[i,j] - (1 - expr)*M, 
                            name=f"Eq12_IntraGroupTime_{i}_{j}")

# [13] Vehicle End Time (Only for nodes i with an allowed arc i -> 0 (avoids KeyError when P->0 is forbidden))
for k in K_trucks:
    for i in All_Nodes:
        if i != 0 and (i, 0) in Edges:
            m.addConstr(
                tau_end[k] >= tau[i] + P[i] + T[i, 0] - (1 - x[i, 0, k]) * M,
                name=f"Eq13_EndTime_{k}_{i}"
            )


# [14] Time Windows (Earliest and Latest service times)
for i in Nodes_P + Nodes_D:
    m.addConstr(tau[i] >= E_win[i], name=f"Eq14_TW_LB_{i}")
    m.addConstr(tau[i] <= D_win[i], name=f"Eq14_TW_UB_{i}")

# [15] Weight Capacity (Total weight of Pickups visited by truck k must not exceed Q_W)
for k in K_trucks:
    load_expr =  gp.quicksum(W[i] * x[j, i, k] 
                            for i in  All_Nodes 
                            for j in  All_Nodes if (j, i) in Edges)
    m.addConstr(load_expr <= Cap_W, name=f"Eq15_WeightCap_{k}")

# [16] Length Capacity (Total length of Pickups visited by truck k must not exceed Q_L)
for k in K_trucks:
    len_expr =  gp.quicksum(L[i] * x[j, i, k] 
                           for i in  All_Nodes 
                           for j in  All_Nodes if (j, i) in Edges)
    m.addConstr(len_expr <= Cap_L, name=f"Eq16_LengthCap_{k}")

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



### SOLVE & QUICK REPORT ###

print(f"Model has {m.numConstrs} constraints and {m.numVars} variables")
print("\nFF and GH node assignments:")
for f in FFs.keys():
    print(f"  FF{f}: {FFs[f]}")
for g in GHs.keys():
    print(f"  GH{g}: {GHs[g]}")
m.optimize()

def val(v):
    try:
        return float(v.X)
    except:
        return float(v)

if m.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
    print(f"[STATUS] {m.status}")
else:
    print("\n =================== SOLUTION REPORT ===================")
    #objective decomposition
    travel_val = travel_cost.getValue()
    wait_gh_dock_val = wait_gh_dock_cost.getValue()
    wait_gh_service_val = wait_gh_service_cost.getValue()
    wait_ff_val = wait_ff_cost.getValue()
    total_obj_val = travel_val + wait_gh_dock_val + wait_gh_service_val + wait_ff_val
    safe_total = total_obj_val if abs(total_obj_val) > 1e-9 else 1.0
    print(f"Objective Value: {m.objVal:.4f}")
    print(" Objective breakdown:")
    print(f"  1) Travel time:            {travel_val:10.4f} ({100.0*travel_val/safe_total:5.1f}%)")
    print(f"  2) GH pre-dock waiting:    {wait_gh_dock_val:10.4f} ({100.0*wait_gh_dock_val/safe_total:5.1f}%)")
    print(f"  3) GH dock waiting:        {wait_gh_service_val:10.4f} ({100.0*wait_gh_service_val/safe_total:5.1f}%)")
    print(f"  4) FF dock waiting:        {wait_ff_val:10.4f} ({100.0*wait_ff_val/safe_total:5.1f}%)")

    # (0) Calculate distances between all FFs and GHs
    print("\n Distances between FFs and GHs in [km]:")
    for f in FFs.keys():
        for g in GHs.keys():
            dist = get_dist(locs[ff_key_by_id[f]], locs[gh_key_by_id[g]])
            print(f"  FF{f} -> GH{g}: {dist:.2f} [km]")

    # 1) Used paths x=1 (per truck)
    print("\n Paths used x[i,j,k]=1:")
    for k in K_trucks:
        used = [(i,j) for (i,j) in Edges if val(x[i,j,k]) > 0.5]
        print(f"  Truck {k}: {used}")

    # 2) Simple route per truck (from depot following successors)
    print("\n Route Followed (from depot 0):")
    routes = {}
    for k in K_trucks:
        succ = {i:j for (i,j) in Edges if val(x[i,j,k]) > 0.5}
        route = [0]
        cur = 0
        visited = set([0])
        # avoids infinite loops in pathological cases
        for _ in range(len(All_Nodes)+2):
            if cur in succ:
                nxt = succ[cur]
                route.append(nxt)
                if nxt in visited: break
                visited.add(nxt)
                cur = nxt
            else:
                break
            routes[k] = route
        print(f"  Truck {k}: {' -> '.join(map(str, route))}")
    
    print("\n Truck utilization and route:")
    used_trucks = []
    travel_time_by_truck = {}
    distance_by_truck = {}
    for k in K_trucks:
        used_arcs = [(i, j) for (i, j) in Edges if val(x[i, j, k]) > 0.5]
        used_flag = 1 if used_arcs else 0
        if used_flag:
            used_trucks.append(k)
        route_travel = sum(T[i, j] for (i, j) in used_arcs)
        travel_time_by_truck[k] = route_travel
        distance_by_truck[k] = route_travel * Speed_mpm
        first_departure = val(tau[0]) if any(i == 0 for (i, _) in used_arcs) else float('nan')
        end_time = val(tau_end[k])
        active_time = max(0.0, end_time - (val(tau[0]) if used_flag else 0.0))
        print(f"  Truck {k}: used={used_flag}, arcs={len(used_arcs)}, travel={route_travel:.2f}, start={first_departure:.2f}, end={end_time:.2f}, active={active_time:.2f}")

    # 3) Node times (tau) and windows
    print("\n Tau per node with specified time windows:")
    for i in All_Nodes:
        t = val(tau[i])
        e = E_win.get(i,None); d = D_win.get(i,None)
        print(f"  node {i:>2}: tau={t:7.2f}  [E={e}, D={d}]")

    # 4) Times/waits at FF and GH (if applicable)
    if FFs:
        print("\n FF times (a_F, d_F, w_F):")
        for k in K_trucks:
            for f in FFs.keys():
                visited_ff = any(
                    val(x[j, i, k]) > 0.5
                    for i in FFs[f] for j in All_Nodes if (j, i) in Edges and j not in FFs[f]
                )
                a_f_str = f"{val(a_F[k, f]):.2f}" if visited_ff else "NaN"
                d_f_str = f"{val(d_F[k, f]):.2f}" if visited_ff else "NaN"
                print(f"  k={k}, FF={f}: a_F={a_f_str}, d_F={d_f_str}, w_F={val(w_F[k,f]):.2f}")
    if GHs:
        print("\n GH times (a_G, d_G, w_D, w_G):")
        for k in K_trucks:
            for g in GHs.keys():
                print(f"  k={k}, GH={g}: a_G={val(a_G[k,g]):.2f}, d_G={val(d_G[k,g]):.2f}, w_D={val(w_D[k,g]):.2f}, w_G={val(w_G[k,g]):.2f}")

    # 5) Dock allocations
    print("\n Dock allocations y[k,d,g]=1:")
    any_y = False
    for k in K_trucks:
        for g in GHs.keys():
            for d in Docks:
                if val(y[k,d,g]) > 0.5:
                    print(f"  k={k} -> GH {g}, dock {d}")
                    any_y = True
    if not any_y: print("  (None)")
    if GHs:
        print("\n Dock congestion by GH:")
        for g in GHs.keys():
            w_pre_vals = [val(w_D[k, g]) for k in K_trucks]
            w_dock_vals = [val(w_G[k, g]) for k in K_trucks]
            dock_assignments = sum(1 for k in K_trucks for d in Docks if val(y[k, d, g]) > 0.5)
            print(f"  GH {g}: assigned={dock_assignments}, pre-dock avg/max={sum(w_pre_vals)/len(w_pre_vals):.2f}/{max(w_pre_vals):.2f}, dock avg/max={sum(w_dock_vals)/len(w_dock_vals):.2f}/{max(w_dock_vals):.2f}")

    # 6) Precedence relations at GHs
    print("\n Precedences eta[k1,k2,g]=1:")
    any_eta = False
    for g in GHs.keys():
        for k1 in K_trucks:
            for k2 in K_trucks:
                if k1 != k2 and val(eta[k1,k2,g]) > 0.5:
                    print(f"  GH {g}: k1={k1} antes de k2={k2}")
                    any_eta = True
    if not any_eta: print("  (None)")
    print()

    # 7) Capacity usage (sum of pickups served per truck)
    print("Capacity usage per truck (weight and length):")
    csv_rows = []
    for k in K_trucks:
        weight = 0.0; length = 0.0
        for i in Nodes_P:
            if any(val(x[j,i,k]) > 0.5 for j in All_Nodes if (j,i) in Edges):
                weight += W[i]; length += L[i]
        print(f"  k={k}: Weight={weight}/{Cap_W}  Length={length}/{Cap_L}")
        csv_rows.append({
            "location_set": selected_locs_name,
            "objective_value": float(m.objVal),
            "obj_travel_time": float(travel_val),
            "obj_wait_gh_pre_dock": float(wait_gh_dock_val),
            "obj_wait_gh_dock": float(wait_gh_service_val),
            "obj_wait_ff_dock": float(wait_ff_val),
            "truck": k,
            "weight_occupied": weight,
            "length_occupied": length,
        })

    csv_output_path = os.path.join(cwd, "sensitivity_locs_objective_weight_length.csv")
    write_header = not os.path.exists(csv_output_path)
    csv_columns = [
        "location_set",
        "objective_value",
        "obj_travel_time",
        "obj_wait_gh_pre_dock",
        "obj_wait_gh_dock",
        "obj_wait_ff_dock",
        "truck",
        "weight_occupied",
        "length_occupied",
    ]
    pd.DataFrame(csv_rows, columns=csv_columns).to_csv(csv_output_path, mode="a", header=write_header, index=False)
    print(f"CSV results appended to: {csv_output_path}")

    print("==================================\n")
    print("\n Solver diagnostics:")
    visited_nodes = [i for i in Nodes_P + Nodes_D if any(val(x[j, i, k]) > 0.5 for k in K_trucks for j in All_Nodes if (j, i) in Edges)]
    if visited_nodes:
        early_slacks = [val(tau[i]) - E_win[i] for i in visited_nodes]
        late_slacks = [D_win[i] - val(tau[i]) for i in visited_nodes]
        on_time_count = sum(1 for i in visited_nodes if val(tau[i]) <= D_win[i] + 1e-6)
        print(f"  On-time nodes: {on_time_count}/{len(visited_nodes)}")
        #print(f"  Early slack min/avg/max: {min(early_slacks):.2f} / {sum(early_slacks)/len(early_slacks):.2f} / {max(early_slacks):.2f}") #how much you are above the earliest allowed time at a node.
        #print(f"  Late slack min/avg/max: {min(late_slacks):.2f} / {sum(late_slacks)/len(late_slacks):.2f} / {max(late_slacks):.2f}") #ow much room is left before you hit the latest allowed time.
    print(f"  Status={m.status}, Runtime={m.Runtime:.3f}s, Nodes={int(m.NodeCount)}")
    print(f"  BestObj={m.objVal:.4f}, BestBound={m.ObjBound:.4f}, Gap={100.0*m.MIPGap:.2f}%")
    print(f"  Trucks used: {len(used_trucks)}/{len(K_trucks)}")

    if ENABLE_PLOTS:
        plot_routes(routes, node_loc_maps, FFs, GHs)
        plot_truck_timeline_gantt(
            routes=routes,
            trucks=K_trucks,
            travel_time=T,
            tau_vals={i: val(tau[i]) for i in All_Nodes},
            proc_times=P,
            nodes_p=Nodes_P,
            ff_nodes=FFs,
            a_f_vals={(k, f): val(a_F[k, f]) for k in K_trucks for f in FFs.keys()},
            d_f_vals={(k, f): val(d_F[k, f]) for k in K_trucks for f in FFs.keys()},
            w_f_vals={(k, f): val(w_F[k, f]) for k in K_trucks for f in FFs.keys()},
            gh_nodes=GHs,
            a_g_vals={(k, g): val(a_G[k, g]) for k in K_trucks for g in GHs.keys()},
            d_g_vals={(k, g): val(d_G[k, g]) for k in K_trucks for g in GHs.keys()},
            w_d_vals={(k, g): val(w_D[k, g]) for k in K_trucks for g in GHs.keys()},
            w_g_vals={(k, g): val(w_G[k, g]) for k in K_trucks for g in GHs.keys()},
            tau_end_vals={k: val(tau_end[k]) for k in K_trucks},
        )
        
        served_pickups = [
            i for i in Nodes_P
            if any(val(x[j, i, k]) > 0.5 for k in K_trucks for j in All_Nodes if (j, i) in Edges)
        ]

        plot_ff_gh_flow_matrix(
            served_pickups=served_pickups,
            n_pickups=n_uld,
            ff_nodes=FFs,
            gh_nodes=GHs,
        )
    

