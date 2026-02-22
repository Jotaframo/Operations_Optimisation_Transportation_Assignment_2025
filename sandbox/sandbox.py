import gurobipy as gp
from gurobipy import GRB
import math

# --- 1. Sample Data Definition ---
# This section creates a small example problem.
# n = 3 ULDs
# F = 2 Freight Forwarders ('F1', 'F2')
# G = 1 Ground Handler ('G1')
# K = 2 Trucks ('T1', 'T2')

n = 3  # Number of ULDs [cite: 398]
n_nodes = 2 * n + 1 # Total nodes (0=depot, 1..n=pickup, n+1..2n=delivery) [cite: 412]

# Sets
N_P = list(range(1, n + 1)) # Pickup nodes [cite: 397]
N_D = list(range(n + 1, 2 * n + 1)) # Delivery nodes [cite: 397]
N_1 = [0] + N_P + N_D # All nodes [cite: 412]
F = ['F1', 'F2'] # Freight forwarder set [cite: 335, 430]
G = ['G1'] # Ground handler set [cite: 333, 430]
K = ['T1', 'T2'] # Truck set [cite: 332, 430]

# --- ULD and Node Parameters ---
# (Fictitious data for the example)

# Node-to-warehouse mapping
# ULD 1: F1 -> G1 (Nodes 1 -> 4)
# ULD 2: F1 -> G1 (Nodes 2 -> 5)
# ULD 3: F2 -> G1 (Nodes 3 -> 6)
FF_nodes = {'F1': [1, 2], 'F2': [3]} # ULD nodes per FF
GH_nodes = {'G1': [4, 5, 6]} # ULD nodes per GH

# Reverse node-to-entity mapping
node_to_FF = {1: 'F1', 2: 'F1', 3: 'F2'}
node_to_GH = {4: 'G1', 5: 'G1', 6: 'G1'}

# Coordinates to compute travel times
locations = {
    0: (5, 5),  # Depot
    1: (0, 5),  # F1 (ULD 1)
    2: (0, 5),  # F1 (ULD 2)
    3: (5, 0),  # F2 (ULD 3)
    4: (10, 5), # G1 (ULD 1)
    5: (10, 5), # G1 (ULD 2)
    6: (10, 5)  # G1 (ULD 3)
}

# Travel times T_ij [cite: 338]
def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

T = {}
for i in N_1:
    for j in N_1:
        T[i, j] = euclidean_dist(locations[i], locations[j])

# ULD parameters (weight, length, processing time)
# W, L, P are defined by node i
W = {1: 10, 2: 12, 3: 15} # Weight (only for pickup nodes) [cite: 329, 430]
L = {1: 1, 2: 1, 3: 1.5} # Length (only for pickup nodes) [cite: 329, 430]
P = {i: 5 for i in N_P} # Processing time at pickup [cite: 331, 430]
P.update({i: 6 for i in N_D}) # Processing time at delivery
P[0] = 0 # No processing time at depot

# Time windows [E_i, D_i] [cite: 417, 430]
E = {0: 0, 1: 10, 2: 10, 3: 15, 4: 50, 5: 50, 6: 60}
D = {0: 999, 1: 100, 2: 100, 3: 110, 4: 200, 5: 200, 6: 210}

# Fleet and Dock Parameters
Q_W = 30 # Truck weight capacity [cite: 332, 430]
Q_L = 3 # Truck length capacity [cite: 332, 430]
DELTA_GH = {'G1': 1} # Number of docks in each GH [cite: 339, 430]
M = 10000 # Constante Big-M

# --- 2. Construction of Arc Set (E_1) ---
# E_1 contains feasible arcs based on LIFO and precedence [cite: 413-415]
E_1 = gp.tuplelist()
# 1. Depot a Pickups
E_1.extend([(0, i) for i in N_P])
# 2. Pickup a Pickup
E_1.extend([(i, j) for i in N_P for j in N_P if i != j])
# 3. Pickup a Delivery (Enlace LIFO)
E_1.extend([(i, i + n) for i in N_P])
# 4. Delivery a Delivery
E_1.extend([(i, j) for i in N_D for j in N_D if i != j])
# 5. Delivery a Depot
E_1.extend([(i, 0) for i in N_D])

# --- 3. Creation of the Gurobi Model ---
m = gp.Model("GHDC_PDPTW_M1")

# --- 4. Variable Definition ---
# [cite: 419-427, 430]
# x_ij^k: binary, 1 if truck k goes from i to j
x = m.addVars(E_1, K, vtype=GRB.BINARY, name="x")

# tau_i: continuous, service start time at node i
tau = m.addVars(N_1, vtype=GRB.CONTINUOUS, name="tau")

# tau_0^k / tau_end^k: service start/end for truck k
# Note: We use tau[0] for the start. tau_end is defined for constraint (13)
tau_end = m.addVars(K, vtype=GRB.CONTINUOUS, name="tau_end") # [cite: 422, 430]

# a_kf^F / d_kf^F: arrival/departure times of truck k at FF f
a_FF = m.addVars(K, F, vtype=GRB.CONTINUOUS, name="a_FF")
d_FF = m.addVars(K, F, vtype=GRB.CONTINUOUS, name="d_FF")

# a_kg^G / d_kg^G: arrival/departure times of truck k at GH g
a_GH = m.addVars(K, G, vtype=GRB.CONTINUOUS, name="a_GH")
d_GH = m.addVars(K, G, vtype=GRB.CONTINUOUS, name="d_GH")

# w_kg^D: waiting time of truck k at GH g (before docking)
w_dock = m.addVars(K, G, vtype=GRB.CONTINUOUS, name="w_dock")

# w_kf^F / w_kg^G: waiting time of truck k at FF f / GH g (while docked)
w_FF_docked = m.addVars(K, F, vtype=GRB.CONTINUOUS, name="w_FF_docked")
w_GH_docked = m.addVars(K, G, vtype=GRB.CONTINUOUS, name="w_GH_docked")

# eta_{k1,k2}^g: binary, 1 if k1 finishes <= k2 docks at GH g
eta = m.addVars(K, K, G, vtype=GRB.BINARY, name="eta")

# y_kd^g: binary, 1 if truck k is assigned to dock d in GH g
# We create a dock set for each GH
Docks = {}
for g in G:
    Docks[g] = list(range(1, DELTA_GH[g] + 1))
y = m.addVars(K, G, [d for g in G for d in Docks[g]], vtype=GRB.BINARY, name="y")

# z_{k1,d1,k2,d2}^g: variable de linealización (continua, como en el paper) [cite: 591, 595]
z = m.addVars(K, K, G, [d1 for g in G for d1 in Docks[g]], [d2 for g in G for d2 in Docks[g]],
              vtype=GRB.CONTINUOUS, name="z")

# --- 5. Objective Function ---
# (1) Minimizar tiempo total de transporte y espera [cite: 435]
travel_time = gp.quicksum(T[i, j] * x[i, j, k] for i, j in E_1 for k in K)
wait_time_dock = gp.quicksum(w_dock[k, g] for k in K for g in G)
wait_time_at_FF = gp.quicksum(w_FF_docked[k, f] for k in K for f in F)
# The paper includes w_kg^G in objective equation (1) [cite: 435], but not in the text [cite: 536]
# We include it to stay faithful to equation (1):
wait_time_at_GH = gp.quicksum(w_GH_docked[k, g] for k in K for g in G)

# The paper sums FF waiting time twice [cite: 435]
# (likely a typo, sum(w_kg^D) + sum(w_kf^F)).
# We follow the text description [cite: 536-537] and eq (26) [cite: 506]
# which implies that w_kg^G is also a waiting time.
# Objective: travel time + pre-dock wait (GH) + docked wait (FF) + docked wait (GH)
m.setObjective(travel_time + wait_time_dock + wait_time_at_FF + wait_time_at_GH, GRB.MINIMIZE)


# --- 6. Constraints ---

# (2) Cada ULD es recogido exactamente una vez [cite: 437]
m.addConstrs((gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i for k in K) == 1
              for i in N_P), name="C2_PickupOnce")

# (3) Same truck for pair (i, i+n) [cite: 440]
for k in K:
    for i in N_P:
        m.addConstr(gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) - 
                    gp.quicksum(x[j, i + n, k] for j, i_n_k in E_1.select('*', i + n) if i_n_k == i + n) == 0,
                    name=f"C3_SameTruck_{k}_{i}")

# (4) Each truck is used at most once (leaves the depot) [cite: 442]
m.addConstrs((gp.quicksum(x[0, j, k] for j_k, j in E_1.select(0, '*') if j_k == 0) <= 1
              for k in K), name="C4_UseTruckOnce")

# (5) Conservación de flujo [cite: 444]
for k in K:
    for i in N_P + N_D:
        m.addConstr(gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) -
                    gp.quicksum(x[i, j, k] for i_k, j in E_1.select(i, '*') if i_k == i) == 0,
                    name=f"C5_FlowCons_{k}_{i}")

# (6) LIFO: First pickup (from depot) -> Last delivery (to depot) [cite: 447]
m.addConstrs((x[0, i, k] - x[i + n, 0, k] == 0
              for i in N_P for k in K), name="C6_LIFO_StartEnd")

# (7) LIFO: Secuencia de pickup (i,j) -> Secuencia de delivery (j+n, i+n) [cite: 450]
m.addConstrs((x[i, j, k] - x[j + n, i + n, k] == 0
              for i in N_P for j in N_P if i != j for k in K), name="C7_LIFO_Seq")

# (8) Each truck visits each FF at most once [cite: 453]
for k in K:
    for f in F:
        m.addConstr(gp.quicksum(x[j, i, k] for i in FF_nodes[f] for j, i_k in E_1.select('*', i) if i_k == i and node_to_FF.get(j, -1) != f) <= 1,
                    name=f"C8_VisitFFOnce_{k}_{f}")

# (9) Each truck visits each GH at most once [cite: 455]
# (We will also use this expression for C29)
visit_GH_expr = {}
for k in K:
    for g in G:
        visit_GH_expr[k, g] = gp.quicksum(x[j, i, k] for i in GH_nodes[g] for j, i_k in E_1.select('*', i) if i_k == i and node_to_GH.get(j, -1) != g)
        m.addConstr(visit_GH_expr[k, g] <= 1, name=f"C9_VisitGHOnce_{k}_{g}")


# (10) Precedencia de tiempo (Pickups) [cite: 457]
for i, j in E_1:
    if j in N_P:
        m.addConstr(tau[j] >= tau[i] + P[i] + T[i, j] - M * (1 - gp.quicksum(x[i, j, k] for k in K)),
                    name=f"C10_TimePrec_P_{i}_{j}")

# (11) Precedencia de tiempo (Hacia GH) [cite: 460]
for k in K:
    for g in G:
        for i, j in E_1:
            if j in GH_nodes[g] and node_to_GH.get(i, -1) != g:
                m.addConstr(tau[j] >= tau[i] + P[i] + T[i, j] - M * (1 - x[i, j, k]) + w_dock[k, g],
                            name=f"C11_TimePrec_toGH_{k}_{i}_{j}")

# (12) Precedencia de tiempo (Dentro de GH) [cite: 463]
for g in G:
    for i in GH_nodes[g]:
        for j in GH_nodes[g]:
            if (i, j) in E_1:
                m.addConstr(tau[j] >= tau[i] + P[i] + T[i, j] - M * (1 - gp.quicksum(x[i, j, k] for k in K)),
                            name=f"C12_TimePrec_inGH_{i}_{j}")

# (13) Precedencia de tiempo (Hacia Depot final) [cite: 465]
for k in K:
    for i, j in E_1.select('*', 0):
        if i in N_D: # Only from delivery nodes
            m.addConstr(tau_end[k] >= tau[i] + P[i] + T[i, 0] - M * (1 - x[i, 0, k]),
                        name=f"C13_TimePrec_toDepot_{k}_{i}")

# (14) Time windows [cite: 467]
m.addConstrs((tau[i] >= E[i] for i in N_1), name="C14_TimeWin_Early")
m.addConstrs((tau[i] <= D[i] for i in N_1), name="C14_TimeWin_Late")
m.addConstrs((tau_end[k] <= D[0] for k in K), name="C14_TimeWin_End") # We assume D[0] is the end of the horizon

# (15) Capacidad de Peso [cite: 470]
m.addConstrs((gp.quicksum(W[i] * gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) for i in N_P) <= Q_W
              for k in K), name="C15_Cap_Weight")

# (16) Capacidad de Longitud [cite: 473]
m.addConstrs((gp.quicksum(L[i] * gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) for i in N_P) <= Q_L
              for k in K), name="C16_Cap_Length")

# (17)-(20) Arrival and departure times at FF [cite: 476-487]
for k in K:
    for f in F:
        # Find arrival time (a_FF)
        for i, j in E_1:
            if j in FF_nodes[f] and node_to_FF.get(i, -1) != f:
                m.addConstr(a_FF[k, f] >= tau[i] + P[i] + T[i, j] - M * (1 - x[i, j, k]), name=f"C17_{k}_{f}_{i}_{j}")
                m.addConstr(a_FF[k, f] <= tau[i] + P[i] + T[i, j] + M * (1 - x[i, j, k]), name=f"C18_{k}_{f}_{i}_{j}")
        # Find departure time (d_FF)
        for i, j in E_1:
             if i in FF_nodes[f] and node_to_FF.get(j, -1) != f:
                m.addConstr(d_FF[k, f] >= tau[i] + P[i] - M * (1 - x[i, j, k]), name=f"C19_{k}_{f}_{i}_{j}")
                m.addConstr(d_FF[k, f] <= tau[i] + P[i] + M * (1 - x[i, j, k]), name=f"C20_{k}_{f}_{i}_{j}")

# (21)-(24) Arrival and departure times at GH [cite: 488-498]
for k in K:
    for g in G:
        # Find arrival time (a_GH)
        for i, j in E_1:
            if j in GH_nodes[g] and node_to_GH.get(i, -1) != g:
                m.addConstr(a_GH[k, g] >= tau[i] + P[i] + T[i, j] - M * (1 - x[i, j, k]), name=f"C21_{k}_{g}_{i}_{j}")
                m.addConstr(a_GH[k, g] <= tau[i] + P[i] + T[i, j] + M * (1 - x[i, j, k]), name=f"C22_{k}_{g}_{i}_{j}")
        # Find departure time (d_GH)
        for i, j in E_1:
             if i in GH_nodes[g] and node_to_GH.get(j, -1) != g:
                m.addConstr(d_GH[k, g] >= tau[i] + P[i] - M * (1 - x[i, j, k]), name=f"C23_{k}_{g}_{i}_{j}")
                m.addConstr(d_GH[k, g] <= tau[i] + P[i] + M * (1 - x[i, j, k]), name=f"C24_{k}_{g}_{i}_{j}")

# (25) Calculation of waiting time at FF dock [cite: 503]
for k in K:
    for f in F:
        proc_time_ff = gp.quicksum(P[i] * gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) for i in FF_nodes[f])
        m.addConstr(w_FF_docked[k, f] >= d_FF[k, f] - a_FF[k, f] - proc_time_ff, name=f"C25_WaitFF_{k}_{f}")

# (26) Calculation of waiting time at GH dock [cite: 506]
for k in K:
    for g in G:
        proc_time_gh = gp.quicksum(P[i] * gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) for i in GH_nodes[g])
        m.addConstr(w_GH_docked[k, g] >= d_GH[k, g] - a_GH[k, g] - w_dock[k, g] - proc_time_gh, name=f"C26_WaitGH_{k}_{g}")

# (27)-(28) Definition of overlap variable 'eta' [cite: 508, 511]
for g in G:
    for k1 in K:
        for k2 in K:
            if k1 != k2:
                docking_time_k2 = a_GH[k2, g] + w_dock[k2, g]
                departure_time_k1 = d_GH[k1, g]
                # (27)
                m.addConstr(departure_time_k1 - docking_time_k2 + M * eta[k1, k2, g] <= M, name=f"C27_Eta_{k1}_{k2}_{g}")
                # (28)
                m.addConstr(-departure_time_k1 + docking_time_k2 - M * eta[k1, k2, g] <= 0, name=f"C28_Eta_{k1}_{k2}_{g}")

# (29) Truck-to-dock assignment if it visits GH [cite: 514]
for k in K:
    for g in G:
        m.addConstr(gp.quicksum(y[k, g, d] for d in Docks[g]) == visit_GH_expr[k, g], name=f"C29_AssignDock_{k}_{g}")

# (30)-(32) Linealización de z = y * y [cite: 517-524]
for g in G:
    for k1 in K:
        for k2 in K:
            if k1 < k2: # Avoid duplicates [cite: 588]
                for d1 in Docks[g]:
                    for d2 in Docks[g]:
                        # (30)
                        m.addConstr(z[k1, k2, g, d1, d2] <= y[k1, g, d1], name=f"C30_LinZ_{k1}_{k2}_{g}_{d1}_{d2}")
                        # (31)
                        m.addConstr(z[k1, k2, g, d1, d2] <= y[k2, g, d2], name=f"C31_LinZ_{k1}_{k2}_{g}_{d1}_{d2}")
                        # (32)
                        m.addConstr(z[k1, k2, g, d1, d2] >= y[k1, g, d1] + y[k2, g, d2] - 1, name=f"C32_LinZ_{k1}_{k2}_{g}_{d1}_{d2}")

# (33) Non-overlap constraint on the same dock [cite: 526]
for g in G:
    for k1 in K:
        for k2 in K:
            if k1 < k2: # [cite: 588]
                for d in Docks[g]:
                    # If k1 and k2 use the same dock d (z=1), one must finish before the other starts
                    m.addConstr(z[k1, k2, g, d, d] <= eta[k1, k2, g] + eta[k2, k1, g], name=f"C33_NoOverlap_{k1}_{k2}_{g}_{d}")

# (34)-(35) Variable types [cite: 529-534]
# (Already defined when creating the variables)
# Note: z is continuous [0,1] as indicated in the paper [cite: 591, 595]

# --- 7. Optimization ---
print("Iniciando optimización del modelo M1...")
m.setParam("TimeLimit", 60) # Time limit of 60 seconds
m.setParam("MIPGap", 0.1) # Optimality gap of 10%
m.optimize()

# --- 8. Results Display (Basic) ---
if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
    print(f"\nSolución encontrada. Objetivo: {m.ObjVal:.2f}")
    
    print("\nRutas de los camiones:")
    for k in K:
        route = []
        current_node = 0
        
        # Find the first stop
        for i, j in E_1.select(0, '*'):
            if x[0, j, k].X > 0.5:
                route = [0, j]
                current_node = j
                break
        
        # Follow the route
        while current_node != 0 and len(route) < n_nodes:
            found_next = False
            for i, j in E_1.select(current_node, '*'):
                if x[i, j, k].X > 0.5:
                    route.append(j)
                    current_node = j
                    if j == 0:
                        break
                    found_next = True
                    break
            if not found_next or current_node == 0:
                break
                
        if len(route) > 1:
            print(f"  Camión {k}: {' -> '.join(map(str, route))}")
            
    print("\nTiempos de servicio (tau):")
    for i in N_1:
        if tau[i].X > 0.01:
            print(f"  Nodo {i}: {tau[i].X:.2f} (Ventana: [{E[i]}, {D[i]}])")

    print("\nTiempos de espera en muelles GH (w_dock):")
    for k in K:
        for g in G:
            if w_dock[k, g].X > 0.01:
                print(f"  Camión {k} en {g}: {w_dock[k, g].X:.2f} min")

elif m.Status == GRB.INFEASIBLE:
    print("\nEl modelo es infactible con los datos de muestra.")
    print("Computando IIS (Irreducible Inconsistent Subsystem) para depurar...")
    m.computeIIS()
    m.write("model_M1.ilp")
    print("IIS escrito en 'model_M1.ilp'. Revisa este archivo para ver las restricciones en conflicto.")
else:
    print(f"\nNo se encontró solución. Estado de Gurobi: {m.Status}")