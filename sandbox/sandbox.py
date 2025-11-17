import gurobipy as gp
from gurobipy import GRB
import math

# --- 1. Definición de Datos de Muestra ---
# Esta sección crea un pequeño problema de ejemplo.
# n = 3 ULDs
# F = 2 Freight Forwarders ('F1', 'F2')
# G = 1 Ground Handler ('G1')
# K = 2 Camiones ('T1', 'T2')

n = 3  # Número de ULDs [cite: 398]
n_nodes = 2 * n + 1 # Total de nodos (0=depot, 1..n=pickup, n+1..2n=delivery) [cite: 412]

# Conjuntos
N_P = list(range(1, n + 1)) # Nodos de recogida (Pickup) [cite: 397]
N_D = list(range(n + 1, 2 * n + 1)) # Nodos de entrega (Delivery) [cite: 397]
N_1 = [0] + N_P + N_D # Todos los nodos [cite: 412]
F = ['F1', 'F2'] # Conjunto de Freight Forwarders [cite: 335, 430]
G = ['G1'] # Conjunto de Ground Handlers [cite: 333, 430]
K = ['T1', 'T2'] # Conjunto de camiones [cite: 332, 430]

# --- Parámetros de ULDs y Nodos ---
# (Datos ficticios para el ejemplo)

# Mapeo de nodos a almacenes
# ULD 1: F1 -> G1 (Nodos 1 -> 4)
# ULD 2: F1 -> G1 (Nodos 2 -> 5)
# ULD 3: F2 -> G1 (Nodos 3 -> 6)
FF_nodes = {'F1': [1, 2], 'F2': [3]} # Nodos ULD por FF
GH_nodes = {'G1': [4, 5, 6]} # Nodos ULD por GH

# Mapeo inverso de nodo a entidad
node_to_FF = {1: 'F1', 2: 'F1', 3: 'F2'}
node_to_GH = {4: 'G1', 5: 'G1', 6: 'G1'}

# Coordenadas para calcular tiempos de viaje
locations = {
    0: (5, 5),  # Depot
    1: (0, 5),  # F1 (ULD 1)
    2: (0, 5),  # F1 (ULD 2)
    3: (5, 0),  # F2 (ULD 3)
    4: (10, 5), # G1 (ULD 1)
    5: (10, 5), # G1 (ULD 2)
    6: (10, 5)  # G1 (ULD 3)
}

# Tiempos de viaje T_ij [cite: 338]
def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

T = {}
for i in N_1:
    for j in N_1:
        T[i, j] = euclidean_dist(locations[i], locations[j])

# Parámetros de ULD (peso, longitud, tiempo proc.)
# W, L, P se definen por nodo i
W = {1: 10, 2: 12, 3: 15} # Peso (sólo para nodos pickup) [cite: 329, 430]
L = {1: 1, 2: 1, 3: 1.5} # Longitud (sólo para nodos pickup) [cite: 329, 430]
P = {i: 5 for i in N_P} # Tiempo de procesamiento en pickup [cite: 331, 430]
P.update({i: 6 for i in N_D}) # Tiempo de procesamiento en delivery
P[0] = 0 # Sin tiempo de procesamiento en depot

# Ventanas de tiempo [E_i, D_i] [cite: 417, 430]
E = {0: 0, 1: 10, 2: 10, 3: 15, 4: 50, 5: 50, 6: 60}
D = {0: 999, 1: 100, 2: 100, 3: 110, 4: 200, 5: 200, 6: 210}

# Parámetros de Flota y Muelle
Q_W = 30 # Capacidad de peso del camión [cite: 332, 430]
Q_L = 3 # Capacidad de longitud del camión [cite: 332, 430]
DELTA_GH = {'G1': 1} # Número de muelles en cada GH [cite: 339, 430]
M = 10000 # Constante Big-M

# --- 2. Construcción del Conjunto de Arcos (E_1) ---
# E_1 contiene arcos factibles según LIFO y precedencia [cite: 413-415]
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

# --- 3. Creación del Modelo Gurobi ---
m = gp.Model("GHDC_PDPTW_M1")

# --- 4. Definición de Variables ---
# [cite: 419-427, 430]
# x_ij^k: binaria, 1 si camión k va de i a j
x = m.addVars(E_1, K, vtype=GRB.BINARY, name="x")

# tau_i: continua, tiempo de inicio de servicio en nodo i
tau = m.addVars(N_1, vtype=GRB.CONTINUOUS, name="tau")

# tau_0^k / tau_end^k: inicio/fin de servicio del camión k
# Nota: Usamos tau[0] para el inicio. tau_end se define para la constr. (13)
tau_end = m.addVars(K, vtype=GRB.CONTINUOUS, name="tau_end") # [cite: 422, 430]

# a_kf^F / d_kf^F: tiempos de llegada/salida del camión k al FF f
a_FF = m.addVars(K, F, vtype=GRB.CONTINUOUS, name="a_FF")
d_FF = m.addVars(K, F, vtype=GRB.CONTINUOUS, name="d_FF")

# a_kg^G / d_kg^G: tiempos de llegada/salida del camión k al GH g
a_GH = m.addVars(K, G, vtype=GRB.CONTINUOUS, name="a_GH")
d_GH = m.addVars(K, G, vtype=GRB.CONTINUOUS, name="d_GH")

# w_kg^D: tiempo de espera del camión k en GH g (antes de muelle)
w_dock = m.addVars(K, G, vtype=GRB.CONTINUOUS, name="w_dock")

# w_kf^F / w_kg^G: tiempo de espera del camión k en FF f / GH g (en muelle)
w_FF_docked = m.addVars(K, F, vtype=GRB.CONTINUOUS, name="w_FF_docked")
w_GH_docked = m.addVars(K, G, vtype=GRB.CONTINUOUS, name="w_GH_docked")

# eta_{k1,k2}^g: binaria, 1 si k1 termina <= k2 atraca en GH g
eta = m.addVars(K, K, G, vtype=GRB.BINARY, name="eta")

# y_kd^g: binaria, 1 si camión k es asignado al muelle d en GH g
# Creamos un conjunto de muelles para cada GH
Docks = {}
for g in G:
    Docks[g] = list(range(1, DELTA_GH[g] + 1))
y = m.addVars(K, G, [d for g in G for d in Docks[g]], vtype=GRB.BINARY, name="y")

# z_{k1,d1,k2,d2}^g: variable de linealización (continua, como en el paper) [cite: 591, 595]
z = m.addVars(K, K, G, [d1 for g in G for d1 in Docks[g]], [d2 for g in G for d2 in Docks[g]],
              vtype=GRB.CONTINUOUS, name="z")

# --- 5. Función Objetivo ---
# (1) Minimizar tiempo total de transporte y espera [cite: 435]
travel_time = gp.quicksum(T[i, j] * x[i, j, k] for i, j in E_1 for k in K)
wait_time_dock = gp.quicksum(w_dock[k, g] for k in K for g in G)
wait_time_at_FF = gp.quicksum(w_FF_docked[k, f] for k in K for f in F)
# El paper incluye w_kg^G en el objetivo en la eq (1) [cite: 435], pero no en el texto [cite: 536]
# Lo incluimos para ser fieles a la ecuación (1):
wait_time_at_GH = gp.quicksum(w_GH_docked[k, g] for k in K for g in G)

# El paper suma el tiempo de espera en FFs dos veces [cite: 435]
# (probablemente un error tipográfico, sum(w_kg^D) + sum(w_kf^F)).
# Seguiremos la descripción de texto [cite: 536-537] y la eq (26) [cite: 506]
# que implica que w_kg^G también es un tiempo de espera.
# Objetivo: Suma de tiempo de viaje + espera antes de muelle (GH) + espera en muelle (FF) + espera en muelle (GH)
m.setObjective(travel_time + wait_time_dock + wait_time_at_FF + wait_time_at_GH, GRB.MINIMIZE)


# --- 6. Restricciones ---

# (2) Cada ULD es recogido exactamente una vez [cite: 437]
m.addConstrs((gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i for k in K) == 1
              for i in N_P), name="C2_PickupOnce")

# (3) Mismo camión para par (i, i+n) [cite: 440]
for k in K:
    for i in N_P:
        m.addConstr(gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) - 
                    gp.quicksum(x[j, i + n, k] for j, i_n_k in E_1.select('*', i + n) if i_n_k == i + n) == 0,
                    name=f"C3_SameTruck_{k}_{i}")

# (4) Cada camión se usa como máximo una vez (sale del depot) [cite: 442]
m.addConstrs((gp.quicksum(x[0, j, k] for j_k, j in E_1.select(0, '*') if j_k == 0) <= 1
              for k in K), name="C4_UseTruckOnce")

# (5) Conservación de flujo [cite: 444]
for k in K:
    for i in N_P + N_D:
        m.addConstr(gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) -
                    gp.quicksum(x[i, j, k] for i_k, j in E_1.select(i, '*') if i_k == i) == 0,
                    name=f"C5_FlowCons_{k}_{i}")

# (6) LIFO: Primer pickup (desde depot) -> Última entrega (a depot) [cite: 447]
m.addConstrs((x[0, i, k] - x[i + n, 0, k] == 0
              for i in N_P for k in K), name="C6_LIFO_StartEnd")

# (7) LIFO: Secuencia de pickup (i,j) -> Secuencia de delivery (j+n, i+n) [cite: 450]
m.addConstrs((x[i, j, k] - x[j + n, i + n, k] == 0
              for i in N_P for j in N_P if i != j for k in K), name="C7_LIFO_Seq")

# (8) Cada camión visita cada FF como máximo una vez [cite: 453]
for k in K:
    for f in F:
        m.addConstr(gp.quicksum(x[j, i, k] for i in FF_nodes[f] for j, i_k in E_1.select('*', i) if i_k == i and node_to_FF.get(j, -1) != f) <= 1,
                    name=f"C8_VisitFFOnce_{k}_{f}")

# (9) Cada camión visita cada GH como máximo una vez [cite: 455]
# (Usaremos esta expresión también para la C29)
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
        if i in N_D: # Solo desde nodos de delivery
            m.addConstr(tau_end[k] >= tau[i] + P[i] + T[i, 0] - M * (1 - x[i, 0, k]),
                        name=f"C13_TimePrec_toDepot_{k}_{i}")

# (14) Ventanas de tiempo [cite: 467]
m.addConstrs((tau[i] >= E[i] for i in N_1), name="C14_TimeWin_Early")
m.addConstrs((tau[i] <= D[i] for i in N_1), name="C14_TimeWin_Late")
m.addConstrs((tau_end[k] <= D[0] for k in K), name="C14_TimeWin_End") # Asumimos D[0] es el fin del horizonte

# (15) Capacidad de Peso [cite: 470]
m.addConstrs((gp.quicksum(W[i] * gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) for i in N_P) <= Q_W
              for k in K), name="C15_Cap_Weight")

# (16) Capacidad de Longitud [cite: 473]
m.addConstrs((gp.quicksum(L[i] * gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) for i in N_P) <= Q_L
              for k in K), name="C16_Cap_Length")

# (17)-(20) Tiempos de llegada y salida en FF [cite: 476-487]
for k in K:
    for f in F:
        # Encuentra el tiempo de llegada (a_FF)
        for i, j in E_1:
            if j in FF_nodes[f] and node_to_FF.get(i, -1) != f:
                m.addConstr(a_FF[k, f] >= tau[i] + P[i] + T[i, j] - M * (1 - x[i, j, k]), name=f"C17_{k}_{f}_{i}_{j}")
                m.addConstr(a_FF[k, f] <= tau[i] + P[i] + T[i, j] + M * (1 - x[i, j, k]), name=f"C18_{k}_{f}_{i}_{j}")
        # Encuentra el tiempo de salida (d_FF)
        for i, j in E_1:
             if i in FF_nodes[f] and node_to_FF.get(j, -1) != f:
                m.addConstr(d_FF[k, f] >= tau[i] + P[i] - M * (1 - x[i, j, k]), name=f"C19_{k}_{f}_{i}_{j}")
                m.addConstr(d_FF[k, f] <= tau[i] + P[i] + M * (1 - x[i, j, k]), name=f"C20_{k}_{f}_{i}_{j}")

# (21)-(24) Tiempos de llegada y salida en GH [cite: 488-498]
for k in K:
    for g in G:
        # Encuentra el tiempo de llegada (a_GH)
        for i, j in E_1:
            if j in GH_nodes[g] and node_to_GH.get(i, -1) != g:
                m.addConstr(a_GH[k, g] >= tau[i] + P[i] + T[i, j] - M * (1 - x[i, j, k]), name=f"C21_{k}_{g}_{i}_{j}")
                m.addConstr(a_GH[k, g] <= tau[i] + P[i] + T[i, j] + M * (1 - x[i, j, k]), name=f"C22_{k}_{g}_{i}_{j}")
        # Encuentra el tiempo de salida (d_GH)
        for i, j in E_1:
             if i in GH_nodes[g] and node_to_GH.get(j, -1) != g:
                m.addConstr(d_GH[k, g] >= tau[i] + P[i] - M * (1 - x[i, j, k]), name=f"C23_{k}_{g}_{i}_{j}")
                m.addConstr(d_GH[k, g] <= tau[i] + P[i] + M * (1 - x[i, j, k]), name=f"C24_{k}_{g}_{i}_{j}")

# (25) Cálculo del tiempo de espera en muelle FF [cite: 503]
for k in K:
    for f in F:
        proc_time_ff = gp.quicksum(P[i] * gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) for i in FF_nodes[f])
        m.addConstr(w_FF_docked[k, f] >= d_FF[k, f] - a_FF[k, f] - proc_time_ff, name=f"C25_WaitFF_{k}_{f}")

# (26) Cálculo del tiempo de espera en muelle GH [cite: 506]
for k in K:
    for g in G:
        proc_time_gh = gp.quicksum(P[i] * gp.quicksum(x[j, i, k] for j, i_k in E_1.select('*', i) if i_k == i) for i in GH_nodes[g])
        m.addConstr(w_GH_docked[k, g] >= d_GH[k, g] - a_GH[k, g] - w_dock[k, g] - proc_time_gh, name=f"C26_WaitGH_{k}_{g}")

# (27)-(28) Definición de la variable de solapamiento 'eta' [cite: 508, 511]
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

# (29) Asignación de camión a muelle si visita GH [cite: 514]
for k in K:
    for g in G:
        m.addConstr(gp.quicksum(y[k, g, d] for d in Docks[g]) == visit_GH_expr[k, g], name=f"C29_AssignDock_{k}_{g}")

# (30)-(32) Linealización de z = y * y [cite: 517-524]
for g in G:
    for k1 in K:
        for k2 in K:
            if k1 < k2: # Evitar duplicados [cite: 588]
                for d1 in Docks[g]:
                    for d2 in Docks[g]:
                        # (30)
                        m.addConstr(z[k1, k2, g, d1, d2] <= y[k1, g, d1], name=f"C30_LinZ_{k1}_{k2}_{g}_{d1}_{d2}")
                        # (31)
                        m.addConstr(z[k1, k2, g, d1, d2] <= y[k2, g, d2], name=f"C31_LinZ_{k1}_{k2}_{g}_{d1}_{d2}")
                        # (32)
                        m.addConstr(z[k1, k2, g, d1, d2] >= y[k1, g, d1] + y[k2, g, d2] - 1, name=f"C32_LinZ_{k1}_{k2}_{g}_{d1}_{d2}")

# (33) Restricción de no solapamiento en el mismo muelle [cite: 526]
for g in G:
    for k1 in K:
        for k2 in K:
            if k1 < k2: # [cite: 588]
                for d in Docks[g]:
                    # Si k1 y k2 usan el mismo muelle d (z=1), entonces uno debe terminar antes de que el otro empiece
                    m.addConstr(z[k1, k2, g, d, d] <= eta[k1, k2, g] + eta[k2, k1, g], name=f"C33_NoOverlap_{k1}_{k2}_{g}_{d}")

# (34)-(35) Tipos de variables [cite: 529-534]
# (Ya definidos al crear las variables)
# Nota: z es continua [0,1] como se indica en el paper [cite: 591, 595]

# --- 7. Optimización ---
print("Iniciando optimización del modelo M1...")
m.setParam("TimeLimit", 60) # Límite de tiempo de 60 segundos
m.setParam("MIPGap", 0.1) # Gap de optimalidad del 10%
m.optimize()

# --- 8. Presentación de Resultados (Básico) ---
if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
    print(f"\nSolución encontrada. Objetivo: {m.ObjVal:.2f}")
    
    print("\nRutas de los camiones:")
    for k in K:
        route = []
        current_node = 0
        
        # Encontrar la primera parada
        for i, j in E_1.select(0, '*'):
            if x[0, j, k].X > 0.5:
                route = [0, j]
                current_node = j
                break
        
        # Seguir la ruta
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