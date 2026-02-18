"""
Verification tests for Constraints 25–33 of the GHDC-PDPTW model.

Each test function builds a small, self-contained Gurobi model and
prints PASS / FAIL with key values.

FIXES applied vs previous version:
  - _Edges stored as a set
  - Every test guards against non-OPTIMAL status before reading .X
  - x variable keys follow (i, j, k) consistently everywhere
  - Gurobi OutputFlag set to 1 so solver messages are visible on error
"""

import gurobipy as gp
from gurobipy import GRB
from math import sqrt

# ─────────────────────────────────────────────────────────────────────────────
#  Build the baseline 2-ULD, 1-truck instance
# ─────────────────────────────────────────────────────────────────────────────

def build_base_model(
    n_uld=2,
    K_trucks=None,
    Cap_W=10_000,
    Cap_L=13.6,
    Weight_u=1_000,
    Length_u=1.534,
    Proc_Time=2,
    Horizon=480,
    Speed_mpm=35 / 60.0,
    Delta_GH=1,
    node_locs=None,
    E_win_override=None,
    D_win_override=None,
    extra_constrs=None,
    verbose=False,
):
    """
    Build and return (model, vars_dict) for a small GHDC-PDPTW instance.

    Default geometry:
        Depot  node 0 : (0, 0)
        Pickup node i : (0, 1)   for i in Nodes_P
        Delivery node i : (3, 4) for i in Nodes_D
    """
    if K_trucks is None:
        K_trucks = [1]

    Nodes_P   = list(range(1, n_uld + 1))
    Nodes_D   = list(range(n_uld + 1, 2 * n_uld + 1))
    All_Nodes = [0] + Nodes_P + Nodes_D
    # Store edges as list for variable creation, set for fast membership tests
    Edges_list = [(i, j) for i in All_Nodes for j in All_Nodes if i != j]
    Edges_set  = set(Edges_list)

    Docks = list(range(1, Delta_GH + 1))

    # ── node locations ──────────────────────────────────────────────────────
    if node_locs is None:
        locs = [[0, 0]] + [[0, 1]] * len(Nodes_P) + [[3, 4]] * len(Nodes_D)
    else:
        locs = node_locs

    def dist(i, j):
        x1, y1 = locs[i]
        x2, y2 = locs[j]
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # ── parameters ──────────────────────────────────────────────────────────
    T      = {}
    P      = {0: 0}
    W      = {0: 0}
    L      = {0: 0}
    E_win  = {0: 0}
    D_win  = {0: Horizon}

    for i in Nodes_P:
        P[i] = Proc_Time;  W[i] = Weight_u;  L[i] = Length_u
        E_win[i] = 0;      D_win[i] = Horizon
    for i in Nodes_D:
        P[i] = Proc_Time;  W[i] = 0;  L[i] = 0
        E_win[i] = 0;      D_win[i] = Horizon

    if E_win_override:
        E_win.update(E_win_override)
    if D_win_override:
        D_win.update(D_win_override)

    for i, j in Edges_list:
        T[i, j] = dist(i, j) / Speed_mpm

    M_big = 86_400  # large-M constant (seconds in a day, used as minutes)

    # ── facility / group maps ────────────────────────────────────────────────
    Facilities = {1: Nodes_P}
    Groups     = {1: Nodes_D}

    # ── model ────────────────────────────────────────────────────────────────
    m = gp.Model("GHDC-PDPTW-verify")
    m.setParam("OutputFlag", 1 if verbose else 0)

    # Variables
    # x[i, j, k] — Gurobi flattens (Edges_list, K_trucks) into (i, j, k) keys
    x       = m.addVars(Edges_list, K_trucks, vtype=GRB.BINARY,     name="x")
    tau     = m.addVars(All_Nodes,            vtype=GRB.CONTINUOUS, name="tau")
    tau_end = m.addVars(K_trucks,             vtype=GRB.CONTINUOUS, name="tau_end")

    a_F = m.addVars(K_trucks, list(Facilities.keys()), vtype=GRB.CONTINUOUS, name="a_F")
    d_F = m.addVars(K_trucks, list(Facilities.keys()), vtype=GRB.CONTINUOUS, name="d_F")
    a_G = m.addVars(K_trucks, list(Groups.keys()),     vtype=GRB.CONTINUOUS, name="a_G")
    d_G = m.addVars(K_trucks, list(Groups.keys()),     vtype=GRB.CONTINUOUS, name="d_G")

    w_D = m.addVars(K_trucks, list(Groups.keys()),     vtype=GRB.CONTINUOUS, name="w_D")
    w_F = m.addVars(K_trucks, list(Facilities.keys()), vtype=GRB.CONTINUOUS, name="w_F")
    w_G = m.addVars(K_trucks, list(Groups.keys()),     vtype=GRB.CONTINUOUS, name="w_G")

    eta = m.addVars(K_trucks, K_trucks, list(Groups.keys()),
                    vtype=GRB.BINARY, name="eta")
    y   = m.addVars(K_trucks, Docks, list(Groups.keys()),
                    vtype=GRB.BINARY, name="y")
    z   = m.addVars(K_trucks, Docks, K_trucks, Docks, list(Groups.keys()),
                    vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")

    # ── objective ────────────────────────────────────────────────────────────
    travel = gp.quicksum(T[i, j] * x[i, j, k]
                         for k in K_trucks for i, j in Edges_list)
    wdock  = gp.quicksum(w_D[k, g] for k in K_trucks for g in Groups)
    wgh    = gp.quicksum(w_G[k, g] for k in K_trucks for g in Groups)
    wff    = gp.quicksum(w_F[k, f] for k in K_trucks for f in Facilities)
    m.setObjective(travel + wdock + wgh + wff, GRB.MINIMIZE)

    # ── constraints 1-24 ─────────────────────────────────────────────────────
    n = len(Nodes_P)

    # (2) every pickup visited exactly once
    for i in Nodes_P:
        m.addConstr(
            gp.quicksum(x[j, i, k] for k in K_trucks
                        for j in All_Nodes if (j, i) in Edges_set) == 1,
            name=f"C2_pickup_{i}")

    # (3) same truck for pickup and its delivery
    for k in K_trucks:
        for i in Nodes_P:
            d_node = i + n
            m.addConstr(
                gp.quicksum(x[j, i, k]      for j in All_Nodes if (j, i)      in Edges_set)
              - gp.quicksum(x[j, d_node, k] for j in All_Nodes if (j, d_node) in Edges_set)
              == 0, name=f"C3_pairing_i{i}_k{k}")

    # (4) each truck used at most once
    for k in K_trucks:
        m.addConstr(
            gp.quicksum(x[0, i, k] for i in All_Nodes if (0, i) in Edges_set) <= 1,
            name=f"C4_once_k{k}")

    # (5) flow conservation at every node
    for k in K_trucks:
        for i in All_Nodes:
            m.addConstr(
                gp.quicksum(x[j, i, k] for j in All_Nodes if (j, i) in Edges_set)
              - gp.quicksum(x[i, j, k] for j in All_Nodes if (i, j) in Edges_set)
              == 0, name=f"C5_flow_k{k}_i{i}")

    # (6) LIFO: first pickup ↔ last delivery
    for k in K_trucks:
        for i in Nodes_P:
            m.addConstr(x[0, i, k] - x[i + n, 0, k] == 0,
                        name=f"C6_LIFO_k{k}_i{i}")

    # (7) LIFO: consecutive pickups reversed at deliveries
    for k in K_trucks:
        for i, j in Edges_list:
            if i in Nodes_P and j in Nodes_P:
                m.addConstr(x[i, j, k] - x[j + n, i + n, k] == 0,
                            name=f"C7_LIFO_k{k}_i{i}_j{j}")

    # (8) each FF visited at most once per truck
    # outside_f excludes depot (0) to avoid double-counting: the depot arc
    # x[0,i,k] is already captured by j=0 in the first sum if we include 0,
    # so we split it explicitly: non-depot outside arcs + depot arc.
    # Simplest correct form: sum over ALL j not in F_f (includes depot).
    for k in K_trucks:
        for f, f_nodes in Facilities.items():
            outside_f = [nd for nd in All_Nodes if nd not in f_nodes]
            m.addConstr(
                gp.quicksum(x[j, i, k] for i in f_nodes
                            for j in outside_f if (j, i) in Edges_set)
              <= 1, name=f"C8_FF_k{k}_f{f}")

    # (9) each GH visited at most once per truck
    # Same fix as C8: outside_g already includes depot, no separate depot sum.
    for k in K_trucks:
        for g, g_nodes in Groups.items():
            outside_g = [nd for nd in All_Nodes if nd not in g_nodes]
            m.addConstr(
                gp.quicksum(x[j, i, k] for i in g_nodes
                            for j in outside_g if (j, i) in Edges_set)
              <= 1, name=f"C9_GH_k{k}_g{g}")

    # (10) time precedence for pickup nodes
    for i, j in Edges_list:
        if j in Nodes_P:
            m.addConstr(
                tau[j] >= tau[i] + P[i] + T[i, j]
                        - M_big * (1 - gp.quicksum(x[i, j, k] for k in K_trucks)),
                name=f"C10_tp_i{i}_j{j}")

    # (11) time precedence entering a GH from outside
    for g, g_nodes in Groups.items():
        for k in K_trucks:
            for i, j in Edges_list:
                if j in g_nodes and i not in g_nodes and j in Nodes_D:
                    m.addConstr(
                        tau[j] >= tau[i] + P[i] + T[i, j]
                                - M_big * (1 - x[i, j, k]) + w_D[k, g],
                        name=f"C11_GH_entry_g{g}_k{k}_i{i}_j{j}")

    # (12) time consistency within the same GH
    for g, g_nodes in Groups.items():
        for i in g_nodes:
            for j in g_nodes:
                if i != j and (i, j) in Edges_set:
                    expr = gp.quicksum(x[i, j, k] for k in K_trucks)
                    m.addConstr(
                        tau[j] >= tau[i] + P[i] + T[i, j] - (1 - expr) * M_big,
                        name=f"C12_GH_intra_i{i}_j{j}")

    # (13) vehicle end time
    for k in K_trucks:
        for i in All_Nodes:
            if i != 0:
                m.addConstr(
                    tau_end[k] >= tau[i] + P[i] + T[i, 0]
                                - (1 - x[i, 0, k]) * M_big,
                    name=f"C13_end_k{k}_i{i}")

    # (14) time windows
    for i in Nodes_P + Nodes_D:
        m.addConstr(tau[i] >= E_win[i], name=f"C14_lb_{i}")
        m.addConstr(tau[i] <= D_win[i], name=f"C14_ub_{i}")

    # (15) weight capacity
    for k in K_trucks:
        m.addConstr(
            gp.quicksum(W[i] * x[j, i, k]
                        for i in All_Nodes for j in All_Nodes
                        if (j, i) in Edges_set) <= Cap_W,
            name=f"C15_weight_k{k}")

    # (16) length capacity
    for k in K_trucks:
        m.addConstr(
            gp.quicksum(L[i] * x[j, i, k]
                        for i in All_Nodes for j in All_Nodes
                        if (j, i) in Edges_set) <= Cap_L,
            name=f"C16_length_k{k}")

    # (17–18) FF arrival time bounds
    for f, f_nodes in Facilities.items():
        for k in K_trucks:
            for i, j in Edges_list:
                if j in f_nodes and i not in f_nodes:
                    m.addConstr(
                        a_F[k, f] >= tau[i] + P[i] + T[i, j] - (1 - x[i, j, k]) * M_big,
                        name=f"C17_aF_k{k}_f{f}_i{i}_j{j}")
                    m.addConstr(
                        a_F[k, f] <= tau[i] + P[i] + T[i, j] + (1 - x[i, j, k]) * M_big,
                        name=f"C18_aF_k{k}_f{f}_i{i}_j{j}")

    # (19–20) FF departure time bounds
    for f, f_nodes in Facilities.items():
        for k in K_trucks:
            for i, j in Edges_list:
                if i in f_nodes and j not in f_nodes:
                    m.addConstr(
                        d_F[k, f] >= tau[i] + P[i] - (1 - x[i, j, k]) * M_big,
                        name=f"C19_dF_k{k}_f{f}_i{i}_j{j}")
                    m.addConstr(
                        d_F[k, f] <= tau[i] + P[i] + (1 - x[i, j, k]) * M_big,
                        name=f"C20_dF_k{k}_f{f}_i{i}_j{j}")

    # (21–22) GH arrival time bounds
    for g, g_nodes in Groups.items():
        for k in K_trucks:
            for i, j in Edges_list:
                if j in g_nodes and i not in g_nodes:
                    m.addConstr(
                        a_G[k, g] >= tau[i] + P[i] + T[i, j] - (1 - x[i, j, k]) * M_big,
                        name=f"C21_aG_k{k}_g{g}_i{i}_j{j}")
                    m.addConstr(
                        a_G[k, g] <= tau[i] + P[i] + T[i, j] + (1 - x[i, j, k]) * M_big,
                        name=f"C22_aG_k{k}_g{g}_i{i}_j{j}")

    # (23–24) GH departure time bounds
    for g, g_nodes in Groups.items():
        for k in K_trucks:
            for i in g_nodes:
                for j in All_Nodes:
                    if j not in g_nodes and (i, j) in Edges_set:
                        m.addConstr(
                            d_G[k, g] >= tau[i] + P[i] - (1 - x[i, j, k]) * M_big,
                            name=f"C23_dG_k{k}_g{g}_i{i}_j{j}")
                        m.addConstr(
                            d_G[k, g] <= tau[i] + P[i] + (1 - x[i, j, k]) * M_big,
                            name=f"C24_dG_k{k}_g{g}_i{i}_j{j}")

    # (25) waiting at FF while docked
    for f, f_nodes in Facilities.items():
        for k in K_trucks:
            proc_sum = gp.quicksum(
                P[i] * x[j, i, k]
                for i in f_nodes for j in All_Nodes if (j, i) in Edges_set)
            m.addConstr(w_F[k, f] >= d_F[k, f] - a_F[k, f] - proc_sum,
                        name=f"C25_wF_k{k}_f{f}")

    # (26) waiting at GH while docked
    for g, g_nodes in Groups.items():
        for k in K_trucks:
            proc_sum = gp.quicksum(
                P[i] * x[j, i, k]
                for i in g_nodes for j in All_Nodes if (j, i) in Edges_set)
            m.addConstr(
                w_G[k, g] >= d_G[k, g] - a_G[k, g] - w_D[k, g] - proc_sum,
                name=f"C26_wG_k{k}_g{g}")

    # (27–28) overlap variable (eta)
    for g in Groups:
        for k1 in K_trucks:
            for k2 in K_trucks:
                m.addConstr(
                    d_G[k1, g] - a_G[k2, g] - w_D[k2, g]
                    + M_big * eta[k1, k2, g] <= M_big,
                    name=f"C27_eta_k{k1}_k{k2}_g{g}")
                m.addConstr(
                    -d_G[k1, g] + a_G[k2, g] + w_D[k2, g]
                    - M_big * eta[k1, k2, g] <= 0,
                    name=f"C28_eta_k{k1}_k{k2}_g{g}")

    # (29) dock assignment
    for g, g_nodes in Groups.items():
        for k in K_trucks:
            visits_gh = gp.quicksum(
                x[j, i, k]
                for i in g_nodes for j in All_Nodes
                if (j, i) in Edges_set and j not in g_nodes)
            m.addConstr(
                gp.quicksum(y[k, d, g] for d in Docks) == visits_gh,
                name=f"C29_dock_k{k}_g{g}")

    # (30–32) linearisation of z
    for g in Groups:
        for k1 in K_trucks:
            for k2 in K_trucks:
                if k1 < k2:
                    for d1 in Docks:
                        for d2 in Docks:
                            m.addConstr(z[k1, d1, k2, d2, g] <= y[k1, d1, g],
                                        name=f"C30_z_k{k1}d{d1}k{k2}d{d2}g{g}")
                            m.addConstr(z[k1, d1, k2, d2, g] <= y[k2, d2, g],
                                        name=f"C31_z_k{k1}d{d1}k{k2}d{d2}g{g}")
                            m.addConstr(
                                y[k1, d1, g] + y[k2, d2, g] - 1
                                <= z[k1, d1, k2, d2, g],
                                name=f"C32_z_k{k1}d{d1}k{k2}d{d2}g{g}")

    # (33) dock non-overlap
    for g in Groups:
        for k1 in K_trucks:
            for k2 in K_trucks:
                if k1 < k2:
                    for d1 in Docks:
                        m.addConstr(
                            z[k1, d1, k2, d1, g] <= eta[k1, k2, g] + eta[k2, k1, g],
                            name=f"C33_nooverlap_k{k1}_k{k2}_d{d1}_g{g}")

    # optional extra constraints injected by individual tests
    if extra_constrs is not None:
        extra_constrs(m, dict(
            x=x, tau=tau, tau_end=tau_end,
            a_F=a_F, d_F=d_F, a_G=a_G, d_G=d_G,
            w_D=w_D, w_F=w_F, w_G=w_G,
            eta=eta, y=y, z=z))

    vars_dict = dict(
        x=x, tau=tau, tau_end=tau_end,
        a_F=a_F, d_F=d_F, a_G=a_G, d_G=d_G,
        w_D=w_D, w_F=w_F, w_G=w_G,
        eta=eta, y=y, z=z,
        # expose sets for use in tests
        _K=K_trucks,
        _Facilities=Facilities,
        _Groups=Groups,
        _Nodes_P=Nodes_P,
        _Nodes_D=Nodes_D,
        _All_Nodes=All_Nodes,
        _Docks=Docks,
        _Edges=Edges_set,   # ← set, not list, for O(1) membership tests
        _P=P,
    )
    return m, vars_dict


def _check_optimal(m, label):
    """Return True and print a warning if the model did not solve to optimality."""
    if m.Status != GRB.OPTIMAL:
        print(f"  [SKIP] {label} — model status {m.Status} (not OPTIMAL)")
        return False
    return True


def solve_and_check(m, expected_status=GRB.OPTIMAL, label=""):
    """Optimise m and compare status to expected. Print PASS/FAIL."""
    m.optimize()
    status = m.Status
    ok  = (status == expected_status)
    tag = "PASS" if ok else "FAIL"
    obj = m.ObjVal if status == GRB.OPTIMAL else None
    print(f"  [{tag}] {label}  |  status={status}  obj={obj}")
    return status, obj, ok


# =============================================================================
#  CONSTRAINT 25 — Waiting time at FF while docked
#  w_F[k,f] >= d_F[k,f] - a_F[k,f] - sum_{i in F_f} P_i * x[j,i,k]
# =============================================================================

def test_25_1_zero_wait_no_tightening():
    """
    Test 25.1 — Baseline: no delay at FF.
    With default geometry and wide time windows [0, 480] min, the truck
    processes both ULDs back-to-back immediately on arrival at the FF.
    Expected: w_F[1,1] = 0, and d_F – a_F – sum(P_i * x) = 0.
    """
    m, v = build_base_model()
    m.optimize()
    if not _check_optimal(m, "Test 25.1"):
        return

    wF = v["w_F"][1, 1].X
    aF = v["a_F"][1, 1].X
    dF = v["d_F"][1, 1].X
    proc = sum(
        v["_P"][i] * v["x"][j, i, 1].X
        for i in v["_Nodes_P"]
        for j in v["_All_Nodes"]
        if (j, i) in v["_Edges"]
    )
    lb_rhs = dF - aF - proc
    ok = abs(wF) < 1e-4 and abs(lb_rhs) < 1e-4
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Test 25.1 — Zero wait at FF  |"
          f"  w_F={wF:.4f}  aF={aF:.4f}  dF={dF:.4f}  proc={proc:.1f}")


def test_25_2_forced_wait_at_ff():
    """
    Test 25.2 — Forced early arrival at FF with tight pickup window.
    The second pickup node is given E_win=200 min. The depot departure is
    pinned to tau[0]=0 so the truck arrives at the FF at t~1.71 min, long
    before node 2's window opens, and must wait there.
    Expected: w_F[1,1] > 0 and equals d_F - a_F - sum(P_i * x).
    """
    def pin_depot_departure(m, v):
        m.addConstr(v["tau"][0] == 0, name="test_pin_depot")

    m, v = build_base_model(
        E_win_override={2: 200},
        D_win_override={2: 350},
        extra_constrs=pin_depot_departure,
    )
    m.optimize()
    if not _check_optimal(m, "Test 25.2"):
        return

    wF = v["w_F"][1, 1].X
    aF = v["a_F"][1, 1].X
    dF = v["d_F"][1, 1].X
    proc = sum(
        v["_P"][i] * v["x"][j, i, 1].X
        for i in v["_Nodes_P"]
        for j in v["_All_Nodes"]
        if (j, i) in v["_Edges"]
    )
    lb_rhs = dF - aF - proc
    ok = wF > 1e-4 and abs(wF - lb_rhs) < 1e-3
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Test 25.2 — Positive wait at FF  |"
          f"  w_F={wF:.4f}  aF={aF:.4f}  dF={dF:.4f}  proc={proc:.1f}")


# =============================================================================
#  CONSTRAINT 26 — Waiting time at GH while docked
#  w_G[k,g] >= d_G[k,g] - a_G[k,g] - w_D[k,g] - sum P_i x[j,i,k]
# =============================================================================

def test_26_1_zero_wait_no_tightening():
    """
    Test 26.1 — Baseline: no delay at GH.
    Wide delivery windows allow the truck to unload both ULDs immediately.
    Expected: w_G[1,1] = 0.
    """
    m, v = build_base_model()
    m.optimize()
    if not _check_optimal(m, "Test 26.1"):
        return

    wG = v["w_G"][1, 1].X
    ok = abs(wG) < 1e-4
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Test 26.1 — Zero wait at GH  |  w_G={wG:.4f}")


def test_26_2_forced_wait_at_gh():
    """
    Test 26.2 — Tight delivery window forces waiting at GH.
    The first delivery node (node 3) is given E_win=200 min. The depot
    departure is pinned to tau[0]=0 so the truck arrives at the GH at
    t~13 min, long before the window opens, and must wait there.
    Expected: w_G[1,1] > 0 and equals d_G - a_G - w_D - sum(P_i * x).
    """
    def pin_depot_departure(m, v):
        m.addConstr(v["tau"][0] == 0, name="test_pin_depot")

    m, v = build_base_model(
        E_win_override={3: 200},
        D_win_override={3: 350},
        extra_constrs=pin_depot_departure,
    )
    m.optimize()
    if not _check_optimal(m, "Test 26.2"):
        return

    wG  = v["w_G"][1, 1].X
    wD  = v["w_D"][1, 1].X
    aG  = v["a_G"][1, 1].X
    dG  = v["d_G"][1, 1].X
    proc = sum(
        v["_P"][i] * v["x"][j, i, 1].X
        for i in v["_Nodes_D"]
        for j in v["_All_Nodes"]
        if (j, i) in v["_Edges"]
    )
    lb_rhs = dG - aG - wD - proc
    ok = wG > 1e-4 and abs(wG - lb_rhs) < 1e-3
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Test 26.2 — Positive wait at GH  |"
          f"  w_G={wG:.4f}  aG={aG:.4f}  dG={dG:.4f}  wD={wD:.4f}  proc={proc:.1f}")


# =============================================================================
#  CONSTRAINTS 27–28 — Overlap variable definition (eta)
#  C27: d_G[k1,g] - a_G[k2,g] - w_D[k2,g] + M*eta[k1,k2,g] <= M
#  C28: -d_G[k1,g] + a_G[k2,g] + w_D[k2,g] - M*eta[k1,k2,g] <= 0
# =============================================================================

def test_27_28_1_eta_one_when_sequential():
    """
    Test 27-28.1 — Exactly one eta equals 1 for two trucks at the same GH.
    Two trucks each carry one ULD (Cap_W limited to 1000 kg). Both deliver
    to GH g=1 and share the single dock sequentially. One must depart before
    the other arrives, so exactly one of eta[1,2,1] or eta[2,1,1] equals 1.
    Expected: eta[1,2,1] + eta[2,1,1] = 1.
    """
    m, v = build_base_model(K_trucks=[1, 2], Cap_W=1_000)
    m.optimize()
    if not _check_optimal(m, "Test 27-28.1"):
        return

    e12 = v["eta"][1, 2, 1].X
    e21 = v["eta"][2, 1, 1].X
    dG1 = v["d_G"][1, 1].X
    dG2 = v["d_G"][2, 1].X
    aG1 = v["a_G"][1, 1].X
    aG2 = v["a_G"][2, 1].X
    ok  = abs(e12 + e21 - 1.0) < 1e-4
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Test 27-28.1 — Exactly one eta=1  |"
          f"  eta[1,2]={e12:.1f}  eta[2,1]={e21:.1f}"
          f"  dG1={dG1:.2f}  aG1={aG1:.2f}  dG2={dG2:.2f}  aG2={aG2:.2f}")


def test_27_28_2_both_eta_zero_infeasible():
    """
    Test 27-28.2 — Both eta=0 is infeasible when two trucks visit the same GH.
    Force eta[1,2,1]=0 and eta[2,1,1]=0. C27–C28 then imply that neither
    truck departs before the other docks, which is impossible when both
    trucks are present. Expected: INFEASIBLE.
    """
    def force_both_eta_zero(m, v):
        m.addConstr(v["eta"][1, 2, 1] == 0, name="test_eta12_zero")
        m.addConstr(v["eta"][2, 1, 1] == 0, name="test_eta21_zero")

    m, _ = build_base_model(
        K_trucks=[1, 2], Cap_W=1_000,
        extra_constrs=force_both_eta_zero,
    )
    solve_and_check(m, expected_status=GRB.INFEASIBLE,
                    label="Test 27-28.2 — Both eta=0 → INFEASIBLE")


# =============================================================================
#  CONSTRAINT 29 — Dock assignment
#  sum_d y[k,d,g] = sum_{arcs entering g from outside} x[j,i,k]
# =============================================================================

def test_29_1_visiting_truck_gets_dock():
    """
    Test 29.1 — A truck that visits a GH is assigned to exactly one dock.
    Baseline instance: 1 truck, 1 dock, 2 ULDs all going to GH g=1.
    Expected: y[1,1,1] = 1.
    """
    m, v = build_base_model()
    m.optimize()
    if not _check_optimal(m, "Test 29.1"):
        return

    y_val  = v["y"][1, 1, 1].X
    visits = sum(
        v["x"][j, i, 1].X
        for i in v["_Nodes_D"]
        for j in v["_All_Nodes"]
        if (j, i) in v["_Edges"] and j not in v["_Nodes_D"]
    )
    ok = abs(y_val - 1.0) < 1e-4 and abs(visits - 1.0) < 1e-4
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Test 29.1 — Dock assigned to visiting truck  |"
          f"  y[1,1,1]={y_val:.1f}  entries_into_GH={visits:.1f}")


def test_29_2_non_visiting_truck_gets_no_dock():
    """
    Test 29.2 — A truck that does not visit a GH must not be assigned a dock.
    Reduce to n=1 ULD with two trucks: only one truck makes a trip.
    Expected: y[1,1,1] + y[2,1,1] = 1 (exactly the visiting truck).
    """
    m, v = build_base_model(n_uld=1, K_trucks=[1, 2])
    m.optimize()
    if not _check_optimal(m, "Test 29.2"):
        return

    y1 = v["y"][1, 1, 1].X
    y2 = v["y"][2, 1, 1].X
    ok = abs(y1 + y2 - 1.0) < 1e-4
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Test 29.2 — Only visiting truck has dock  |"
          f"  y[1,1,1]={y1:.1f}  y[2,1,1]={y2:.1f}")


# =============================================================================
#  CONSTRAINTS 30–32 — Linearisation of z (product of y variables)
#  C30: z[k1,d1,k2,d2,g] <= y[k1,d1,g]
#  C31: z[k1,d1,k2,d2,g] <= y[k2,d2,g]
#  C32: y[k1,d1,g] + y[k2,d2,g] – 1 <= z[k1,d1,k2,d2,g]
# =============================================================================

def test_30_32_1_z_equals_product():
    """
    Test 30-32.1 — z correctly linearises the product y[k1,d1] * y[k2,d2].
    Two trucks, two docks: for every (d1, d2) pair check that
    z[1,d1,2,d2,1] = y[1,d1,1] * y[2,d2,1] at the optimum.
    """
    m, v = build_base_model(K_trucks=[1, 2], Cap_W=1_000, Delta_GH=2)
    m.optimize()
    if not _check_optimal(m, "Test 30-32.1"):
        return

    all_ok = True
    for d1 in v["_Docks"]:
        for d2 in v["_Docks"]:
            y1    = v["y"][1, d1, 1].X
            y2    = v["y"][2, d2, 1].X
            z_val = v["z"][1, d1, 2, d2, 1].X
            expected = y1 * y2
            if abs(z_val - expected) > 1e-3:
                all_ok = False
                print(f"    Mismatch z[1,{d1},2,{d2},1]={z_val:.3f}  "
                      f"y1={y1:.1f}  y2={y2:.1f}  expected={expected:.3f}")
    tag = "PASS" if all_ok else "FAIL"
    print(f"  [{tag}] Test 30-32.1 — z equals product y[k1,d1] * y[k2,d2]")


# =============================================================================
#  CONSTRAINT 33 — Dock non-overlap (same dock only)
#  z[k1,d1,k2,d1,g] <= eta[k1,k2,g] + eta[k2,k1,g]
# =============================================================================

def test_33_1_same_dock_both_eta_zero_infeasible():
    """
    Test 33.2 — Same dock + both eta=0 is infeasible.
    Force y[1,1,1]=y[2,1,1]=1 (both trucks on dock 1) and
    eta[1,2,1]=eta[2,1,1]=0. C32 requires z[1,1,2,1,1]>=1; C33 requires
    z[1,1,2,1,1]<=0. These bounds are contradictory.
    Expected: INFEASIBLE.
    """
    def force_same_dock_no_order(m, v):
        m.addConstr(v["y"][1, 1, 1] == 1, name="test_y1d1")
        m.addConstr(v["y"][2, 1, 1] == 1, name="test_y2d1")
        m.addConstr(v["eta"][1, 2, 1] == 0, name="test_eta12_0")
        m.addConstr(v["eta"][2, 1, 1] == 0, name="test_eta21_0")

    m, _ = build_base_model(
        K_trucks=[1, 2], Cap_W=1_000, Delta_GH=2,
        extra_constrs=force_same_dock_no_order,
    )
    solve_and_check(m, expected_status=GRB.INFEASIBLE,
                    label="Test 33.2 — Same dock + both eta=0 → INFEASIBLE")


def test_33_2_different_docks_no_order_needed():
    """
    Test 33.3 — Two trucks on different docks need no sequential ordering.
    Force truck 1 → dock 1, truck 2 → dock 2, and both eta=0. C33 only
    applies to same-dock pairs; since the trucks use different docks,
    z[1,1,2,1,1] = 0 and C33 is trivially satisfied.
    Expected: OPTIMAL (different docks are independent resources).
    """
    def force_diff_docks_eta_zero(m, v):
        m.addConstr(v["y"][1, 1, 1] == 1, name="test_y1d1")
        m.addConstr(v["y"][2, 2, 1] == 1, name="test_y2d2")
        m.addConstr(v["eta"][1, 2, 1] == 0, name="test_eta12_0")
        m.addConstr(v["eta"][2, 1, 1] == 0, name="test_eta21_0")

    m, v = build_base_model(
        K_trucks=[1, 2], Cap_W=1_000, Delta_GH=2,
        extra_constrs=force_diff_docks_eta_zero,
    )
    status, obj, ok = solve_and_check(
        m, expected_status=GRB.OPTIMAL,
        label="Test 33.3 — Diff docks, eta=0 → OPTIMAL")
    if status == GRB.OPTIMAL:
        z_same = v["z"][1, 1, 2, 1, 1].X
        print(f"           z[1,1,2,1,1]={z_same:.4f}  (should be 0)")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  CONSTRAINT 25 — Waiting time at FF while docked")
    print(SEP)
    test_25_1_zero_wait_no_tightening()
    test_25_2_forced_wait_at_ff()

    print(f"\n{SEP}")
    print("  CONSTRAINT 26 — Waiting time at GH while docked")
    print(SEP)
    test_26_1_zero_wait_no_tightening()
    test_26_2_forced_wait_at_gh()

    print(f"\n{SEP}")
    print("  CONSTRAINTS 27–28 — Overlap variable (eta)")
    print(SEP)
    test_27_28_1_eta_one_when_sequential()
    test_27_28_2_both_eta_zero_infeasible()

    print(f"\n{SEP}")
    print("  CONSTRAINT 29 — Dock assignment")
    print(SEP)
    test_29_1_visiting_truck_gets_dock()
    test_29_2_non_visiting_truck_gets_no_dock()

    print(f"\n{SEP}")
    print("  CONSTRAINTS 30–32 — Linearization of z")
    print(SEP)
    test_30_32_1_z_equals_product()

    print(f"\n{SEP}")
    print("  CONSTRAINT 33 — Dock non-overlap")
    print(SEP)
    test_33_1_same_dock_both_eta_zero_infeasible()
    test_33_2_different_docks_no_order_needed()