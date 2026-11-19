import copy
import json
import math
from decimal import Decimal, ROUND_HALF_UP
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ROB Studio | Career Venture Lab (Stage 1)",
    layout="wide",
)


def money(val) -> Decimal:
    return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def initial_state(
    team_id: str = "TEAM-ROB-01",
    selected_course: str = "Precalculus",
    selected_level: str = "Foundation",
) -> dict:
    pathway_id = "ROB-2" if selected_course == "Precalculus" else "ROB-3"
    return {
        "team_id": team_id,
        "course": selected_course,
        "pathway_id": pathway_id,
        "selected_level": selected_level,
        "teacher_approved_level": "Foundation",
        "snapshot_id": "SNAP-000",
        "state_version": 1,
        "simulation_date": "2027-01-01",
        "base_period": "monthly",
        "working_days": 20,
        "currency": "USD",
        "startup": {
            "owner_contribution": "12000.00",
            "equipment_assets": "4000.00",
            "pre_opening_expense": "0.00",
            "opening_inventory": "3000.00",
            "post_startup_cash": "5000.00",
            "retained_earnings": "0.00",
        },
        "cash": "5000.00",
        "stock_units": 100,
        "inventory_value": "3000.00",
        "debt": "0.00",
        "ledger": [],
        "decisions": [],
        "campaigns": [],
    }


def compute_rob_step(
    state: dict,
    inputs: dict,
    mode: str = "normal",
    capacity_override: int | None = None,
) -> tuple[dict, str]:
    """Deterministic ROB monthly algorithm using Decimal money and integer units."""
    run_id = str(inputs["run_id"]).strip()
    if not run_id:
        return state, "ERROR: run_id is required."

    if any(row["run_id"] == run_id for row in state["ledger"]):
        return state, f"DUPLICATE IGNORED: run_id '{run_id}' already exists in ledger."

    price = money(inputs["price"])
    requested = int(inputs["requested"])
    staff_hours = float(inputs["staff_hours"])
    hours_per_unit = float(inputs["hours_per_unit"])
    restock_units = int(inputs["restock_units"])
    defect_rate = float(inputs["defect_rate"])

    if price <= 0 or requested < 0 or staff_hours < 0 or hours_per_unit <= 0:
        return state, "ERROR: Invalid numeric bounds on price, demand, or hours."
    if restock_units < 0 or not (0.0 <= defect_rate <= 1.0):
        return state, "ERROR: restock_units must be >= 0 and defect_rate in [0, 1]."

    comp_unit_cost = money(inputs["components_COGS"])
    noncomp_unit_cost = (
        money(inputs["assembly_labor"])
        + money(inputs["packaging"])
        + money(inputs["warranty_consumables"])
    )
    var_per_unit = money(comp_unit_cost + noncomp_unit_cost)

    fixed_cost = (
        money(inputs["workspace"])
        + money(inputs["software"])
        + money(inputs["equipment_maintenance"])
        + money(inputs["admin"])
    )
    ad_cost = money(inputs["ad_cost"])
    interest = money(inputs.get("interest", 0))
    principal = money(inputs.get("loan_principal", 0))
    capex = money(inputs.get("additional_capex", 0))
    refund_amount = money(inputs.get("refund_amount", 0))

    if mode == "fixture" and capacity_override is not None:
        capacity = int(capacity_override)
    else:
        capacity = math.floor(staff_hours / hours_per_unit)

    available_stock = int(state["stock_units"]) + restock_units
    attempted = min(requested, capacity, available_stock)
    failed = math.floor(attempted * defect_rate)
    sold = attempted - failed

    revenue = money(price * sold) - refund_amount
    consumed_comp_cost = money(comp_unit_cost * attempted)
    noncomp_var_cash = money(noncomp_unit_cost * attempted)
    variable_expense = money(var_per_unit * attempted)
    total_operating_expense = money(variable_expense + fixed_cost + ad_cost)
    operating_profit = money(revenue - total_operating_expense)
    net_profit = money(operating_profit - interest)

    # Cash outflow for components happens on restock; expense happens when consumed
    restock_cash_outflow = money(comp_unit_cost * restock_units)
    ending_cash = money(
        Decimal(state["cash"])
        + revenue
        - restock_cash_outflow
        - noncomp_var_cash
        - fixed_cost
        - ad_cost
        - interest
        - principal
        - capex
    )
    ending_stock = available_stock - attempted
    ending_inv_value = money(comp_unit_cost * ending_stock)

    next_state = copy.deepcopy(state)
    new_version = int(state["state_version"]) + 1
    entry = {
        "run_id": run_id,
        "date": inputs["simulation_date"],
        "mode": mode,
        "requested": requested,
        "capacity": capacity,
        "attempted": attempted,
        "failed": failed,
        "sold": sold,
        "unserved_demand": max(0, requested - attempted),
        "revenue": str(revenue),
        "consumed_comp_cost": str(consumed_comp_cost),
        "variable_expense": str(variable_expense),
        "fixed_expense": str(fixed_cost),
        "ad_cost": str(ad_cost),
        "total_expense": str(total_operating_expense),
        "operating_profit": str(operating_profit),
        "net_profit": str(net_profit),
        "ending_cash": str(ending_cash),
        "ending_stock_units": ending_stock,
        "ending_inventory_value": str(ending_inv_value),
        "state_version": new_version,
    }
    next_state["ledger"].append(entry)
    next_state["cash"] = str(ending_cash)
    next_state["stock_units"] = ending_stock
    next_state["inventory_value"] = str(ending_inv_value)
    next_state["state_version"] = new_version
    next_state["snapshot_id"] = f"SNAP-{new_version:03d}"
    next_state["simulation_date"] = inputs["simulation_date"]
    return next_state, f"APPROVED & RECORDED: {run_id} (state_version v{new_version})"


def validate_and_load_snapshot(raw_json: str, expected_team_id: str, current_version: int) -> tuple[dict | None, str]:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return None, f"REJECTED: Invalid JSON format ({exc})."

    required_keys = {
        "team_id", "course", "pathway_id", "snapshot_id", "state_version",
        "cash", "stock_units", "inventory_value", "ledger"
    }
    if not required_keys.issubset(data.keys()):
        missing = sorted(required_keys - set(data.keys()))
        return None, f"REJECTED: Missing required snapshot fields: {missing}"

    if data["team_id"] != expected_team_id:
        return None, f"REJECTED: Team ID mismatch (expected '{expected_team_id}', got '{data['team_id']}')."

    if int(data["state_version"]) < current_version:
        return None, (
            f"REJECTED (STALE VERSION): Imported v{data['state_version']} is older than "
            f"current v{current_version}."
        )

    return data, f"RESTORED: Loaded {data['snapshot_id']} (v{data['state_version']})."


if "sim_state" not in st.session_state:
    st.session_state.sim_state = initial_state()

state = st.session_state.sim_state

# --- SIDEBAR: COURSE, LEVEL & PERSISTENCE CONTROLS ---
st.sidebar.header("Venture & Course Controls")
team_id_input = st.sidebar.text_input("Team ID", value=state["team_id"])
course_choice = st.sidebar.selectbox(
    "Selected Course (Mixed Class Switch)",
    ["Precalculus", "AP Calculus"],
    index=0 if state["course"] == "Precalculus" else 1,
)
level_choice = st.sidebar.selectbox(
    "Challenge Level",
    ["Foundation", "Applied", "Advanced"],
    index=["Foundation", "Applied", "Advanced"].index(state["selected_level"]),
)
teacher_approved = st.sidebar.checkbox(
    "Teacher Approved Level Change",
    value=(level_choice == "Foundation"),
    help="Foundation is pre-approved. Applied/Advanced require teacher approval.",
)

state["team_id"] = team_id_input
state["course"] = course_choice
state["pathway_id"] = "ROB-2" if course_choice == "Precalculus" else "ROB-3"
if teacher_approved or level_choice == "Foundation":
    state["selected_level"] = level_choice
else:
    st.sidebar.warning("Level locked at Foundation until teacher approval is checked.")
    state["selected_level"] = "Foundation"

st.sidebar.markdown("---")
st.sidebar.subheader("Save / Restore JSON Snapshot")
st.sidebar.download_button(
    label="Export Snapshot JSON",
    data=json.dumps(state, indent=2),
    file_name=f"{state['team_id']}_{state['snapshot_id']}_v{state['state_version']}.json",
    mime="application/json",
)
uploaded_file = st.sidebar.file_uploader("Import Snapshot JSON", type=["json"])
if uploaded_file is not None:
    loaded, status_msg = validate_and_load_snapshot(
        uploaded_file.read().decode("utf-8"),
        expected_team_id=state["team_id"],
        current_version=int(state["state_version"]),
    )
    if loaded is not None:
        st.session_state.sim_state = loaded
        st.sidebar.success(status_msg)
        st.rerun()
    else:
        st.sidebar.error(status_msg)

if st.sidebar.button("Reset to Opening Baseline"):
    st.session_state.sim_state = initial_state(
        team_id=team_id_input,
        selected_course=course_choice,
        selected_level=state["selected_level"],
    )
    st.rerun()

# --- HEADER ---
st.title("ROB Studio: Robotics Kit & Repair Simulation (Stage 1)")
st.caption(
    f"Team: {state['team_id']} | Course: {state['course']} ({state['pathway_id']}) | "
    f"Level: {state['selected_level']} | Snapshot: {state['snapshot_id']} (v{state['state_version']}) | "
    "Fictional Teaching Simulation Only"
)

# --- TOP KPI METRICS ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Spendable Cash", f"${Decimal(state['cash']):,.2f}")
k2.metric("Component Stock", f"{state['stock_units']} sets")
k3.metric("Inventory Value", f"${Decimal(state['inventory_value']):,.2f}")
k4.metric("Ledger State Version", f"v{state['state_version']}")

tab_overview, tab_inputs, tab_math, tab_tests = st.tabs(
    ["1. Overview & Ledger", "2. Inputs & Simulation Step", "3. Math Skills & Levels", "4. Fixture & Acceptance Tests"]
)

# --- TAB 1: OVERVIEW & LEDGER ---
with tab_overview:
    st.subheader("Startup Financing & Opening Balance Sheet (Fictional Defaults)")
    st.dataframe(
        pd.DataFrame(
            [
                {"Item": "Owner Contribution (Equity)", "Amount (USD)": "12,000.00", "Accounting Treatment": "Equity inflow; not sales revenue"},
                {"Item": "Equipment Assets (Tools & Test Rig)", "Amount (USD)": "4,000.00", "Accounting Treatment": "Capital asset; not expensed again"},
                {"Item": "Opening Inventory (100 sets @ $30)", "Amount (USD)": "3,000.00", "Accounting Treatment": "Asset; expensed as COGS when consumed"},
                {"Item": "Post-Startup Spendable Cash", "Amount (USD)": "5,000.00", "Accounting Treatment": "Opening cash balance"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Dated Event Ledger (Actual Simulated Transactions)")
    if state["ledger"]:
        ledger_df = pd.DataFrame(state["ledger"])
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Ledger CSV",
            data=ledger_df.to_csv(index=False),
            file_name=f"{state['team_id']}_ledger_v{state['state_version']}.csv",
            mime="text/csv",
        )
    else:
        st.info("No simulation steps recorded yet. Use Tab 2 to run a month or Tab 4 to replay monthly_fixture.csv.")

# --- TAB 2: INPUTS & SIMULATION STEP ---
with tab_inputs:
    st.subheader("Monthly Operating Inputs & Cost Lines")
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_date = st.text_input("Simulation Date (YYYY-MM-DD)", value="2027-01-31")
        run_id = st.text_input("Unique Run ID", value=f"RUN-{state['state_version']:03d}")
        price = st.number_input("Kit Price ($/kit)", min_value=1.0, value=120.0, step=5.0)
        requested = st.number_input("Requested Demand (kits)", min_value=0, value=40, step=1)
    with c2:
        staff_hours = st.number_input("Staff Hours / Month", min_value=0.0, value=90.0, step=3.0)
        hours_per_unit = st.number_input("Hours per Kit", min_value=0.5, value=3.0, step=0.5)
        restock_units = st.number_input("Restock Component Sets Purchased", min_value=0, value=0, step=5)
        defect_rate = st.slider("Defect Rate (0.00 to 1.00)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    with c3:
        comp_cost = st.number_input("Components COGS ($/attempted)", min_value=0.0, value=30.0, step=1.0)
        labor_cost = st.number_input("Assembly Labor ($/attempted)", min_value=0.0, value=20.0, step=1.0)
        pack_cost = st.number_input("Packaging ($/attempted)", min_value=0.0, value=10.0, step=1.0)
        warr_cost = st.number_input("Warranty Consumables ($/attempted)", min_value=0.0, value=5.0, step=1.0)

    st.markdown("**Monthly Fixed Expenses ($600 baseline) & Marketing**")
    f1, f2, f3, f4, f5 = st.columns(5)
    workspace = f1.number_input("Workspace ($)", min_value=0.0, value=250.0, step=10.0)
    software = f2.number_input("Software ($)", min_value=0.0, value=100.0, step=10.0)
    maint = f3.number_input("Equip. Maint. ($)", min_value=0.0, value=150.0, step=10.0)
    admin = f4.number_input("Admin ($)", min_value=0.0, value=100.0, step=10.0)
    ad_cost = f5.number_input("Own Marketing ($)", min_value=0.0, value=0.0, step=25.0)

    derived_cap = math.floor(staff_hours / hours_per_unit)
    var_unit_sum = comp_cost + labor_cost + pack_cost + warr_cost
    fixed_sum = workspace + software + maint + admin
    st.info(
        f"Preview (Uncommitted): Derived Capacity = {derived_cap} kits | "
        f"Variable Cost = ${var_unit_sum:.2f}/attempted kit | Fixed Cost = ${fixed_sum:.2f}/month"
    )

    if st.button("Approve & Advance Simulation One Month", type="primary"):
        step_inputs = {
            "simulation_date": sim_date,
            "run_id": run_id,
            "price": price,
            "requested": requested,
            "staff_hours": staff_hours,
            "hours_per_unit": hours_per_unit,
            "restock_units": restock_units,
            "defect_rate": defect_rate,
            "components_COGS": comp_cost,
            "assembly_labor": labor_cost,
            "packaging": pack_cost,
            "warranty_consumables": warr_cost,
            "workspace": workspace,
            "software": software,
            "equipment_maintenance": maint,
            "admin": admin,
            "ad_cost": ad_cost,
        }
        updated_state, msg = compute_rob_step(state, step_inputs, mode="normal")
        if msg.startswith("APPROVED"):
            st.session_state.sim_state = updated_state
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

# --- TAB 3: MATH SKILLS & CHALLENGE LEVEL PANEL ---
with tab_math:
    st.subheader(f"Persistent Math Panel: {state['course']} ({state['pathway_id']}) — {state['selected_level']} Level")
    F_val = float(fixed_sum)
    v_val = float(var_unit_sum)
    p0_val = float(price)
    D0_val = 40.0
    cap_val = int(derived_cap)

    if state["course"] == "Precalculus":
        if state["selected_level"] == "Foundation":
            st.markdown(
                f"""
                - **Business Question:** How does the average cost per robotics kit change as monthly production $s$ increases up to capacity?
                - **Equations:** Total Cost $C(s) = {F_val:.0f} + {v_val:.0f}s$ (dollars); Average Cost $A(s) = \\frac{{C(s)}}{{s}} = \\frac{{{F_val:.0f}}}{{s}} + {v_val:.0f}$ (dollars/kit)
                - **Practical Domain:** $0 < s \\le {cap_val}$ kits/month ($s = 0$ is excluded because division by zero is undefined)
                - **Asymptotes:** Vertical asymptote at $s = 0$ ($A(0)$ undefined); horizontal asymptote at $y = {v_val:.2f}$ ($/kit)
                - **Baseline Check:** At $s = {cap_val}$, $C({cap_val}) = \\${F_val + v_val * cap_val:,.2f}$ and $A({cap_val}) = \\${(F_val / cap_val + v_val):,.2f}$/kit
                """
            )
            s_vals = np.arange(1, max(2, cap_val + 1))
            math_df = pd.DataFrame(
                {
                    "Completed Kits (s)": s_vals,
                    "Total Cost C(s) ($)": F_val + v_val * s_vals,
                    "Average Cost A(s) ($/kit)": np.round(F_val / s_vals + v_val, 2),
                    "Horizontal Asymptote ($/kit)": v_val,
                }
            )
            base_chart = alt.Chart(math_df).encode(x=alt.X("Completed Kits (s):Q", title="Completed Kits s (0 < s <= capacity)"))
            line_a = base_chart.mark_line(point=True, strokeWidth=2.5).encode(
                y=alt.Y("Average Cost A(s) ($/kit):Q", title="Average Cost ($/kit)"),
                tooltip=["Completed Kits (s)", "Total Cost C(s) ($)", "Average Cost A(s) ($/kit)"],
            )
            asymp = alt.Chart(pd.DataFrame({"y": [v_val]})).mark_rule(strokeDash=[6, 4]).encode(y="y:Q")
            st.altair_chart((line_a + asymp).properties(height=320), use_container_width=True)
            st.dataframe(math_df, use_container_width=True, hide_index=True)
            st.warning(
                "Independent Transfer Check: Explain in your own words why A(s) approaches $65/kit as s grows, "
                "and why s = 0 cannot be evaluated."
            )
        elif state["selected_level"] == "Applied":
            breakpoint_p = p0_val * (2.0 - cap_val / D0_val)
            st.markdown(
                f"""
                - **Applied Composition Model:** $q(p) = \\max(0, 80 - \\frac{{1}}{{3}}p)$, $s(p) = \\min({cap_val}, q(p))$, $R(p) = p \\cdot s(p)$, $C(s(p)) = {F_val:.0f} + {v_val:.0f}s(p)$
                - **Price Domain:** $0 \\le p \\le 240$ ($/kit); **Capacity Breakpoint:** $p = \\${breakpoint_p:.2f}$
                """
            )
            p_grid = np.linspace(0, 240, 49)
            q_grid = np.maximum(0, 80.0 - (1.0 / 3.0) * p_grid)
            s_grid = np.minimum(cap_val, q_grid)
            prof_grid = p_grid * s_grid - (F_val + v_val * s_grid)
            comp_df = pd.DataFrame({"Price ($)": p_grid, "Demand q(p)": np.round(q_grid, 2), "Realized s(p)": np.round(s_grid, 2), "Continuous Profit ($)": np.round(prof_grid, 2)})
            st.line_chart(comp_df.set_index("Price ($)")[["Continuous Profit ($)"]])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            months = np.arange(1, 13)
            d_seas = D0_val * (1.0 + 0.20 * np.sin(2.0 * np.pi * (months - 1) / 12.0))
            s_seas = np.minimum(cap_val, np.floor(d_seas))
            seas_df = pd.DataFrame({"Month (t)": months, "Assumed Seasonal Demand D(t)": np.round(d_seas, 2), "Clipped Sales": s_seas})
            st.markdown("**Advanced Seasonal Extension:** $D(t) = D_0(1 + 0.20\\sin(2\\pi(t-1)/12))$ clipped at capacity (labeled assumed seasonality).")
            st.line_chart(seas_df.set_index("Month (t)"))
            st.dataframe(seas_df, use_container_width=True, hide_index=True)

    else:  # AP Calculus (ROB-3)
        a_val = 2.0 * D0_val  # 80
        b_val = D0_val / p0_val  # 1/3
        crit_p = (a_val + b_val * v_val) / (2.0 * b_val)  # 152.50
        crit_profit = (crit_p - v_val) * (a_val - b_val * crit_p) - F_val
        if state["selected_level"] == "Foundation":
            st.markdown(
                f"""
                - **Business Question:** How does unconstrained monthly profit change with kit price $p$ on the positive-demand interval?
                - **Equations:** Demand $q(p) = {a_val:.0f} - {b_val:.4f}p$; Unconstrained Profit $P(p) = (p - {v_val:.0f})({a_val:.0f} - {b_val:.4f}p) - {F_val:.0f}$
                - **Derivative:** $P'(p) = {a_val + b_val * v_val:.2f} - {2 * b_val:.4f}p$ (units: **dollars of monthly profit per dollar increase in price**)
                - **Domain:** $0 \\le p \\le {2 * p0_val:.0f}$ ($/kit)
                - **Critical Point & Endpoints:** $P'(p) = 0 \\implies p^* = \\${crit_p:.2f}$ ($P(\\$152.50) = \\${crit_profit:,.2f}$); Endpoints: $P(0) = -\\$5,800.00$, $P(240) = -\\$600.00$
                """
            )
            p_vals = np.linspace(0, 240, 49)
            p_prof = (p_vals - v_val) * (a_val - b_val * p_vals) - F_val
            p_deriv = (a_val + b_val * v_val) - 2.0 * b_val * p_vals
            calc_df = pd.DataFrame(
                {
                    "Price p ($)": p_vals,
                    "Unconstrained Profit P(p) ($)": np.round(p_prof, 2),
                    "Derivative P'(p) ($ profit / $ price)": np.round(p_deriv, 2),
                }
            )
            chart_p = alt.Chart(calc_df).mark_line(strokeWidth=2.5).encode(
                x=alt.X("Price p ($):Q", title="Kit Price p ($)"),
                y=alt.Y("Unconstrained Profit P(p) ($):Q", title="Profit ($)"),
                tooltip=["Price p ($)", "Unconstrained Profit P(p) ($)", "Derivative P'(p) ($ profit / $ price)"],
            )
            zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
            st.altair_chart((chart_p + zero_line).properties(height=320), use_container_width=True)
            st.dataframe(calc_df, use_container_width=True, hide_index=True)
            st.warning(
                "Independent Transfer Check: Verify P'(152.50) = 0 by hand and explain what the sign of P'(120) "
                "tells the business about raising price from $120."
            )
        elif state["selected_level"] == "Applied":
            corner_p = (a_val - cap_val) / b_val
            st.markdown(
                f"""
                - **Applied Constrained Optimization:** $s(p) = \\min({cap_val}, \\max(0, {a_val:.0f} - {b_val:.4f}p))$
                - **Capacity Corner:** $p = \\frac{{{a_val:.0f} - {cap_val}}}{{{b_val:.4f}}} = \\${corner_p:.2f}$ ($P'$ undefined at genuine corner)
                """
            )
            p_vals = np.linspace(0, 240, 49)
            s_vals = np.minimum(cap_val, np.maximum(0, a_val - b_val * p_vals))
            prof_c = (p_vals - v_val) * s_vals - F_val
            app_df = pd.DataFrame({"Price p ($)": p_vals, "Constrained Sales s(p)": np.round(s_vals, 2), "Constrained Profit ($)": np.round(prof_c, 2)})
            st.line_chart(app_df.set_index("Price p ($)")[["Constrained Profit ($)"]])
            st.dataframe(app_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("**Advanced Integration Exercise:** $r(t) = 100 + 20t$ and $c(t) = 60 + 5t$ ($/day) over $0 \\le t \\le 10$ days gives accumulated profit = **$1,150.00**.")

# --- TAB 4: FIXTURE & ACCEPTANCE TESTS ---
with tab_tests:
    st.subheader("ROB Sample Fixture Replay (Fixture Mode)")
    st.warning(
        "Fixture Mode Warning: Replaying monthly_fixture.csv uses explicit capacity_override values "
        "to verify ledger arithmetic. Do not mix Fixture Mode and Normal Mode in an active run."
    )
    if st.button("Reset & Replay 3-Month Fixture (sample_data/monthly_fixture.csv)"):
        test_state = initial_state(
            team_id=state["team_id"],
            selected_course=state["course"],
            selected_level=state["selected_level"],
        )
        fixture_rows = [
            ("2027-01-31", "FIX-2027-01", 40, 30, 0),
            ("2027-02-28", "FIX-2027-02", 44, 30, 100),
            ("2027-03-31", "FIX-2027-03", 36, 22, 0),
        ]
        for f_date, f_id, f_req, f_cap, f_ad in fixture_rows:
            f_in = {
                "simulation_date": f_date,
                "run_id": f_id,
                "price": 120,
                "requested": f_req,
                "staff_hours": 90,
                "hours_per_unit": 3,
                "restock_units": 0,
                "defect_rate": 0,
                "components_COGS": 30,
                "assembly_labor": 20,
                "packaging": 10,
                "warranty_consumables": 5,
                "workspace": 250,
                "software": 100,
                "equipment_maintenance": 150,
                "admin": 100,
                "ad_cost": f_ad,
            }
            test_state, _ = compute_rob_step(test_state, f_in, mode="fixture", capacity_override=f_cap)
        st.session_state.sim_state = test_state
        st.success("Replayed 3 fixture rows. Inspect Overview & Ledger tab to verify totals.")
        st.rerun()