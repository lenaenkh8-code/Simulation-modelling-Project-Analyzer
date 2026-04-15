import io
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Project Duration Analyzer",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
div[data-testid="stMetric"] {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 12px;
    background-color: white;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    background-color: white;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
}
</style>
""", unsafe_allow_html=True)

UNIT_OPTIONS = ["days", "weeks", "hours", "months", "minutes"]

UNIT_TO_MINUTES = {
    "minutes": 1,
    "hours": 60,
    "days": 60 * 24,
    "weeks": 60 * 24 * 7,
    "months": 60 * 24 * 30,
}

DEFAULT_DATA = pd.DataFrame([
    {"Activity": "Design", "Label": "A", "Immediate predecessors": "-", "Minimum duration": 16, "Average duration": 21, "Maximum duration": 26, "Unit of measure": "days", "Owner": "Engineering", "Phase": "Planning"},
    {"Activity": "Build prototype", "Label": "B", "Immediate predecessors": "A", "Minimum duration": 3, "Average duration": 6, "Maximum duration": 9, "Unit of measure": "days", "Owner": "Engineering", "Phase": "Build"},
    {"Activity": "Evaluate equipment", "Label": "C", "Immediate predecessors": "A", "Minimum duration": 5, "Average duration": 7, "Maximum duration": 9, "Unit of measure": "days", "Owner": "Operations", "Phase": "Testing"},
    {"Activity": "Test prototype", "Label": "D", "Immediate predecessors": "B", "Minimum duration": 2, "Average duration": 3, "Maximum duration": 4, "Unit of measure": "days", "Owner": "QA", "Phase": "Testing"},
    {"Activity": "Write equipment report", "Label": "E", "Immediate predecessors": "C,D", "Minimum duration": 4, "Average duration": 6, "Maximum duration": 8, "Unit of measure": "days", "Owner": "Operations", "Phase": "Reporting"},
    {"Activity": "Write methods report", "Label": "F", "Immediate predecessors": "C,D", "Minimum duration": 6, "Average duration": 8, "Maximum duration": 10, "Unit of measure": "days", "Owner": "QA", "Phase": "Reporting"},
    {"Activity": "Write final report", "Label": "G", "Immediate predecessors": "E,F", "Minimum duration": 1, "Average duration": 2, "Maximum duration": 3, "Unit of measure": "days", "Owner": "PMO", "Phase": "Close"},
])

CSV_TEMPLATE = """Activity,Label,Immediate predecessors,Minimum duration,Average duration,Maximum duration,Unit of measure,Owner,Phase
Design,A,-,16,21,26,days,Engineering,Planning
Build prototype,B,A,3,6,9,days,Engineering,Build
Evaluate equipment,C,A,5,7,9,days,Operations,Testing
Test prototype,D,B,2,3,4,days,QA,Testing
Write equipment report,E,"C,D",4,6,8,days,Operations,Reporting
Write methods report,F,"C,D",6,8,10,days,QA,Reporting
Write final report,G,"E,F",1,2,3,days,PMO,Close
"""

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Predecessors": "Immediate predecessors",
        "Immediate predecessor": "Immediate predecessors",
        "Minimum": "Minimum duration",
        "Average": "Average duration",
        "MostLikely": "Average duration",
        "Maximum": "Maximum duration",
        "Unit": "Unit of measure",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    required = [
        "Activity", "Label", "Immediate predecessors", "Minimum duration",
        "Average duration", "Maximum duration", "Unit of measure"
    ]
    optional = ["Owner", "Phase"]

    for col in required:
        if col not in df.columns:
            df[col] = "" if col in ["Activity", "Label", "Immediate predecessors", "Unit of measure"] else np.nan
    for col in optional:
        if col not in df.columns:
            df[col] = ""

    df = df[required + optional].copy()
    for col in ["Activity", "Label", "Immediate predecessors", "Unit of measure", "Owner", "Phase"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["Label"] = df["Label"].str.upper()
    df["Immediate predecessors"] = df["Immediate predecessors"].replace("", "-")
    df["Unit of measure"] = df["Unit of measure"].str.lower().replace("", "days")

    for col in ["Minimum duration", "Average duration", "Maximum duration"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def parse_predecessors(text: str):
    text = str(text).strip()
    if text in {"", "-", "None", "none", "nan"}:
        return []
    return [x.strip().upper() for x in text.split(",") if x.strip()]

def validate_df(df: pd.DataFrame):
    errors = []
    active = df[(df["Activity"] != "") | (df["Label"] != "")].copy()
    if active.empty:
        return ["Please enter at least one activity."]
    if active["Activity"].eq("").any():
        errors.append("Every row must have an Activity name.")
    if active["Label"].eq("").any():
        errors.append("Every row must have a Label.")
    if active["Label"].duplicated().any():
        dupes = active.loc[active["Label"].duplicated(), "Label"].unique().tolist()
        errors.append(f"Duplicate labels found: {', '.join(dupes)}")
    if active[["Minimum duration", "Average duration", "Maximum duration"]].isna().any().any():
        errors.append("Minimum duration, Average duration, and Maximum duration must all be numeric.")
    bad_order = active[(active["Minimum duration"] > active["Average duration"]) | (active["Average duration"] > active["Maximum duration"])]
    if not bad_order.empty:
        labels = ", ".join(bad_order["Label"].tolist())
        errors.append(f"Each row must satisfy Minimum ≤ Average ≤ Maximum. Check: {labels}")
    bad_units = active[~active["Unit of measure"].isin(UNIT_OPTIONS)]
    if not bad_units.empty:
        labels = ", ".join(bad_units["Label"].tolist())
        errors.append(f"Invalid unit of measure found. Check: {labels}")
    label_set = set(active["Label"].tolist())
    for _, row in active.iterrows():
        for pred in parse_predecessors(row["Immediate predecessors"]):
            if pred not in label_set:
                errors.append(f"Activity {row['Label']} references missing predecessor: {pred}")
    return errors

def add_standardized_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["factor_to_minutes"] = out["Unit of measure"].map(UNIT_TO_MINUTES)
    out["min_std"] = out["Minimum duration"] * out["factor_to_minutes"]
    out["avg_std"] = out["Average duration"] * out["factor_to_minutes"]
    out["max_std"] = out["Maximum duration"] * out["factor_to_minutes"]
    out["expected_std"] = (out["min_std"] + 4 * out["avg_std"] + out["max_std"]) / 6
    out["variance_std"] = ((out["max_std"] - out["min_std"]) / 6) ** 2
    out["stddev_std"] = np.sqrt(out["variance_std"])
    out["risk_range_std"] = out["max_std"] - out["min_std"]
    return out

def topological_order(df: pd.DataFrame):
    preds = {row["Label"]: parse_predecessors(row["Immediate predecessors"]) for _, row in df.iterrows()}
    succ = defaultdict(list)
    indeg = {label: 0 for label in preds}
    for label, pred_list in preds.items():
        indeg[label] = len(pred_list)
        for p in pred_list:
            succ[p].append(label)
    q = deque([node for node, deg in indeg.items() if deg == 0])
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for nxt in succ[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    if len(order) != len(preds):
        raise ValueError("Cycle detected in predecessor relationships.")
    return order, preds, succ

def compute_schedule(df_std: pd.DataFrame):
    order, preds, succ = topological_order(df_std)
    durations = dict(zip(df_std["Label"], df_std["expected_std"]))
    es, ef = {}, {}
    for node in order:
        es[node] = max([ef[p] for p in preds[node]], default=0.0)
        ef[node] = es[node] + durations[node]
    project_duration = max(ef.values()) if ef else 0.0
    lf, ls = {}, {}
    end_nodes = [n for n in order if len(succ[n]) == 0]
    for node in reversed(order):
        if node in end_nodes:
            lf[node] = project_duration
        else:
            lf[node] = min(ls[s] for s in succ[node])
        ls[node] = lf[node] - durations[node]
    slack = {n: ls[n] - es[n] for n in order}
    critical_nodes = [n for n in order if abs(slack[n]) < 1e-9]
    out = df_std.copy()
    out["ES_std"] = out["Label"].map(es)
    out["EF_std"] = out["Label"].map(ef)
    out["LS_std"] = out["Label"].map(ls)
    out["LF_std"] = out["Label"].map(lf)
    out["Slack_std"] = out["Label"].map(slack)
    out["Critical"] = out["Label"].isin(critical_nodes)
    return out, project_duration, critical_nodes

def simulate_project(df_std: pd.DataFrame, n_sims: int, random_seed: int = 42):
    rng = np.random.default_rng(random_seed)
    order, preds, _ = topological_order(df_std)
    records = df_std.set_index("Label")[["min_std", "avg_std", "max_std"]].to_dict("index")
    results = np.zeros(n_sims)
    critical_finish_counts = defaultdict(int)
    for i in range(n_sims):
        sampled = {label: rng.triangular(v["min_std"], v["avg_std"], v["max_std"]) for label, v in records.items()}
        ef = {}
        for node in order:
            start = max([ef[p] for p in preds[node]], default=0.0)
            ef[node] = start + sampled[node]
        finish = max(ef.values()) if ef else 0.0
        results[i] = finish
        for node in order:
            if abs(ef[node] - finish) < 1e-8:
                critical_finish_counts[node] += 1
    crit_df = pd.DataFrame({
        "Label": list(critical_finish_counts.keys()),
        "Critical finish frequency": [v / n_sims for v in critical_finish_counts.values()]
    }).sort_values("Critical finish frequency", ascending=False)
    return results, crit_df

def convert_from_minutes(value_in_minutes: float, display_unit: str) -> float:
    return value_in_minutes / UNIT_TO_MINUTES[display_unit]

def convert_series_from_minutes(series: pd.Series, display_unit: str) -> pd.Series:
    return series / UNIT_TO_MINUTES[display_unit]

def create_histogram(values, mean_val, percentile_val, service_level, display_unit):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=30)
    ax.axvline(mean_val, linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}")
    ax.axvline(percentile_val, linestyle=":", linewidth=2, label=f"P{service_level} = {percentile_val:.2f}")
    ax.set_xlabel(f"Project completion time ({display_unit})")
    ax.set_ylabel("Frequency")
    ax.set_title("B. Histogram of possible project durations")
    ax.legend()
    return fig

def create_gantt_chart(df_display, display_unit):
    plot_df = df_display.sort_values(["ES", "Label"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, max(4, len(plot_df) * 0.5)))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["Expected duration"], left=plot_df["ES"])
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Label"] + " - " + plot_df["Activity"])
    ax.invert_yaxis()
    ax.set_xlabel(f"Time ({display_unit})")
    ax.set_title("Expected project timeline")
    return fig

def create_risk_chart(df_display, display_unit):
    temp = df_display.copy()
    temp["Impact score"] = temp["Risk range"] * np.where(temp["Critical"], 1.5, 0.75)
    temp = temp.sort_values("Impact score", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(10, max(4, len(temp) * 0.45)))
    ax.barh(temp["Label"] + " - " + temp["Activity"], temp["Impact score"])
    ax.set_xlabel(f"Indicative risk score ({display_unit})")
    ax.set_title("Top schedule uncertainty drivers")
    return fig

def make_graphviz(df, critical_nodes):
    lines = ["digraph G {", 'rankdir=LR;', 'node [shape=box, style="rounded,filled"];']
    for _, row in df.iterrows():
        fill = "#f3d0d0" if row["Label"] in critical_nodes else "#e8eef5"
        safe_activity = str(row["Activity"]).replace('"', "'")
        lines.append(f'"{row["Label"]}" [label="{row["Label"]}: {safe_activity}", fillcolor="{fill}"];')
    for _, row in df.iterrows():
        for pred in parse_predecessors(row["Immediate predecessors"]):
            lines.append(f'"{pred}" -> "{row["Label"]}";')
    lines.append("}")
    return "\n".join(lines)

def generate_exec_summary(project_name, mean_val, service_level, deadline, p80, p90, p95, critical_nodes, display_unit):
    cp = " → ".join(critical_nodes) if critical_nodes else "Not identified"
    return f"""Project: {project_name}

Assignment-aligned answers
A. Mean duration of the project: {mean_val:.2f} {display_unit}
B. Histogram: generated in the dashboard to show the distribution of possible completion times
C. {service_level}% service-level completion time: {deadline:.2f} {display_unit}

Additional interpretation
- P80 = {p80:.2f} {display_unit}
- P90 = {p90:.2f} {display_unit}
- P95 = {p95:.2f} {display_unit}
- Expected critical path: {cp}

Managerial takeaway
The mean gives the central estimate, while the P{service_level} value is more appropriate for planning if you want a stronger confidence buffer against delay.
"""

def build_excel(activity_df: pd.DataFrame, summary_df: pd.DataFrame, criticality_df: pd.DataFrame):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        activity_df.to_excel(writer, sheet_name="Activity Analysis", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        criticality_df.to_excel(writer, sheet_name="Criticality", index=False)
    buffer.seek(0)
    return buffer

def run_analysis(df, display_unit, n_sims, random_seed, service_level):
    active = df[(df["Activity"] != "") | (df["Label"] != "")].copy()
    df_std = add_standardized_columns(active)
    df_results, expected_duration_std, critical_nodes = compute_schedule(df_std)
    sim_results_std, crit_df = simulate_project(df_std, n_sims=n_sims, random_seed=random_seed)
    sim_mean = convert_from_minutes(float(np.mean(sim_results_std)), display_unit)
    sim_std = convert_from_minutes(float(np.std(sim_results_std, ddof=1)), display_unit)
    p50 = convert_from_minutes(float(np.percentile(sim_results_std, 50)), display_unit)
    p80 = convert_from_minutes(float(np.percentile(sim_results_std, 80)), display_unit)
    p90 = convert_from_minutes(float(np.percentile(sim_results_std, 90)), display_unit)
    p95 = convert_from_minutes(float(np.percentile(sim_results_std, 95)), display_unit)
    service_deadline = convert_from_minutes(float(np.percentile(sim_results_std, service_level)), display_unit)
    hit_expected = float((sim_results_std <= expected_duration_std).mean())
    df_display = df_results.copy()
    for col_std, col_out in [
        ("expected_std", "Expected duration"),
        ("stddev_std", "Std. dev."),
        ("risk_range_std", "Risk range"),
        ("ES_std", "ES"),
        ("EF_std", "EF"),
        ("LS_std", "LS"),
        ("LF_std", "LF"),
        ("Slack_std", "Slack"),
    ]:
        df_display[col_out] = convert_series_from_minutes(df_display[col_std], display_unit)
    df_display["Variance"] = df_display["variance_std"] / (UNIT_TO_MINUTES[display_unit] ** 2)
    summary_df = pd.DataFrame({
        "Metric": [
            f"A. Mean duration ({display_unit})",
            f"Simulation std. dev. ({display_unit})",
            f"P50 ({display_unit})",
            f"P80 ({display_unit})",
            f"P90 ({display_unit})",
            f"P95 ({display_unit})",
            f"C. P{service_level} completion time ({display_unit})",
            "Probability of finishing within expected duration",
            "Expected critical path",
        ],
        "Value": [sim_mean, sim_std, p50, p80, p90, p95, service_deadline, hit_expected, " -> ".join(critical_nodes)]
    })
    return {
        "df_display": df_display,
        "sim_values_display": np.array([convert_from_minutes(x, display_unit) for x in sim_results_std]),
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "p50": p50,
        "p80": p80,
        "p90": p90,
        "p95": p95,
        "service_deadline": service_deadline,
        "critical_nodes": critical_nodes,
        "summary_df": summary_df,
        "criticality_df": crit_df,
        "hit_expected": hit_expected,
    }

with st.sidebar:
    st.header("Settings")
    project_name = st.text_input("Project name", value="Computer Design Project")
    display_unit = st.selectbox("Display results in", UNIT_OPTIONS, index=0)
    n_sims = st.slider("Monte Carlo simulations", min_value=1000, max_value=50000, value=12000, step=1000)
    service_level = st.slider("Target service level", min_value=50, max_value=99, value=95, step=1)
    random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    st.markdown("---")
    st.download_button("Download CSV template", data=CSV_TEMPLATE, file_name="project_template.csv", mime="text/csv")
    st.info("Each activity can use its own unit of measure. The app standardizes everything internally, then reports results in your selected display unit.")

st.title("Project Duration Analyzer")
st.caption("Consistent with the Computer Design assignment, but flexible enough for other companies and projects.")

tab1, tab2, tab3, tab4 = st.tabs(["Input data", "Assignment dashboard", "Advanced insights", "Outputs & guide"])

with tab1:
    st.subheader("1) Enter project activities")
    st.write("This version keeps the assignment wording while adding a per-activity unit dropdown beside Maximum duration.")
    source_choice = st.radio("Starting point", ["Use default computer design example", "Upload CSV"], horizontal=True)
    if source_choice == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            base_df = normalize_df(pd.read_csv(uploaded))
        else:
            base_df = DEFAULT_DATA.copy()
    else:
        base_df = DEFAULT_DATA.copy()

    edited_df = st.data_editor(
        base_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_order=["Activity", "Label", "Immediate predecessors", "Minimum duration", "Average duration", "Maximum duration", "Unit of measure", "Owner", "Phase"],
        column_config={
            "Activity": st.column_config.TextColumn("Activity"),
            "Label": st.column_config.TextColumn("Label"),
            "Immediate predecessors": st.column_config.TextColumn("Immediate predecessors", help="Use - if none, or comma-separated labels such as C,D"),
            "Minimum duration": st.column_config.NumberColumn("Minimum duration", min_value=0.0, step=1.0),
            "Average duration": st.column_config.NumberColumn("Average duration", min_value=0.0, step=1.0),
            "Maximum duration": st.column_config.NumberColumn("Maximum duration", min_value=0.0, step=1.0),
            "Unit of measure": st.column_config.SelectboxColumn("Unit of measure", options=UNIT_OPTIONS, required=True),
            "Owner": st.column_config.TextColumn("Owner"),
            "Phase": st.column_config.TextColumn("Phase"),
        },
        key="project_editor",
    )

    clean_df = normalize_df(edited_df)
    validation_errors = validate_df(clean_df)
    if validation_errors:
        for err in validation_errors:
            st.error(err)
    else:
        st.success("Input looks valid.")
        q1, q2, q3, q4 = st.columns(4)
        active = clean_df[(clean_df["Activity"] != "") | (clean_df["Label"] != "")]
        q1.metric("Activities", int(len(active)))
        q2.metric("Start nodes", int((active["Immediate predecessors"] == "-").sum()))
        q3.metric("Owners listed", int((active["Owner"] != "").sum()))
        q4.metric("Phases listed", int((active["Phase"] != "").sum()))
        st.dataframe(active, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Assignment dashboard")
    st.markdown("This tab is organized directly around the assignment questions: **A. mean duration**, **B. histogram**, and **C. 95% service-level completion time**.")
    validation_errors = validate_df(clean_df)
    if validation_errors:
        st.warning("Please fix the input issues in the Input data tab before running the analysis.")
    else:
        results = run_analysis(clean_df, display_unit, n_sims, int(random_seed), service_level)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("A. Mean duration", f"{results['sim_mean']:.2f} {display_unit}")
        c2.metric("Std. deviation", f"{results['sim_std']:.2f} {display_unit}")
        c3.metric(f"C. P{service_level} completion time", f"{results['service_deadline']:.2f} {display_unit}")
        c4.metric("Hit expected schedule", f"{results['hit_expected']:.1%}")
        c5.metric("Critical path", " → ".join(results["critical_nodes"]) if results["critical_nodes"] else "N/A")
        st.markdown("### Assignment answers")
        st.info(f"**A. Mean duration:** {results['sim_mean']:.2f} {display_unit}. **C. P{service_level} service level:** {results['service_deadline']:.2f} {display_unit}.")
        left, right = st.columns([1.3, 1])
        with left:
            st.pyplot(create_histogram(results["sim_values_display"], results["sim_mean"], results["service_deadline"], service_level, display_unit), use_container_width=True)
        with right:
            percentile_df = pd.DataFrame({
                "Measure": ["P50", "P80", "P90", "P95", f"P{service_level}"],
                f"Value ({display_unit})": [results["p50"], results["p80"], results["p90"], results["p95"], results["service_deadline"]]
            })
            st.markdown("#### Percentile reference points")
            st.dataframe(percentile_df, use_container_width=True, hide_index=True)
        st.markdown("### Activity-level results")
        show_cols = [
            "Activity", "Label", "Immediate predecessors", "Minimum duration", "Average duration", "Maximum duration", "Unit of measure", "Owner", "Phase",
            "Expected duration", "Std. dev.", "Variance", "Risk range", "ES", "EF", "LS", "LF", "Slack", "Critical"
        ]
        st.dataframe(results["df_display"][show_cols].sort_values(["ES", "Label"]), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Advanced insights")
    validation_errors = validate_df(clean_df)
    if validation_errors:
        st.warning("Please fix the input issues in the Input data tab before viewing advanced insights.")
    else:
        results = run_analysis(clean_df, display_unit, n_sims, int(random_seed), service_level)
        left, right = st.columns(2)
        with left:
            st.pyplot(create_gantt_chart(results["df_display"], display_unit), use_container_width=True)
        with right:
            st.pyplot(create_risk_chart(results["df_display"], display_unit), use_container_width=True)
        st.markdown("### Dependency network")
        gv = make_graphviz(results["df_display"], results["critical_nodes"])
        st.graphviz_chart(gv, use_container_width=True)
        st.markdown("### Delay hotspots")
        hotspot = results["df_display"].copy()
        hotspot["Priority"] = np.where(
            hotspot["Critical"] & (hotspot["Risk range"] >= hotspot["Risk range"].median()),
            "High",
            np.where(hotspot["Critical"], "Medium", "Monitor")
        )
        st.dataframe(
            hotspot[["Activity", "Label", "Owner", "Phase", "Unit of measure", "Risk range", "Slack", "Critical", "Priority"]].sort_values(["Priority", "Risk range"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True
        )
        if not results["criticality_df"].empty:
            st.markdown("### Simulation-based critical finish frequency")
            st.dataframe(results["criticality_df"], use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Outputs & guide")
    validation_errors = validate_df(clean_df)
    if validation_errors:
        st.warning("Please fix the input issues in the Input data tab before downloading outputs.")
    else:
        results = run_analysis(clean_df, display_unit, n_sims, int(random_seed), service_level)
        exec_summary = generate_exec_summary(project_name, results["sim_mean"], service_level, results["service_deadline"], results["p80"], results["p90"], results["p95"], results["critical_nodes"], display_unit)
        st.markdown("### Copy-ready executive summary")
        st.text_area("Summary", exec_summary, height=250)
        excel_file = build_excel(results["df_display"], results["summary_df"], results["criticality_df"])
        st.download_button("Download Excel results", data=excel_file, file_name="project_duration_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download executive summary as text", data=exec_summary, file_name="project_duration_summary.txt", mime="text/plain")
        st.markdown("### User guide")
        st.markdown(f"""
        **Why this version is more comprehensive**
        - Keeps the assignment wording and structure
        - Still answers A, B, and C directly
        - Supports per-activity time-unit dropdowns
        - Adds owners, phases, critical path, risk hotspots, and dependency visualization
        - Provides exportable outputs for submission or reporting

        **Recommended workflow**
        1. Enter activities and choose a unit of measure for each activity.
        2. Review the Assignment dashboard for the required outputs.
        3. Use Advanced insights for stronger interpretation and presentation value.
        4. Export the Excel file and summary.

        **Current display unit**
        - Results are currently shown in **{display_unit}**.
        """)
