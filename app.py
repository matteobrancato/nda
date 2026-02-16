import json
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Jira Timesheet Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    .metric-card {
        padding: 1.2rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.8rem; opacity: 0.9;
                       text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card h1 { margin: 0.3rem 0 0; font-size: 1.8rem; font-weight: 700; }
    .metric-card .sub { font-size: 0.75rem; opacity: 0.8; margin-top: 2px; }
    .mc-da  { background: linear-gradient(135deg, #51cf66, #37b24d); }
    .mc-tst { background: linear-gradient(135deg, #339af0, #1c7ed6); }
    .mc-nda { background: linear-gradient(135deg, #ff6b6b, #e03131); }
    .mc-tot { background: linear-gradient(135deg, #667eea, #764ba2); }
    .mc-lve { background: linear-gradient(135deg, #ffd43b, #fab005); color: #333; }
    .mc-prd { background: linear-gradient(135deg, #20c997, #0ca678); }
    .mc-dlv { background: linear-gradient(135deg, #845ef7, #7048e8); }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa; padding: 0.8rem; border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; border-radius: 8px 8px 0 0;
    }
    .section-divider { margin: 1.5rem 0; border-top: 2px solid #e9ecef; }
    .kw-tag {
        display: inline-block; background: #e9ecef; padding: 2px 8px;
        border-radius: 10px; font-size: 0.75rem; margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
JIRA_BASE_URL = "https://elab-aswatson.atlassian.net/browse/"
CONFIG_PATH = Path(__file__).parent / "keyword_config.json"

DEFAULT_CONFIG = {
    "nda_ticket_prefixes": ["NDA"],
    "nda_summary_keywords": [
        "Alignment Meetings", "Documentation creation", "Planning",
        "Setup", "Team Support", "Onboarding",
    ],
    "testing_summary_keywords": [
        "Duty Activities", "Regression Analysis", "Code refactoring",
        "Maintenance", "CI/CD maintenance",
        "Test Data Checks", "Retesting", "Bug Retesting",
        "Demand",
    ],
    "leave_keywords": [
        "Holiday", "Sick Leave", "Vacation", "PTO", "Leave",
        "Day Off", "Personal Leave", "Maternity", "Paternity",
        "Bereavement", "Ferie", "Malattia", "Permesso",
        "Public holidays",
    ],
}

# Status mapping
DELIVERED_STATUSES = {"Done", "In Review", "Closed", "Resolved", "Released"}
IN_PROGRESS_STATUSES = {"In Progress", "In Development", "In QA"}
NOT_STARTED_STATUSES = {"To Do", "Open", "Backlog", "New"}

CATEGORY_COLORS = {"NDA": "#ff6b6b", "DA": "#51cf66", "TESTING": "#339af0"}
DELIVERY_COLORS = {"Delivered": "#51cf66", "In Progress": "#ffd43b",
                   "Not Started": "#dee2e6", "N/A": "#adb5bd"}


# ---------------------------------------------------------------------------
# Keyword config persistence
# ---------------------------------------------------------------------------
def load_config() -> dict:
    """Load keyword config from JSON file, falling back to defaults."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                saved = json.load(f)
            # Merge with defaults so new keys are always present
            merged = DEFAULT_CONFIG.copy()
            merged.update(saved)
            return merged
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    """Persist keyword config to JSON file."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def get_config() -> dict:
    """Get config from session state, loading from disk on first run."""
    if "kw_config" not in st.session_state:
        st.session_state.kw_config = load_config()
    return st.session_state.kw_config


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
def contains_any(text: str, keywords: list[str],
                 word_boundary: bool = False) -> bool:
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        if word_boundary:
            if re.search(r'\b' + re.escape(kw_lower) + r'\b', text_lower):
                return True
        else:
            if kw_lower in text_lower:
                return True
    return False


def classify_nda(ticket_no: str, summary: str, cfg: dict) -> str:
    if contains_any(str(ticket_no), cfg["nda_ticket_prefixes"]):
        return "NDA"
    if contains_any(str(summary), cfg["nda_summary_keywords"]):
        return "NDA"
    return "NOT NDA"


def classify_category(nda_flag: str, summary: str, cfg: dict) -> str:
    if nda_flag == "NDA":
        return "NDA"
    if contains_any(str(summary), cfg["testing_summary_keywords"],
                    word_boundary=True):
        return "TESTING"
    return "DA"


def classify_delivery(status) -> str:
    if status is None or (isinstance(status, float) and pd.isna(status)):
        return "N/A"
    s = str(status).strip()
    if not s or s.lower() == "none":
        return "N/A"
    if s in DELIVERED_STATUSES:
        return "Delivered"
    if s in IN_PROGRESS_STATUSES:
        return "In Progress"
    if s in NOT_STARTED_STATUSES:
        return "Not Started"
    sl = s.lower()
    if any(k in sl for k in ["done", "closed", "resolved", "review",
                               "released"]):
        return "Delivered"
    if any(k in sl for k in ["progress", "development", "qa"]):
        return "In Progress"
    return "Not Started"


def detect_leave(summary: str) -> str | None:
    if not isinstance(summary, str):
        return None
    s = summary.lower()
    if any(k in s for k in ["public holiday", "holiday"]):
        return "Public Holiday"
    if any(k in s for k in ["vacation", "ferie"]):
        return "Vacation"
    if any(k in s for k in ["sick", "malattia"]):
        return "Sick Leave"
    if any(k in s for k in ["pto", "personal leave", "permesso"]):
        return "Personal Leave"
    if "maternity" in s:
        return "Maternity Leave"
    if "paternity" in s:
        return "Paternity Leave"
    if "bereavement" in s:
        return "Bereavement Leave"
    if "leave" in s:
        return "Other Leave"
    return None


def detect_issue_type(summary: str) -> str:
    if not isinstance(summary, str):
        return "Other"
    s = summary.lower()
    if detect_leave(summary):
        return "Leave"
    if any(k in s for k in ["meeting", "alignment", "standup", "sync",
                              "call"]):
        return "Meeting"
    if any(k in s for k in ["documentation", "ppts", "wiki"]):
        return "Documentation"
    if any(k in s for k in ["code review"]):
        return "Code Review"
    if any(k in s for k in ["ci/cd", "pipeline", "deploy", "release"]):
        return "CI/CD"
    if any(k in s for k in ["support", "helpdesk"]):
        return "Support"
    if any(k in s for k in ["onboarding", "knowledge transfer"]):
        return "Onboarding"
    if any(k in s for k in ["regression", "retesting", "smoke test",
                              "automation", "testim"]):
        return "Testing / QA"
    if re.search(r'\bqa\b', s):
        return "Testing / QA"
    if re.search(r'\btest\b', s):
        return "Testing / QA"
    if any(k in s for k in ["bug", "defect", "error", "crash"]):
        return "Bug Fix"
    if any(k in s for k in ["maintenance", "refactor", "cleanup", "duty"]):
        return "Maintenance"
    if any(k in s for k in ["feature", "implement", "create", "add ",
                              "upgrade", "ac ", "ac1", "ac2", "verify"]):
        return "Development"
    return "Task"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_flat_sheet(uploaded_file) -> pd.DataFrame | None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        target = None
        for name in xls.sheet_names:
            if "flat" in name.lower() and "group" in name.lower():
                target = name
                break
        if target is None:
            st.error(
                f"Sheet 'Flat (Groupable)' not found. "
                f"Available sheets: {', '.join(xls.sheet_names)}"
            )
            return None
        return pd.read_excel(xls, sheet_name=target)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower().rstrip(".")
        if cl in ("ticket no", "ticket no.", "issue key", "key", "ticket"):
            col_map[c] = "Ticket No"
        elif cl in ("summary", "issue summary"):
            col_map[c] = "Summary"
        elif cl in ("status", "issue status"):
            col_map[c] = "Status"
        elif cl in ("hr. spent", "hrs. spent", "hr spent", "logged hours",
                     "hours", "time spent (hours)", "time spent",
                     "log hours", "total hours"):
            col_map[c] = "Hours"
        elif cl in ("log user", "user", "username", "assignee",
                     "user name", "display name", "full name", "worker"):
            col_map[c] = "User"
        elif cl in ("worklog date", "log date", "date", "work date",
                     "created"):
            col_map[c] = "Date"
        elif cl in ("group name", "group"):
            col_map[c] = "Group"
        elif cl in ("ori. estm.", "ori. estm", "original estimate",
                     "estimate", "estimated hours"):
            col_map[c] = "Estimate"
        elif cl in ("total worklogs",):
            col_map[c] = "Total Worklogs"
        elif cl in ("epic link", "epic", "epic name", "parent"):
            col_map[c] = "Epic"
        elif cl in ("issue type", "issuetype", "type"):
            col_map[c] = "Issue Type"
        elif cl in ("project", "project name", "project key"):
            col_map[c] = "Project"
    return df.rename(columns=col_map)


def process_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if "Ticket No" not in df.columns:
        st.error("Column 'Ticket No' not found. Check your file format.")
        return df
    if "Summary" not in df.columns:
        df["Summary"] = ""
    if "Hours" not in df.columns:
        st.error("Could not find hours column (expected 'Hr. Spent').")
        return df

    df["Hours"] = pd.to_numeric(df["Hours"], errors="coerce").fillna(0)

    df["NDA Flag"] = df.apply(
        lambda r: classify_nda(str(r["Ticket No"]),
                               str(r["Summary"]), cfg), axis=1
    )
    df["Category"] = df.apply(
        lambda r: classify_category(r["NDA Flag"],
                                    str(r["Summary"]), cfg), axis=1
    )

    if "Status" in df.columns:
        df["Delivery"] = df["Status"].apply(classify_delivery)
    else:
        df["Status"] = "Unknown"
        df["Delivery"] = "Unknown"

    df["Leave Type"] = df["Summary"].apply(detect_leave)
    df["Is Leave"] = df["Leave Type"].notna()
    df["Issue Category"] = df["Summary"].apply(detect_issue_type)
    df["Jira Link"] = df["Ticket No"].apply(
        lambda x: f"{JIRA_BASE_URL}{x}"
        if pd.notna(x) and str(x).strip() else ""
    )
    df["Project"] = df["Ticket No"].apply(
        lambda x: str(x).split("-")[0]
        if pd.notna(x) and "-" in str(x) else "Unknown"
    )
    if "User" not in df.columns:
        df["User"] = "Unknown"

    # Sanitize columns to avoid Arrow serialization errors from mixed types.
    # Must happen after Is Leave is computed (it checks Leave Type notna).
    for col in df.columns:
        if col == "Is Leave":
            df[col] = df[col].astype(bool)
        elif col == "Hours":
            continue  # already numeric
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].fillna("").astype(str)

    return df


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def metric_card(label: str, value: str, css_class: str,
                sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card {css_class}">'
        f'<h3>{label}</h3><h1>{value}</h1>{sub_html}</div>'
    )


def safe_pct(num: float, den: float) -> float:
    return (num / den * 100) if den else 0


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make a dataframe safe for Arrow/st.dataframe serialization.

    Fixes mixed-type columns, None values in object columns, and
    ensures all dtypes are clean for pyarrow conversion.
    """
    out = df.copy()
    for col in out.columns:
        # Bool columns are fine
        if pd.api.types.is_bool_dtype(out[col]):
            continue
        # Numeric columns: coerce any stray non-numeric values
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce")
            continue
        # Everything else: force to string, replace None/NaN with ""
        try:
            out[col] = out[col].fillna("").astype(str)
        except (TypeError, ValueError):
            out[col] = out[col].apply(
                lambda x: str(x) if x is not None else ""
            )
    return out


def display_table(df: pd.DataFrame, cols: list[str], key: str = ""):
    available = [c for c in cols if c in df.columns]
    display = df[available].reset_index(drop=True)
    if "Ticket No" in display.columns and "Jira Link" not in display.columns:
        display["Jira Link"] = display["Ticket No"].apply(
            lambda t: f"{JIRA_BASE_URL}{t}" if pd.notna(t) else ""
        )
    display = sanitize_df(display)
    col_config = {"Hours": st.column_config.NumberColumn(format="%.2f")}
    if "Jira Link" in display.columns:
        col_config["Jira Link"] = st.column_config.LinkColumn("Jira Link")
    if "Estimate" in display.columns:
        col_config["Estimate"] = st.column_config.NumberColumn(format="%.1f")
    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config=col_config, key=key if key else None,
    )


# ---------------------------------------------------------------------------
# Keyword configuration UI
# ---------------------------------------------------------------------------
def render_keyword_config() -> bool:
    """Render the keyword configuration panel in sidebar.

    Returns True if config was changed and data needs reprocessing.
    """
    cfg = get_config()
    changed = False

    with st.sidebar.expander("Keyword Configuration", expanded=False):
        st.caption(
            "Edit the keywords used to classify tickets. "
            "Changes are saved automatically and persist between sessions."
        )

        # --- NDA ticket prefixes ---
        st.markdown("**NDA Ticket Prefixes**")
        st.caption("Tickets starting with these prefixes are classified as NDA")
        nda_prefixes_text = st.text_area(
            "NDA prefixes (one per line)",
            value="\n".join(cfg["nda_ticket_prefixes"]),
            height=68,
            key="cfg_nda_prefixes",
            label_visibility="collapsed",
        )
        new_nda_prefixes = [
            s.strip() for s in nda_prefixes_text.split("\n") if s.strip()
        ]

        # --- NDA summary keywords ---
        st.markdown("**NDA Summary Keywords**")
        st.caption(
            "If the summary contains any of these, "
            "the ticket is classified as NDA"
        )
        nda_summary_text = st.text_area(
            "NDA summary keywords (one per line)",
            value="\n".join(cfg["nda_summary_keywords"]),
            height=150,
            key="cfg_nda_summary",
            label_visibility="collapsed",
        )
        new_nda_summary = [
            s.strip() for s in nda_summary_text.split("\n") if s.strip()
        ]

        # --- Testing summary keywords ---
        st.markdown("**Testing Summary Keywords**")
        st.caption(
            "If NOT NDA, and the summary contains any of these, "
            "the ticket is classified as TESTING. "
            "Uses word-boundary matching to avoid false positives"
        )
        testing_text = st.text_area(
            "Testing summary keywords (one per line)",
            value="\n".join(cfg["testing_summary_keywords"]),
            height=150,
            key="cfg_testing",
            label_visibility="collapsed",
        )
        new_testing = [
            s.strip() for s in testing_text.split("\n") if s.strip()
        ]

        # --- Leave keywords ---
        st.markdown("**Leave Keywords**")
        st.caption("Used to detect and sub-classify leave entries")
        leave_text = st.text_area(
            "Leave keywords (one per line)",
            value="\n".join(cfg["leave_keywords"]),
            height=150,
            key="cfg_leave",
            label_visibility="collapsed",
        )
        new_leave = [
            s.strip() for s in leave_text.split("\n") if s.strip()
        ]

        # Check for changes
        if (new_nda_prefixes != cfg["nda_ticket_prefixes"]
                or new_nda_summary != cfg["nda_summary_keywords"]
                or new_testing != cfg["testing_summary_keywords"]
                or new_leave != cfg["leave_keywords"]):
            changed = True
            cfg["nda_ticket_prefixes"] = new_nda_prefixes
            cfg["nda_summary_keywords"] = new_nda_summary
            cfg["testing_summary_keywords"] = new_testing
            cfg["leave_keywords"] = new_leave
            st.session_state.kw_config = cfg
            save_config(cfg)
            st.success("Configuration saved!", icon="✅")

        st.markdown("---")

        # Reset button
        if st.button("Reset to defaults", key="cfg_reset"):
            st.session_state.kw_config = DEFAULT_CONFIG.copy()
            save_config(DEFAULT_CONFIG)
            st.rerun()

        # Summary of current config
        st.markdown("---")
        st.markdown("**Current classification logic:**")
        st.markdown(
            f"1. Ticket prefix in "
            f"`{cfg['nda_ticket_prefixes']}` → **NDA**\n"
            f"2. Summary matches NDA keywords "
            f"({len(cfg['nda_summary_keywords'])}) → **NDA**\n"
            f"3. Summary matches Testing keywords "
            f"({len(cfg['testing_summary_keywords'])}) → **TESTING**\n"
            f"4. Everything else → **DA**"
        )

    return changed


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    st.title("Jira Timesheet Dashboard")
    st.caption(
        "Upload a Jira Assistant Excel export to analyse timesheet data"
    )

    # Sidebar: upload
    with st.sidebar:
        st.header("Upload")
        uploaded = st.file_uploader(
            "Jira Assistant Excel (.xlsx)", type=["xlsx", "xls"]
        )
        st.markdown("---")
        st.markdown(
            "Automatically reads the **Flat (Groupable)** sheet "
            "regardless of how many sheets exist."
        )

    # Keyword configuration (always available, even before upload)
    config_changed = render_keyword_config()
    cfg = get_config()

    if uploaded is None:
        st.info(
            "Upload an Excel file exported from Jira Assistant. "
            "The dashboard will locate the "
            "'Flat (Groupable)' sheet automatically."
        )
        return

    # Load
    raw = load_flat_sheet(uploaded)
    if raw is None or raw.empty:
        return

    df = normalize_columns(raw)
    df = process_data(df, cfg)

    if "Category" not in df.columns:
        st.error("Processing failed.")
        return

    # Sidebar filters
    with st.sidebar:
        st.markdown("---")
        st.subheader("Filters")
        all_users = sorted(df["User"].dropna().unique().tolist())
        sel_users = st.multiselect("Person", all_users, default=all_users)

        all_groups = (sorted(df["Group"].dropna().unique().tolist())
                      if "Group" in df.columns else [])
        if all_groups:
            sel_groups = st.multiselect("Group", all_groups,
                                        default=all_groups)
        else:
            sel_groups = []

        all_cats = sorted(df["Category"].unique().tolist())
        sel_cats = st.multiselect("Category", all_cats, default=all_cats)

        all_statuses = sorted(df["Status"].dropna().unique().tolist())
        sel_statuses = st.multiselect("Status", all_statuses,
                                      default=all_statuses)

    # Apply filters
    mask = (df["User"].isin(sel_users)
            & df["Category"].isin(sel_cats)
            & df["Status"].isin(sel_statuses))
    if all_groups and sel_groups:
        mask = mask & df["Group"].isin(sel_groups)
    fdf = df[mask].copy()

    if fdf.empty:
        st.warning("No data matches the current filters.")
        return

    # -------------------------------------------------------------------
    # KPIs
    # -------------------------------------------------------------------
    total_h = fdf["Hours"].sum()
    da_h = fdf.loc[fdf["Category"] == "DA", "Hours"].sum()
    test_h = fdf.loc[fdf["Category"] == "TESTING", "Hours"].sum()
    nda_h = fdf.loc[fdf["Category"] == "NDA", "Hours"].sum()
    leave_h = fdf.loc[fdf["Is Leave"], "Hours"].sum()
    productive_h = da_h + test_h
    delivered_h = fdf.loc[fdf["Delivery"] == "Delivered", "Hours"].sum()

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.markdown(metric_card(
        "Total Hours", f"{total_h:,.1f}", "mc-tot"),
        unsafe_allow_html=True)
    c2.markdown(metric_card(
        "DA", f"{da_h:,.1f}h", "mc-da",
        f"{safe_pct(da_h, total_h):.0f}% of total"),
        unsafe_allow_html=True)
    c3.markdown(metric_card(
        "Testing", f"{test_h:,.1f}h", "mc-tst",
        f"{safe_pct(test_h, total_h):.0f}% of total"),
        unsafe_allow_html=True)
    c4.markdown(metric_card(
        "NDA", f"{nda_h:,.1f}h", "mc-nda",
        f"{safe_pct(nda_h, total_h):.0f}% of total"),
        unsafe_allow_html=True)
    c5.markdown(metric_card(
        "Leave", f"{leave_h:,.1f}h", "mc-lve",
        f"{safe_pct(leave_h, total_h):.0f}% of total"),
        unsafe_allow_html=True)
    c6.markdown(metric_card(
        "Productive", f"{safe_pct(productive_h, total_h):.0f}%", "mc-prd",
        f"{productive_h:,.1f}h"),
        unsafe_allow_html=True)
    c7.markdown(metric_card(
        "Delivered", f"{safe_pct(delivered_h, total_h):.0f}%", "mc-dlv",
        f"{delivered_h:,.1f}h"),
        unsafe_allow_html=True)

    st.markdown("")

    # -------------------------------------------------------------------
    # Tabs
    # -------------------------------------------------------------------
    tabs = st.tabs([
        "Overview", "By Person", "Category Detail",
        "Delivery Status", "Leave Analysis", "Raw Data",
    ])

    # === OVERVIEW ======================================================
    with tabs[0]:
        col_pie, col_right = st.columns([1, 1])

        with col_pie:
            st.subheader("Hours by Category")
            cat_totals = (fdf.groupby("Category")["Hours"]
                          .sum().reset_index()
                          .sort_values("Hours", ascending=False))
            fig_pie = px.pie(
                cat_totals, names="Category", values="Hours",
                color="Category", color_discrete_map=CATEGORY_COLORS,
                hole=0.45,
            )
            fig_pie.update_traces(
                textinfo="label+percent+value",
                texttemplate="%{label}<br>%{value:.1f}h (%{percent})",
                hovertemplate=(
                    "%{label}: %{value:.1f}h (%{percent})<extra></extra>"
                ),
            )
            fig_pie.update_layout(
                margin=dict(t=20, b=20, l=20, r=20), height=400,
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            drill = st.selectbox(
                "Drill into category",
                ["-- select --"] + cat_totals["Category"].tolist(),
            )
            if drill != "-- select --":
                sub = fdf[fdf["Category"] == drill].sort_values(
                    "Hours", ascending=False
                )
                st.markdown(
                    f"**{drill}** -- {sub['Hours'].sum():.1f}h total, "
                    f"{sub['Ticket No'].nunique()} unique tickets"
                )
                display_table(sub, [
                    "Ticket No", "Summary", "User", "Hours",
                    "Status", "Delivery", "Issue Category",
                ])

        with col_right:
            st.subheader("Key Ratios")
            ratios = {
                "DA / Total": f"{safe_pct(da_h, total_h):.1f}%",
                "Testing / Total": f"{safe_pct(test_h, total_h):.1f}%",
                "NDA / Total": f"{safe_pct(nda_h, total_h):.1f}%",
                "Leave / Total": f"{safe_pct(leave_h, total_h):.1f}%",
                "Productive (DA+Testing) / Total":
                    f"{safe_pct(productive_h, total_h):.1f}%",
                "Testing / Productive":
                    f"{safe_pct(test_h, productive_h):.1f}%",
                "DA / Productive":
                    f"{safe_pct(da_h, productive_h):.1f}%",
                "Delivered / Total":
                    f"{safe_pct(delivered_h, total_h):.1f}%",
                "Delivered / Productive":
                    f"{safe_pct(delivered_h, productive_h):.1f}%",
            }
            st.dataframe(
                sanitize_df(
                    pd.DataFrame(list(ratios.items()),
                                 columns=["Ratio", "Value"])
                ),
                use_container_width=True, hide_index=True,
            )

            st.subheader("Hours by Person & Category")
            pc = (fdf.groupby(["User", "Category"])["Hours"]
                  .sum().reset_index())
            fig_bar = px.bar(
                pc, x="User", y="Hours", color="Category",
                color_discrete_map=CATEGORY_COLORS, barmode="stack",
            )
            fig_bar.update_layout(
                margin=dict(t=20, b=20), height=350,
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Issue Type Distribution")
        it = (fdf.groupby("Issue Category")["Hours"]
              .sum().reset_index()
              .sort_values("Hours", ascending=False))
        fig_it = px.bar(
            it, x="Issue Category", y="Hours", color="Hours",
            color_continuous_scale="Viridis",
        )
        fig_it.update_layout(margin=dict(t=20, b=20), height=300)
        st.plotly_chart(fig_it, use_container_width=True)

        if "Group" in fdf.columns and fdf["Group"].nunique() > 1:
            st.subheader("Hours by Group")
            grp = (fdf.groupby(["Group", "Category"])["Hours"]
                   .sum().reset_index())
            fig_grp = px.bar(
                grp, x="Group", y="Hours", color="Category",
                color_discrete_map=CATEGORY_COLORS, barmode="stack",
            )
            fig_grp.update_layout(margin=dict(t=20, b=20), height=300)
            st.plotly_chart(fig_grp, use_container_width=True)

    # === BY PERSON =====================================================
    with tabs[1]:
        st.subheader("Person Summary")

        rows = []
        for user in sorted(fdf["User"].unique()):
            ud = fdf[fdf["User"] == user]
            t = ud["Hours"].sum()
            da = ud.loc[ud["Category"] == "DA", "Hours"].sum()
            tst = ud.loc[ud["Category"] == "TESTING", "Hours"].sum()
            nda = ud.loc[ud["Category"] == "NDA", "Hours"].sum()
            lv = ud.loc[ud["Is Leave"], "Hours"].sum()
            dlv = ud.loc[ud["Delivery"] == "Delivered", "Hours"].sum()
            rows.append({
                "User": user,
                "Total Hours": round(t, 1),
                "Unique Tickets": ud["Ticket No"].nunique(),
                "DA Hours": round(da, 1),
                "Testing Hours": round(tst, 1),
                "NDA Hours": round(nda, 1),
                "Leave Hours": round(lv, 1),
                "DA %": round(safe_pct(da, t), 1),
                "Testing %": round(safe_pct(tst, t), 1),
                "NDA %": round(safe_pct(nda, t), 1),
                "Productive %": round(safe_pct(da + tst, t), 1),
                "Delivered %": round(safe_pct(dlv, t), 1),
            })
        psummary = (pd.DataFrame(rows)
                    .sort_values("Total Hours", ascending=False))
        st.dataframe(sanitize_df(psummary), use_container_width=True,
                     hide_index=True)

        st.subheader("Category Distribution per Person")
        users_list = sorted(fdf["User"].unique())
        cols_per_row = min(4, len(users_list))
        for i in range(0, len(users_list), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(users_list):
                    break
                user = users_list[idx]
                ud = fdf[fdf["User"] == user]
                if ud.empty:
                    continue
                uc = (ud.groupby("Category")["Hours"]
                      .sum().reset_index())
                fig_u = px.pie(
                    uc, names="Category", values="Hours",
                    color="Category",
                    color_discrete_map=CATEGORY_COLORS,
                    hole=0.4, title=user,
                )
                fig_u.update_traces(textinfo="percent+value")
                fig_u.update_layout(
                    margin=dict(t=40, b=10, l=10, r=10),
                    height=250, showlegend=False, title_font_size=12,
                )
                col.plotly_chart(fig_u, use_container_width=True)

        st.markdown(
            '<div class="section-divider"></div>',
            unsafe_allow_html=True,
        )
        st.subheader("Person Deep Dive")
        sel_person = st.selectbox(
            "Select person", users_list, key="person_deep"
        )
        pd_data = fdf[fdf["User"] == sel_person]

        if not pd_data.empty:
            p_t = pd_data["Hours"].sum()
            p_da = pd_data.loc[
                pd_data["Category"] == "DA", "Hours"
            ].sum()
            p_tst = pd_data.loc[
                pd_data["Category"] == "TESTING", "Hours"
            ].sum()
            p_nda = pd_data.loc[
                pd_data["Category"] == "NDA", "Hours"
            ].sum()
            p_dlv = pd_data.loc[
                pd_data["Delivery"] == "Delivered", "Hours"
            ].sum()

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Total", f"{p_t:.1f}h")
            mc2.metric("DA",
                        f"{p_da:.1f}h ({safe_pct(p_da, p_t):.0f}%)")
            mc3.metric("Testing",
                        f"{p_tst:.1f}h ({safe_pct(p_tst, p_t):.0f}%)")
            mc4.metric("NDA",
                        f"{p_nda:.1f}h ({safe_pct(p_nda, p_t):.0f}%)")
            mc5.metric("Delivered",
                        f"{p_dlv:.1f}h ({safe_pct(p_dlv, p_t):.0f}%)")

            display_table(
                pd_data.sort_values("Hours", ascending=False),
                ["Ticket No", "Summary", "Hours", "Category", "Status",
                 "Delivery", "Issue Category", "Leave Type"],
                key="person_table",
            )

    # === CATEGORY DETAIL ===============================================
    with tabs[2]:
        st.subheader("Category Breakdown")

        for cat in ["DA", "TESTING", "NDA"]:
            color = CATEGORY_COLORS[cat]
            cat_data = fdf[fdf["Category"] == cat]
            if cat_data.empty:
                continue
            ch = cat_data["Hours"].sum()
            cp = safe_pct(ch, total_h)

            st.markdown(
                f"### <span style='color:{color}'>{cat}</span> "
                f"-- {ch:.1f}h ({cp:.0f}%)",
                unsafe_allow_html=True,
            )

            dlv_counts = (cat_data.groupby("Delivery")["Hours"]
                          .sum().reset_index())
            dc1, dc2 = st.columns([1, 2])
            with dc1:
                fig_dlv = px.pie(
                    dlv_counts, names="Delivery", values="Hours",
                    hole=0.4, color="Delivery",
                    color_discrete_map=DELIVERY_COLORS,
                )
                fig_dlv.update_traces(textinfo="percent+value")
                fig_dlv.update_layout(
                    margin=dict(t=10, b=10), height=220,
                    showlegend=True,
                )
                st.plotly_chart(fig_dlv, use_container_width=True)

            with dc2:
                top = (
                    cat_data
                    .groupby(["Ticket No", "Summary", "Status",
                              "Delivery"])["Hours"]
                    .sum().reset_index()
                    .sort_values("Hours", ascending=False)
                    .head(15)
                )
                display_table(
                    top,
                    ["Ticket No", "Summary", "Hours", "Status",
                     "Delivery"],
                    key=f"cat_{cat}_top",
                )

            ppl = (cat_data.groupby("User")["Hours"]
                   .sum().reset_index()
                   .sort_values("Hours", ascending=False))
            fig_p = px.bar(
                ppl, x="User", y="Hours",
                color_discrete_sequence=[color],
            )
            fig_p.update_layout(
                margin=dict(t=10, b=20), height=250,
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown("---")

    # === DELIVERY STATUS ===============================================
    with tabs[3]:
        st.subheader("Delivery Status Analysis")
        st.markdown(
            "Tickets classified as **Delivered** (Done, In Review), "
            "**In Progress**, or **Not Started** (To Do)."
        )

        d1, d2 = st.columns([1, 1])
        with d1:
            dlv_all = (fdf.groupby("Delivery")["Hours"]
                       .sum().reset_index())
            fig_da = px.pie(
                dlv_all, names="Delivery", values="Hours",
                color="Delivery",
                color_discrete_map=DELIVERY_COLORS, hole=0.45,
            )
            fig_da.update_traces(
                textinfo="label+percent+value",
                texttemplate="%{label}<br>%{value:.1f}h (%{percent})",
            )
            fig_da.update_layout(
                margin=dict(t=20, b=20), height=380,
            )
            st.plotly_chart(fig_da, use_container_width=True)

        with d2:
            dc = (fdf.groupby(["Category", "Delivery"])["Hours"]
                  .sum().reset_index())
            fig_dc = px.bar(
                dc, x="Category", y="Hours", color="Delivery",
                color_discrete_map=DELIVERY_COLORS, barmode="stack",
            )
            fig_dc.update_layout(
                margin=dict(t=20, b=20), height=380,
            )
            st.plotly_chart(fig_dc, use_container_width=True)

        st.subheader("Delivery by Person")
        dp = (fdf.groupby(["User", "Delivery"])["Hours"]
              .sum().reset_index())
        fig_dp = px.bar(
            dp, x="User", y="Hours", color="Delivery",
            color_discrete_map=DELIVERY_COLORS, barmode="stack",
        )
        fig_dp.update_layout(
            margin=dict(t=20, b=20), height=350,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_dp, use_container_width=True)

        st.subheader("Delivery Summary per Person")
        dlv_rows = []
        for user in sorted(fdf["User"].unique()):
            ud = fdf[fdf["User"] == user]
            t = ud["Hours"].sum()
            dlv = ud.loc[
                ud["Delivery"] == "Delivered", "Hours"
            ].sum()
            ip = ud.loc[
                ud["Delivery"] == "In Progress", "Hours"
            ].sum()
            ns = ud.loc[
                ud["Delivery"] == "Not Started", "Hours"
            ].sum()
            dlv_rows.append({
                "User": user,
                "Total Hours": round(t, 1),
                "Delivered": round(dlv, 1),
                "In Progress": round(ip, 1),
                "Not Started": round(ns, 1),
                "Delivered %": round(safe_pct(dlv, t), 1),
            })
        st.dataframe(
            sanitize_df(
                pd.DataFrame(dlv_rows)
                .sort_values("Total Hours", ascending=False)
            ),
            use_container_width=True, hide_index=True,
        )

        # Delivered tickets per person with expandable detail
        st.subheader("Delivered Tickets per Person")
        delivered = fdf[fdf["Delivery"].isin(["Delivered"])]
        if delivered.empty:
            st.info("No delivered tickets found.")
        else:
            for user in sorted(delivered["User"].unique()):
                ud = delivered[delivered["User"] == user]
                # Aggregate unique tickets
                user_tickets = (
                    ud.groupby(["Ticket No", "Summary", "Status"])
                    ["Hours"].sum().reset_index()
                    .sort_values("Hours", ascending=False)
                )
                n_tickets = len(user_tickets)
                total_h_user = user_tickets["Hours"].sum()

                with st.expander(
                    f"**{user}** -- {n_tickets} tickets, "
                    f"{total_h_user:.1f}h delivered"
                ):
                    user_tickets["Jira Link"] = user_tickets[
                        "Ticket No"
                    ].apply(
                        lambda t: f"{JIRA_BASE_URL}{t}"
                        if pd.notna(t) else ""
                    )
                    st.dataframe(
                        sanitize_df(user_tickets),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Jira Link": st.column_config.LinkColumn(
                                "Jira Link"
                            ),
                            "Hours": st.column_config.NumberColumn(
                                format="%.2f"
                            ),
                        },
                    )

        st.markdown("---")
        st.subheader("Not Yet Delivered Tickets (In Progress / To Do)")
        not_dlv = fdf[fdf["Delivery"].isin(["In Progress", "Not Started"])]
        if not_dlv.empty:
            st.success("All tickets are delivered.")
        else:
            not_dlv_agg = (
                not_dlv
                .groupby(["Ticket No", "Summary", "Status",
                          "Delivery", "Category"])["Hours"]
                .sum().reset_index()
                .sort_values("Hours", ascending=False)
            )
            display_table(
                not_dlv_agg,
                ["Ticket No", "Summary", "Hours", "Status",
                 "Delivery", "Category"],
                key="not_delivered",
            )

    # === LEAVE ANALYSIS ================================================
    with tabs[4]:
        st.subheader("Leave Analysis")
        leave_data = fdf[fdf["Is Leave"]].copy()

        if leave_data.empty:
            st.info("No leave entries detected.")
        else:
            l_total = leave_data["Hours"].sum()
            st.markdown(
                f"**Total leave: {l_total:.1f}h** "
                f"({safe_pct(l_total, total_h):.1f}% of all hours)"
            )

            lc1, lc2 = st.columns(2)
            with lc1:
                lt = (leave_data.groupby("Leave Type")["Hours"]
                      .sum().reset_index()
                      .sort_values("Hours", ascending=False))
                fig_lt = px.pie(
                    lt, names="Leave Type", values="Hours", hole=0.4,
                    color_discrete_sequence=(
                        px.colors.qualitative.Set3
                    ),
                )
                fig_lt.update_traces(textinfo="label+percent+value")
                fig_lt.update_layout(
                    margin=dict(t=20, b=20), height=350,
                )
                st.plotly_chart(fig_lt, use_container_width=True)

            with lc2:
                lp = (leave_data
                      .groupby(["User", "Leave Type"])["Hours"]
                      .sum().reset_index())
                fig_lp = px.bar(
                    lp, x="User", y="Hours", color="Leave Type",
                    barmode="stack",
                    color_discrete_sequence=(
                        px.colors.qualitative.Set3
                    ),
                )
                fig_lp.update_layout(
                    margin=dict(t=20, b=20), height=350,
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig_lp, use_container_width=True)

            st.subheader("Leave per Person")
            lv_rows = []
            for user in sorted(leave_data["User"].unique()):
                ud = leave_data[leave_data["User"] == user]
                row = {
                    "User": user,
                    "Total Leave Hours": round(ud["Hours"].sum(), 1),
                }
                for lt_name in sorted(leave_data["Leave Type"].unique()):
                    row[lt_name] = round(
                        ud.loc[ud["Leave Type"] == lt_name,
                               "Hours"].sum(), 1
                    )
                lv_rows.append(row)
            st.dataframe(
                sanitize_df(pd.DataFrame(lv_rows)),
                use_container_width=True, hide_index=True,
            )

            st.subheader("Leave Detail")
            display_table(
                leave_data.sort_values(
                    ["User", "Hours"], ascending=[True, False]
                ),
                ["User", "Ticket No", "Summary", "Leave Type",
                 "Hours", "Status"],
                key="leave_detail",
            )

    # === RAW DATA ======================================================
    with tabs[5]:
        st.subheader("Processed Data")
        st.markdown(f"**{len(fdf)} rows** after filters")

        display_table(fdf, list(fdf.columns), key="raw_data")

        csv = fdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered data as CSV",
            data=csv,
            file_name=(
                f"jira_report_{datetime.now().strftime('%Y%m%d')}.csv"
            ),
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
