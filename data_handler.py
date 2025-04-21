# -*- coding: utf-8 -*-
# --- Imports ---
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import traceback
import logging # Logger type hint comes from here
import json
from typing import Dict, Any, Tuple, Optional, Sequence, List # Keep this import
import os
from pathlib import Path
import time

# File locking import (Unix-specific)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# --- Pandas Option ---
pd.set_option('future.no_silent_downcasting', True)

# --- Constants ---
CORE_REQUIRED_THRESHOLD_COLS = ["Over_Capacity", "Unusual_Spike"]
EXPECTED_THRESHOLD_COLS = ["SiteID", "Over_Capacity", "Unusual_Spike", "Repeated_Days", "station_name"]
DEFAULT_REPEATED_DAYS = 4
STATIC_MIN_THRESHOLD = 0
BUFFER_PERCENTAGE = 0.10
BUFFER_NUM_BANDS = 8
BUFFER_START_COLOR_RGBA = (128, 0, 128, 0.2)
BUFFER_END_COLOR_RGBA = (128, 0, 128, 0.0)

# --- Path Definition ---
try:
    script_dir = Path(__file__).resolve().parent
    csv_filename = "thresholds.csv"
    THRESHOLDS_CSV_PATH = script_dir / csv_filename
    if not THRESHOLDS_CSV_PATH.is_file():
        THRESHOLDS_CSV_PATH_FALLBACK = Path(csv_filename)
        if THRESHOLDS_CSV_PATH_FALLBACK.is_file():
             THRESHOLDS_CSV_PATH = THRESHOLDS_CSV_PATH_FALLBACK
             print(f"WARNING: Threshold file not found at '{script_dir / csv_filename}'. Falling back to relative path '{csv_filename}' ({THRESHOLDS_CSV_PATH}).")
        else:
             print(f"ERROR: Threshold file not found at primary path '{script_dir / csv_filename}' or fallback relative path '{csv_filename}'.")
             if not THRESHOLDS_CSV_PATH.exists(): # Check if original path still doesn't exist
                 print(f"ERROR: Neither primary nor fallback threshold file path exists.")

    else:
        print(f"INFO: Using threshold file path: {THRESHOLDS_CSV_PATH}")
except NameError:
    csv_filename = "thresholds.csv"; THRESHOLDS_CSV_PATH = Path(csv_filename)
    print(f"WARNING: Could not determine script directory. Using relative path: {THRESHOLDS_CSV_PATH}")
    if not THRESHOLDS_CSV_PATH.is_file(): print(f"ERROR: Relative threshold file path '{THRESHOLDS_CSV_PATH}' does not exist.")

# --- Global Threshold Variable ---
thresholds_df_global = None

# --- File Locking Functions ---
def acquire_lock(file_handle, lock_type, logger, timeout=5):
    """Tries to acquire a file lock (shared or exclusive). Returns True on success, False on timeout/error."""
    if not HAS_FCNTL: return True
    start_time = time.time(); lock_type_str = "Shared" if lock_type == fcntl.LOCK_SH else "Exclusive"
    while time.time() - start_time < timeout:
        try:
            fcntl.flock(file_handle, lock_type | fcntl.LOCK_NB); logger.debug(f"{lock_type_str} lock acquired for {file_handle.name}"); return True
        except (BlockingIOError, OSError) as e:
            if isinstance(e, OSError) and e.errno not in [11, 13]: logger.error(f"Unexpected OSError ({e.errno}) acquiring lock: {e}", exc_info=True); raise
            time.sleep(0.1)
    logger.error(f"Could not acquire {lock_type_str} lock on {file_handle.name} within {timeout}s."); return False

def release_lock(file_handle, logger):
    """Releases a file lock if fcntl is available and handle is valid."""
    if HAS_FCNTL and file_handle and not file_handle.closed:
        try: fcntl.flock(file_handle, fcntl.LOCK_UN); logger.debug(f"Lock released for {file_handle.name}")
        except Exception as e: logger.error(f"Error releasing lock for {file_handle.name}: {e}", exc_info=True)

# --- Threshold Loading Function ---
def load_thresholds(file_path: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Loads the thresholds CSV file into a DataFrame using file locking. Returns None on error."""
    global thresholds_df_global
    file_path_str = str(file_path); logger.info(f"Attempting to load thresholds from: {file_path_str}")
    f = None; lock_acquired = False
    try:
        if not file_path.is_file():
             raise FileNotFoundError(f"File not found at '{file_path_str}'")
        f = open(file_path_str, 'r')
        lock_acquired = acquire_lock(f, fcntl.LOCK_SH if HAS_FCNTL else 0, logger)

        if not lock_acquired:
            logger.error(f"Failed to acquire read lock for {file_path_str}. Aborting load.")
            if f: # Check if file handle exists before trying to close
                f.close()
            return None

        thresholds_df = pd.read_csv(f); logger.info(f"Successfully read CSV. Validating columns...")
        missing_core = [c for c in CORE_REQUIRED_THRESHOLD_COLS if c not in thresholds_df.columns]
        if missing_core: logger.error(f"Missing CORE columns in '{file_path_str}': {missing_core}"); return None
        if "SiteID" not in thresholds_df.columns: logger.error(f"'SiteID' column missing in '{file_path_str}'."); return None
        if 'station_name' not in thresholds_df.columns: logger.warning(f"'station_name' column missing in '{file_path_str}'."); thresholds_df['station_name'] = 'N/A'
        thresholds_df["SiteID_str"] = thresholds_df["SiteID"].astype(str)
        logger.info(f"Thresholds loaded successfully from '{file_path_str}'."); thresholds_df_global = thresholds_df; return thresholds_df
    except Exception as e: logger.error(f"Error loading thresholds from '{file_path_str}': {e}", exc_info=True); return None
    finally:
        if lock_acquired and f: release_lock(f, logger)
        if f and not f.closed: f.close()
# --- END load_thresholds Function ---

# --- Get Site Thresholds Function ---
def get_site_thresholds(site_id: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """ Gets and validates thresholds for a specific site from the global DataFrame. """
    logger.info(f"Getting thresholds for SiteID {site_id}...")
    if thresholds_df_global is None or thresholds_df_global.empty: logger.error("Global thresholds empty."); return None
    if "SiteID_str" not in thresholds_df_global.columns: logger.error("'SiteID_str' missing."); return None
    site_row = thresholds_df_global[thresholds_df_global["SiteID_str"] == str(site_id)]
    if site_row.empty: logger.warning(f"SiteID {site_id} not found."); return None
    try:
        row = site_row.iloc[0]; v_thr = {"min_val": float(STATIC_MIN_THRESHOLD)}; missing = []; errors = []
        for col in CORE_REQUIRED_THRESHOLD_COLS:
            raw = row.get(col)
            if raw is None or pd.isna(raw): missing.append(f"'{col}' (missing)"); continue
            num = pd.to_numeric(raw, errors='coerce')
            if pd.isna(num): missing.append(f"'{col}' ('{raw}' not numeric)")
            else:
                if col=="Over_Capacity": v_thr["max_val"] = float(num)
                elif col=="Unusual_Spike": v_thr["spike_unusual"] = float(num)
        raw_rep = row.get("Repeated_Days")
        if raw_rep is None or pd.isna(raw_rep):
            logger.warning(f"Site {site_id}: Repeated_Days missing/NaN. Using default {DEFAULT_REPEATED_DAYS}")
            v_thr["repeated_days"] = int(DEFAULT_REPEATED_DAYS)
        else:
            try:
                rep_int = int(pd.to_numeric(raw_rep, errors='raise'))
                if rep_int >= 2: v_thr["repeated_days"] = rep_int
                else: errors.append(f"'Repeated_Days' ({rep_int}) < 2. Using default."); v_thr["repeated_days"] = int(DEFAULT_REPEATED_DAYS)
            except (ValueError, TypeError): errors.append(f"'Repeated_Days' ('{raw_rep}') invalid. Using default."); v_thr["repeated_days"] = int(DEFAULT_REPEATED_DAYS)
        if missing: logger.error(f"Missing/invalid CORE thresholds SiteID {site_id}: {missing}"); return None
        if errors: [logger.warning(f"SiteID {site_id}: {e}") for e in errors]
        if "max_val" not in v_thr or "spike_unusual" not in v_thr: logger.error(f"Internal error populating thresholds dict SiteID {site_id}."); return None
        logger.info(f"Thresholds validated SiteID {site_id}: {v_thr}"); return v_thr
    except Exception as e: logger.error(f"Unexpected error validating thresholds SiteID {site_id}: {e}", exc_info=True); return None

# --- Apply Flagging Function ---
def apply_flagging(df: pd.DataFrame, thresholds: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """ Applies data quality flags based on provided thresholds. """
    logger.info("Applying flagging logic...")
    required_keys = ['min_val', 'max_val', 'spike_unusual', 'repeated_days']
    flag_cols = ['FLAG_LESS_THAN_Min._Value', 'FLAG_ZERO', 'FLAG_REPEATED', 'FLAG_GREATER_THAN_MaxValue', 'UNUSUAL_SPIKE', 'FLAG_BELOW_CAPACITY']
    def ensure_cols(d): d['FLAGGED']=False; [d.setdefault(c, False) for c in flag_cols]; return d
    if not thresholds or not all(k in thresholds for k in required_keys): logger.error(f"Invalid thresholds for flagging (missing: {[k for k in required_keys if not thresholds or k not in thresholds]})"); return ensure_cols(df.copy())
    min_v, max_v, spike, rep_days = thresholds["min_val"], thresholds["max_val"], thresholds["spike_unusual"], thresholds["repeated_days"]
    logger.info(f"Flagging with: Min={min_v}, Max={max_v}, Spike={spike}, RepDays={rep_days}")
    df_p = df.copy()
    if 'DISCHARGE' not in df_p.columns: logger.warning("'DISCHARGE' column missing."); return ensure_cols(df_p)
    df_p['DISCHARGE'] = pd.to_numeric(df_p['DISCHARGE'], errors='coerce')
    df_p['FLAG_LESS_THAN_Min._Value'] = (df_p['DISCHARGE'] < min_v) & (df_p['DISCHARGE'].notna()) & (df_p['DISCHARGE'] != 0)
    df_p['FLAG_ZERO'] = df_p['DISCHARGE'] == 0
    df_p['FLAG_BELOW_CAPACITY'] = (df_p['DISCHARGE'] < 0) & (df_p['DISCHARGE'].notna())
    df_p['FLAG_GREATER_THAN_MaxValue'] = (df_p['DISCHARGE'] > max_v) & (df_p['DISCHARGE'].notna())
    df_p['RATE_OF_CHANGE'] = df_p['DISCHARGE'].diff().abs()
    df_p['UNUSUAL_SPIKE'] = (df_p['RATE_OF_CHANGE'] > spike) & (df_p['RATE_OF_CHANGE'].notna())
    df_p['FLAG_REPEATED'] = False
    non_zero = df_p['DISCHARGE'].where((df_p['DISCHARGE'] != 0) & df_p['DISCHARGE'].notna())
    if not non_zero.isna().all(): g_ids=(non_zero != non_zero.shift()).cumsum(); r_counts=non_zero.groupby(g_ids).transform('size'); df_p.loc[non_zero.notna(), 'FLAG_REPEATED'] = r_counts >= rep_days
    existing = [c for c in flag_cols if c in df_p.columns]; df_p['FLAGGED'] = df_p[existing].any(axis=1) if existing else False
    logger.info(f"Flagging complete. Total flagged: {df_p['FLAGGED'].sum()}")
    return df_p

# --- Date Validation Function ---
def validate_date(date_str):
    """Validate date string format YYYY-MM-DD. Returns datetime object or None."""
    if not date_str: return None
    try: return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError: return None

# --- Color Interpolation Function ---
def interpolate_color(c1, c2, fr):
    """ Interpolates between two RGBA tuples based on fraction (0.0 to 1.0). """
    r1,g1,b1,a1=c1; r2,g2,b2,a2=c2; fr=max(0.0,min(1.0,fr)); r=int(r1+(r2-r1)*fr); g=int(g1+(g2-g1)*fr); b=int(b1+(b2-b1)*fr); a=a1+(a2-a1)*fr; r=max(0,min(255,r)); g=max(0,min(255,g)); b=max(0,min(255,b)); return f'rgba({r},{g},{b},{a:.4f})'

# --- Gradient Buffer Function ---
def add_gradient_buffer(fig, dates, mean_value, buffer, start_color, end_color, num_bands, logger):
    """ Adds gradient filled buffer bands around a central line value. """
    if buffer <= 0 or num_bands <= 0: logger.warning("Skipping gradient: invalid buffer/bands."); return
    n=len(dates);
    if n<2: logger.warning("Skipping gradient: not enough points."); return
    x_poly = list(dates) + list(dates)[::-1]
    for i in range(num_bands - 1, -1, -1):
        outer_f=(i+1)/num_bands; inner_f=i/num_bands; band_color=interpolate_color(start_color, end_color, outer_f)
        y_low_u=mean_value+inner_f*buffer; y_high_u=mean_value+outer_f*buffer
        if np.isfinite(y_low_u) and np.isfinite(y_high_u): y_upper = [y_low_u]*n + [y_high_u]*n; fig.add_trace(go.Scatter(x=x_poly, y=y_upper, fill='toself', fillcolor=band_color, line=dict(width=0), hoverinfo="skip", showlegend=False, mode='lines'))
        y_high_l=mean_value-inner_f*buffer; y_low_l=mean_value-outer_f*buffer
        if np.isfinite(y_high_l) and np.isfinite(y_low_l): y_lower = [y_low_l]*n + [y_high_l]*n; fig.add_trace(go.Scatter(x=x_poly, y=y_lower, fill='toself', fillcolor=band_color, line=dict(width=0), hoverinfo="skip", showlegend=False, mode='lines'))

# --- Core Plot Generation Function ---
# LATEST: Includes Max Threshold in Legend
def generate_plot_for_site(site_id: str, start_date_str_requested: Optional[str], end_date_str_requested: Optional[str], is_reset: bool, logger: logging.Logger) -> Tuple[Optional[go.Figure], Optional[str], str, str, str, str, Optional[Dict], Optional[Dict]]:
    """
    Generates Plotly figure for a site with legend positioned on the right,
    increased font sizes, and max threshold in legend.
    Returns: tuple(fig, error_msg, station_name, start_date_actual, end_date_actual, units, site_thresholds_dict, stats_dict)
    """
    station_name = "N/A"; units = 'Unknown Units'; site_thresholds = None; stats_dict = None
    actual_start_date_str = start_date_str_requested or ""; actual_end_date_str = end_date_str_requested or ""
    logger.info(f"Plot Gen Start: Site={site_id}, ReqStart={start_date_str_requested}, ReqEnd={end_date_str_requested}, Reset={is_reset}")

    # 1. Get Thresholds
    site_thresholds = get_site_thresholds(site_id, logger)
    if thresholds_df_global is not None and 'station_name' in thresholds_df_global.columns and 'SiteID_str' in thresholds_df_global.columns:
        site_row = thresholds_df_global[thresholds_df_global['SiteID_str'] == str(site_id)]
        if not site_row.empty: station_name = site_row['station_name'].iloc[0]
    if site_thresholds is None: err = f"Thresholds missing/invalid for SiteID {site_id}."; logger.error(err); return None, err, station_name, actual_start_date_str, actual_end_date_str, units, None, None

    # 2. Fetch Data
    api_end = datetime.now().strftime('%Y-%m-%d') if (is_reset or not validate_date(end_date_str_requested)) else end_date_str_requested
    api_start = "1900-01-01" if (is_reset or not validate_date(start_date_str_requested)) else start_date_str_requested
    logger.info(f"API Call Params: Start={api_start}, End={api_end}")
    try:
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&start_date={api_start}&end_date={api_end}&f=json"
        logger.info(f"Fetching: {api_url}")
        response = requests.get(api_url, timeout=45); response.raise_for_status(); data = response.json()
    except requests.exceptions.RequestException as e: err = f"API Error site {site_id}: {e}"; logger.error(err, exc_info=True); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None
    except (json.JSONDecodeError, ValueError) as e: err = f"JSON Decode Error site {site_id}: {e}. Snippet: {response.text[:200]}..."; logger.error(err); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None

    # 3. Process Data
    metadata = {f: data.get(f, "N/A") for f in ["station_name", "units"]}; units = metadata.get('units', 'Unknown Units'); units = units if units and units!='N/A' else 'Unknown Units'
    if metadata.get('station_name') and metadata['station_name'] != 'N/A': station_name = metadata['station_name']
    logger.info(f"API Meta: Name={station_name}, Units={units}")
    if "data" not in data or not isinstance(data["data"], list) or not data["data"]: err = f"No 'data' in API response site {site_id}."; logger.warning(err); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None
    try:
        df = pd.DataFrame(data["data"], columns=["date", "value"]); df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce'); df.dropna(subset=['Date'], inplace=True); df = df.sort_values('Date').reset_index(drop=True)
        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')
    except Exception as e: err = f"Error processing API data site {site_id}: {e}"; logger.error(err, exc_info=True); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None
    if df.empty: err = f"No valid data points after processing site {site_id}."; logger.warning(err); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None

    # 4. Filter Dates
    min_data_dt, max_data_dt = df['Date'].min(), df['Date'].max(); start_req_dt_obj = validate_date(start_date_str_requested); end_req_dt_obj = validate_date(end_date_str_requested)
    start_dt_final = min_data_dt if (is_reset or not start_req_dt_obj) else max(start_req_dt_obj, min_data_dt); end_dt_final = max_data_dt if (is_reset or not end_req_dt_obj) else min(end_req_dt_obj, max_data_dt)
    if pd.isna(start_dt_final) or pd.isna(end_dt_final) or start_dt_final > end_dt_final: logger.warning("Date range invalid/no overlap. Using full data range."); start_dt_final, end_dt_final = min_data_dt, max_data_dt
    actual_start_date_str = start_dt_final.strftime('%Y-%m-%d') if pd.notna(start_dt_final) else ""; actual_end_date_str = end_dt_final.strftime('%Y-%m-%d') if pd.notna(end_dt_final) else ""
    logger.info(f"Final Plot Range: {actual_start_date_str} to {actual_end_date_str}")
    df_filtered = df.loc[(df['Date'] >= start_dt_final) & (df['Date'] <= end_dt_final)].copy().reset_index(drop=True) if pd.notna(start_dt_final) and pd.notna(end_dt_final) else pd.DataFrame()
    if df_filtered.empty: err = f"No data for site {site_id} in range [{actual_start_date_str} to {actual_end_date_str}]."; logger.warning(err); return None, err, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds, None
    df = df_filtered; logger.info(f"Processing {len(df)} points after date filtering.")

    # 5. Apply Flagging
    df = apply_flagging(df, site_thresholds, logger)

    # 6. Create Plot Figure
    plot_title = f"Data from {actual_start_date_str} to {actual_end_date_str}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['DISCHARGE'], mode='lines', name='Discharge', line=dict(color='lightgray', width=1.5), connectgaps=False, hoverinfo='skip', showlegend=False))
    min_val_thresh = site_thresholds.get("min_val", float('nan')); max_val_thresh = site_thresholds.get("max_val", float('nan'))
    spike_unusual_thresh = site_thresholds.get("spike_unusual", float('nan')); repeated_days_thresh = site_thresholds.get("repeated_days", DEFAULT_REPEATED_DAYS)
    fmt_spike = f"{spike_unusual_thresh:.2f}" if pd.notna(spike_unusual_thresh) else "N/A"; fmt_max = f"{max_val_thresh:.2f}" if pd.notna(max_val_thresh) else "N/A"
    flag_plot_info = {'FLAG_BELOW_CAPACITY': ('red', 'Below Measuring Capacity'), 'FLAG_ZERO': ('blue', 'Zero Discharge'), 'FLAG_REPEATED': ('green', f'Repeated (>{repeated_days_thresh}d)'), 'FLAG_GREATER_THAN_MaxValue': ('purple', f'Over Max ({fmt_max})'), 'UNUSUAL_SPIKE': ('orange', f"Spike (>{fmt_spike})")}
    hover_tmpl = (f'<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Value:</b> %{{y:.2f}} {units}<br><b>Flag:</b> %{{meta}}<extra></extra>')
    for flag_col, (color, label) in flag_plot_info.items():
        if flag_col in df.columns and df[flag_col].any():
            subset = df.loc[df[flag_col]]; fig.add_trace(go.Scatter(x=subset['Date'], y=subset['DISCHARGE'], mode='markers', marker=dict(color=color, size=6, symbol='circle'), name=label, meta=label, hovertemplate=hover_tmpl, showlegend=True))

    # Add threshold lines
    min_plot_dt, max_plot_dt = df["Date"].min(), df["Date"].max()
    if pd.notna(min_plot_dt) and pd.notna(max_plot_dt):
        plot_date_range = [min_plot_dt, max_plot_dt]
        if pd.notna(min_val_thresh) and min_val_thresh != 0:
            fig.add_trace(go.Scatter(x=plot_date_range, y=[min_val_thresh]*2, mode='lines', name=f"Min Thr ({min_val_thresh:.2f})", line=dict(color="gray", dash="dash", width=1), hoverinfo='skip', showlegend=False)) # Min still hidden
        if pd.notna(max_val_thresh):
            fig.add_trace(go.Scatter(x=plot_date_range, y=[max_val_thresh]*2, mode='lines', name=f"Max Thr ({max_val_thresh:.2f})", line=dict(color="purple", dash="dash", width=1), hoverinfo='skip', showlegend=True)) # Max now shown
            if max_val_thresh > 0:
                buffer = max_val_thresh * BUFFER_PERCENTAGE
                if len(df['Date']) >= 2: add_gradient_buffer(fig, df['Date'], max_val_thresh, buffer, BUFFER_START_COLOR_RGBA, BUFFER_END_COLOR_RGBA, BUFFER_NUM_BANDS, logger)

    # 6a. Calculate Statistics
    stats_dict = None; discharge_num = df['DISCHARGE'].dropna()
    if not discharge_num.empty: stats_dict = {"count": f"{discharge_num.count():,}", "mean": f"{discharge_num.mean():.2f}" if pd.notna(discharge_num.mean()) else "N/A", "min": f"{discharge_num.min():.2f}" if pd.notna(discharge_num.min()) else "N/A", "max": f"{discharge_num.max():.2f}" if pd.notna(discharge_num.max()) else "N/A", "units": units}; logger.info(f"Stats: {stats_dict}")
    else: logger.warning("No numeric discharge data for stats.")

    # 7. Finalize Layout
    fig.update_layout(
        title=dict(text=plot_title, x=0.5, y=0.97, font_size=24),
        xaxis=dict(title_text="Date", title_font_size=20, tickfont_size=16, showline=False, zeroline=True, zerolinewidth=1.5, zerolinecolor='darkgrey'),
        yaxis=dict(title_text=units, title_font_size=20, tickfont_size=16, showline=False, zeroline=True, zerolinewidth=1.5, zerolinecolor='darkgrey'),
        showlegend=True,
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02, bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1, font_size=15),
        annotations=None, template="plotly_white", margin=dict(t=50, r=150, b=50, l=50), height=550, hovermode='closest'
    )

    logger.info(f"Plot generated successfully for {site_id} [{actual_start_date_str} to {actual_end_date_str}]")
    return fig, None, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds, stats_dict
# --- END generate_plot_for_site ---


# --- Update Threshold in CSV Function ---
def update_threshold_in_csv(site_id: str, new_thresholds: Dict[str, Any], logger: logging.Logger) -> Tuple[bool, str]:
    """ Updates the threshold values for a given site_id in the CSV file. """
    global thresholds_df_global
    f = None; lock_acquired = False
    success_status = False; return_message = "An unknown issue occurred during threshold update."
    try:
        file_path_str = str(THRESHOLDS_CSV_PATH)
        if not THRESHOLDS_CSV_PATH.is_file(): raise FileNotFoundError(f"Threshold file '{file_path_str}' not found.")
        f = open(file_path_str, 'r+')
        lock_acquired = acquire_lock(f, fcntl.LOCK_EX if HAS_FCNTL else 0, logger)
        if not lock_acquired:
            msg = f"Failed to acquire write lock for {file_path_str}. Update aborted."
            logger.error(msg); success_status = False; return_message = "Error: Could not save thresholds (file busy)."
        else:
            temp_df = pd.read_csv(f); site_id_col = "SiteID"
            row_index = temp_df[temp_df[site_id_col].astype(str) == str(site_id)].index
            if not row_index.empty:
                idx = row_index[0]; logger.info(f"Updating thresholds SiteID {site_id} index {idx}...")
                temp_df.loc[idx, 'Over_Capacity'] = new_thresholds['max_val']; temp_df.loc[idx, 'Unusual_Spike'] = new_thresholds['spike_unusual']
                if 'Repeated_Days' not in temp_df.columns: temp_df['Repeated_Days'] = DEFAULT_REPEATED_DAYS
                temp_df.loc[idx, 'Repeated_Days'] = new_thresholds['repeated_days']
                f.seek(0); f.truncate(); temp_df.to_csv(f, index=False); f.flush()
                if hasattr(os, 'fsync'):
                    try: os.fsync(f.fileno()); logger.debug(f"os.fsync completed: {file_path_str}")
                    except OSError as fsync_err: logger.warning(f"os.fsync failed: {fsync_err}")
                logger.info(f"Thresholds updated ok: {file_path_str}"); success_status = True; return_message = f"Thresholds for Site ID {site_id} updated."
            else:
                logger.error(f"SiteID {site_id} not found in {file_path_str} for update."); success_status = False; return_message = f"Error: Site ID {site_id} not found."
    except FileNotFoundError as e: logger.error(e); success_status=False; return_message = "Error: Threshold file not found."
    except PermissionError: logger.error(f"Permission denied writing: '{THRESHOLDS_CSV_PATH}'."); success_status=False; return_message = "Error: Permission denied saving."
    except Exception as e: logger.error(f"Unexpected error updating CSV {site_id}: {e}", exc_info=True); success_status=False; return_message = "Unexpected error saving thresholds."
    finally:
        if lock_acquired and f: release_lock(f, logger)
        if f and not f.closed: f.close(); logger.debug("File closed in finally block.")
    if success_status and return_message.startswith("Thresholds for Site ID"):
         load_thresholds(THRESHOLDS_CSV_PATH, logger) # Reload only on successful write
    return success_status, return_message
# --- END update_threshold_in_csv ---