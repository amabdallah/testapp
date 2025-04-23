# -*- coding: utf-8 -*-
# --- Imports ---
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import traceback
import logging
import json
from typing import Dict, Any, Tuple, Optional, Sequence, List
import sys
from pathlib import Path

# --- Import functions and variables from threshold_manager ---
try:
    from threshold_manager import (
        get_site_thresholds,
        load_thresholds,
        thresholds_df_global,
        THRESHOLDS_CSV_PATH,
        DEFAULT_REPEATED_THRESHOLD
    )
except ImportError as e:
      print(f"FATAL ERROR: Could not import from threshold_manager.py. Error: {e}", file=sys.stderr)
      def get_site_thresholds(site_id, logger): logger.error("threshold_manager import failed"); return None
      def load_thresholds(path, logger): logger.error("threshold_manager import failed"); return None
      thresholds_df_global = None
      THRESHOLDS_CSV_PATH = "thresholds.csv"
      DEFAULT_REPEATED_THRESHOLD = 4


# --- Constants for Plotting ---
BUFFER_PERCENTAGE = 0.10; BUFFER_NUM_BANDS = 8
BUFFER_START_COLOR_RGBA = (128, 0, 128, 0.2); BUFFER_END_COLOR_RGBA = (128, 0, 128, 0.0)
REVIEW_BAR_COLORS = {'Reviewed': 'green', 'Raw': 'blue', 'Unknown': 'orange'}
COLOR_NORMAL = 'lightgray'; COLOR_QUALIFIED = 'red'; COLOR_UNKNOWN = 'orange'
REVIEW_BAR_Y_VALUE = 1; PROGRESS_BAR_TEXT_SIZE = 20

# Define legend groups
LG_DISCHARGE = 'discharge_group'
LG_FLAGS = 'flags_group'
LG_THRESHOLDS = 'thresholds_group'
LG_REVIEW_STATUS = 'review_status_bar_group' # Keep consistent group name for status items


# --- Utility Functions ---
def validate_date(date_str: Optional[str]) -> Optional[datetime]:
    # Use current time if available for comparison logic if needed
    current_time_info = "Current time is Wednesday, April 23, 2025 at 12:37:28 PM MDT." # From context
    # This utility doesn't use current time, but keeping context visible
    if not date_str: return None
    try: return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError: return None

def interpolate_color(c1, c2, fr):
    r1,g1,b1,a1=c1; r2,g2,b2,a2=c2; fr=max(0.0,min(1.0,fr)); r=int(r1+(r2-r1)*fr); g=int(g1+(g2-g1)*fr); b=int(b1+(b2-b1)*fr); a=a1+(a2-a1)*fr; r=max(0,min(255,r)); g=max(0,min(255,g)); b=max(0,min(255,b)); return f'rgba({r},{g},{b},{a:.4f})'

def add_gradient_buffer(fig, dates, mean_value, buffer, start_color, end_color, num_bands, logger):
    if buffer <= 0 or num_bands <= 0: logger.warning("Skipping gradient: invalid buffer/bands."); return
    if dates is None or len(dates) < 2: logger.warning("Skipping gradient: not enough points."); return
    n=len(dates); x_poly = list(dates) + list(dates)[::-1]
    for i in range(num_bands - 1, -1, -1):
        outer_f=(i+1)/num_bands; inner_f=i/num_bands; band_color=interpolate_color(start_color, end_color, outer_f)
        y_low_u=mean_value+inner_f*buffer; y_high_u=mean_value+outer_f*buffer
        # Make sure to pass legendgroup=None or specific group if buffer should be legend'd
        if np.isfinite(y_low_u) and np.isfinite(y_high_u): y_upper = [y_low_u]*n + [y_high_u]*n; fig.add_trace(go.Scatter(x=x_poly, y=y_upper, fill='toself', fillcolor=band_color, line=dict(width=0), hoverinfo="skip", showlegend=False, mode='lines', legendgroup=LG_THRESHOLDS)) # Assign group if needed
        y_high_l=mean_value-inner_f*buffer; y_low_l=mean_value-outer_f*buffer
        if np.isfinite(y_high_l) and np.isfinite(y_low_l): y_lower = [y_low_l]*n + [y_high_l]*n; fig.add_trace(go.Scatter(x=x_poly, y=y_lower, fill='toself', fillcolor=band_color, line=dict(width=0), hoverinfo="skip", showlegend=False, mode='lines', legendgroup=LG_THRESHOLDS)) # Assign group if needed

# --- Data Flagging Function ---
def apply_flagging(df: pd.DataFrame, thresholds: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    logger.info("Applying flagging logic to the DataFrame...")
    required_keys = ['min_val', 'max_val', 'spike_unusual', 'repeated_values_threshold']
    flag_cols = ['FLAG_LESS_THAN_Min._Value','FLAG_ZERO','FLAG_REPEATED','FLAG_GREATER_THAN_MaxValue','UNUSUAL_SPIKE','FLAG_BELOW_CAPACITY']
    def ensure_flag_columns(dframe):
        dframe['FLAGGED'] = False
        for col in flag_cols:
            if col not in dframe.columns: dframe[col] = False
        return dframe
    if not thresholds or not all(k in thresholds for k in required_keys):
        missing = [k for k in required_keys if not thresholds or k not in thresholds]; logger.error(f"Invalid thresholds for flagging. Missing: {missing}."); return ensure_flag_columns(df.copy())
    min_v = thresholds["min_val"]; max_v = thresholds["max_val"]; spike_thresh = thresholds["spike_unusual"]; rep_thresh = thresholds["repeated_values_threshold"]
    logger.info(f"Flagging parameters: Min={min_v}, Max={max_v}, Spike Threshold={spike_thresh}, Repeated Threshold={rep_thresh}")
    df_processed = df.copy()
    if 'DISCHARGE' not in df_processed.columns: logger.warning("'DISCHARGE' column missing."); return ensure_flag_columns(df_processed)
    df_processed['DISCHARGE'] = pd.to_numeric(df_processed['DISCHARGE'], errors='coerce')
    df_processed['FLAG_BELOW_CAPACITY'] = (df_processed['DISCHARGE'] < 0) & (df_processed['DISCHARGE'].notna()); df_processed['FLAG_ZERO'] = df_processed['DISCHARGE'] == 0
    df_processed['FLAG_LESS_THAN_Min._Value'] = (df_processed['DISCHARGE'] < min_v) & (df_processed['DISCHARGE'].notna()) & (df_processed['DISCHARGE'] > 0)
    df_processed['FLAG_GREATER_THAN_MaxValue'] = (df_processed['DISCHARGE'] > max_v) & (df_processed['DISCHARGE'].notna())
    if 'Date' in df_processed.columns: df_processed = df_processed.sort_values('Date')
    else: logger.warning("Date column missing. Rate of change/spike inaccurate.")
    df_processed['RATE_OF_CHANGE'] = df_processed['DISCHARGE'].diff().abs()
    df_processed['UNUSUAL_SPIKE'] = (df_processed['RATE_OF_CHANGE'] > spike_thresh) & (df_processed['RATE_OF_CHANGE'].notna())
    df_processed['FLAG_REPEATED'] = False
    non_zero_discharge = df_processed['DISCHARGE'].where((df_processed['DISCHARGE'] != 0) & df_processed['DISCHARGE'].notna())
    if not non_zero_discharge.isna().all():
        group_ids = (non_zero_discharge != non_zero_discharge.shift()).cumsum(); repeat_counts = non_zero_discharge.groupby(group_ids).transform('size')
        df_processed.loc[non_zero_discharge.notna(), 'FLAG_REPEATED'] = repeat_counts >= rep_thresh
    existing_flags = [c for c in flag_cols if c in df_processed.columns]
    if existing_flags: df_processed['FLAGGED'] = df_processed[existing_flags].any(axis=1)
    else: df_processed['FLAGGED'] = False
    num_flagged = df_processed['FLAGGED'].sum(); logger.info(f"Flagging complete. Total flagged: {num_flagged}")
    for flag in existing_flags:
          count = df_processed[flag].sum()
          if count > 0: logger.debug(f" - {flag}: {count} points")
    return df_processed


# --- Core Plot Generation Function ---
def generate_plot_for_site(
    site_id: str,
    start_date_str_requested: Optional[str],
    end_date_str_requested: Optional[str],
    is_reset: bool,
    logger: logging.Logger
) -> Tuple[Optional[go.Figure], Optional[str], str, str, str, str, Optional[Dict], Optional[Dict]]:
    """
    Generates a Plotly figure including review status progress bar, qualifiers,
    and color-coded discharge line based on qualifiers.
    """
    # Initialize return values
    station_name: str = "N/A"; units: str = 'Unknown Units'; site_thresholds: Optional[Dict] = None
    stats_dict: Optional[Dict] = None; actual_start_date_str: str = start_date_str_requested or ""
    actual_end_date_str: str = end_date_str_requested or ""
    fig: Optional[go.Figure] = None # Initialize fig to None
    df: Optional[pd.DataFrame] = None # Initialize df to None

    logger.info(f"--- Plot Generation Initiated (Inside generate_plot_for_site) ---")
    logger.info(f"Parameters: SiteID='{site_id}', Req Start='{start_date_str_requested}', Req End='{end_date_str_requested}', Reset={is_reset}")

    # --- Check Prerequisite: Thresholds Loaded ---
    if thresholds_df_global is None or thresholds_df_global.empty:
        logger.warning("Global thresholds not loaded or empty. Attempting load now...")
        loaded_df_in_context = load_thresholds(THRESHOLDS_CSV_PATH, logger)
        if loaded_df_in_context is None or loaded_df_in_context.empty:
              err = (f"CRITICAL ERROR: Threshold load attempt FAILED. File '{THRESHOLDS_CSV_PATH}' could not be loaded or is empty. Cannot process site data.")
              logger.error(f"{err} (SiteID: {site_id})"); logger.error(f"Global state after failed load attempt: {'None' if thresholds_df_global is None else ('Empty' if thresholds_df_global.empty else 'Not None/Empty?')}")
              return None, err, station_name, actual_start_date_str, actual_end_date_str, units, None, None
        else: logger.info("Successfully loaded thresholds within plot function context.")

    # 1. Get Site-Specific Thresholds
    logger.info(f"Retrieving thresholds for SiteID: {site_id}")
    site_thresholds = get_site_thresholds(site_id, logger)
    if site_thresholds is None:
        err = (f"Error: Threshold data missing/invalid for SiteID {site_id} in '{THRESHOLDS_CSV_PATH}'. Cannot generate plot.")
        logger.error(err); return None, err, station_name, actual_start_date_str, actual_end_date_str, units, None, None

    # 2. Fetch and Process API Data Block
    try:
        # --- Fetch ---
        api_end_date = datetime.now().strftime('%Y-%m-%d') if (is_reset or not validate_date(end_date_str_requested)) else end_date_str_requested
        api_start_date = "1900-01-01" if (is_reset or not validate_date(start_date_str_requested)) else start_date_str_requested
        logger.info(f"API Call Date Range: Start='{api_start_date}', End='{api_end_date}'")
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&start_date={api_start_date}&end_date={api_end_date}&f=json"
        logger.info(f"Fetching data from API: {api_url}")
        response = requests.get(api_url, timeout=45); response.raise_for_status(); data = response.json()
        logger.info("API data fetched successfully.")

        # --- Process ---
        metadata = { f: data.get(f, "N/A") for f in ["station_name", "units"] }; api_units = metadata.get('units', 'Unknown Units'); units = api_units if api_units and api_units != 'N/A' else units; logger.info(f"Units: '{units}'")
        api_station_name = metadata.get('station_name');
        if api_station_name and api_station_name != 'N/A': station_name = api_station_name; logger.info(f"Station name from API: '{station_name}'")
        else: logger.warning(f"API missing station name. Using default: '{station_name}'")

        if "data" not in data or not isinstance(data.get("data"), list) or not data["data"]:
            err = f"No time series 'data' in API response for SiteID {site_id}."; logger.warning(err); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None

        # --- Create DataFrame --- >> df assigned here <<
        df = pd.DataFrame(data["data"], columns=["date", "value"]); df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce'); df.dropna(subset=['Date'], inplace=True); df = df.sort_values('Date').reset_index(drop=True)
        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce'); logger.info(f"API data processed. Shape: {df.shape}.")

        if df.empty:
            err = f"No valid data points after processing API response for SiteID {site_id}."; logger.warning(err); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None

    # --- Handle API/Processing Errors ---
    except requests.exceptions.Timeout:
         err = f"API request timed out for SiteID {site_id}. URL: {api_url}"; logger.error(err); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None
    except requests.exceptions.RequestException as e:
        err = f"API Error fetching data for SiteID {site_id}: {e}. URL: {api_url}"; logger.error(err, exc_info=True); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None
    except (json.JSONDecodeError, ValueError) as e:
        err = (f"JSON Decode Error for SiteID {site_id}: {e}. Response snippet: '{response.text[:200]}...' URL: {api_url}"); logger.error(err); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None
    except Exception as e: # Catch unexpected errors during processing
        err = f"Unexpected error processing API data for SiteID {site_id}: {e}"; logger.error(err, exc_info=True); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None

    # --- If df was successfully created, proceed with further processing ---
    # All steps from here MUST have a valid 'df'

    # 4. Filter Dates
    min_data_dt, max_data_dt = df['Date'].min(), df['Date'].max(); start_req_dt_obj = validate_date(start_date_str_requested); end_req_dt_obj = validate_date(end_date_str_requested)
    start_dt_final = min_data_dt if (is_reset or not start_req_dt_obj) else max(start_req_dt_obj, min_data_dt); end_dt_final = max_data_dt if (is_reset or not end_req_dt_obj) else min(end_req_dt_obj, max_data_dt)
    if pd.isna(start_dt_final) or pd.isna(end_dt_final) or start_dt_final > end_dt_final: logger.warning("Date range invalid/no overlap. Using full data range."); start_dt_final, end_dt_final = min_data_dt, max_data_dt
    actual_start_date_str = start_dt_final.strftime('%Y-%m-%d') if pd.notna(start_dt_final) else ""; actual_end_date_str = end_dt_final.strftime('%Y-%m-%d') if pd.notna(end_dt_final) else ""
    logger.info(f"Final Plot Date Range: {actual_start_date_str} to {actual_end_date_str}")
    if pd.notna(start_dt_final) and pd.notna(end_dt_final):
        df_filtered = df.loc[(df['Date'] >= start_dt_final) & (df['Date'] <= end_dt_final)].copy().reset_index(drop=True)
        if df_filtered.empty:
             err = f"No data for site {site_id} in range [{actual_start_date_str} to {actual_end_date_str}]."; logger.warning(err); return None, err, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds, None
        df = df_filtered # Update df to the filtered version
    else:
        # This case should ideally not be reached if fallback logic works
         err = f"Could not determine valid final date range for filtering."; logger.error(err); return None, err, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds, None

    logger.info(f"Processing {len(df)} points after date filtering.")

    # --- Initialize Qualifiers Column ---
    df['Qualifiers'] = pd.Series(dtype='object')

    # --- Apply Specific Qualifiers for September ---
    now_dt = datetime.now(); qualifier_datetime_str = now_dt.strftime('%Y-%m-%d %H:%M')
    september_qualifier = {"Qualifier#1": "Qualifier-Edits Introduced", "Person": "Adel Abdallah", "DateTime": qualifier_datetime_str}
    september_mask = df['Date'].dt.month == 9; num_sept_rows = september_mask.sum()
    if num_sept_rows > 0:
        logger.info(f"Applying September qualifiers to {num_sept_rows} rows...")
        indices_to_update = df.loc[september_mask].index
        for index in indices_to_update: df.at[index, 'Qualifiers'] = september_qualifier.copy()
        logger.info("September qualifiers applied.")
    else: logger.info("No data points found in September. No qualifiers applied.")

    # 5. Add ReviewStatus column
    if not df.empty and 'Date' in df.columns: # Check df not empty again just in case
        latest_date = df['Date'].max()
        # Use current_time from context if needed here, e.g., for the one_year_ago logic
        # current_dt = datetime(2025, 4, 23, 12, 37, 28) # Example parsing from context
        if pd.notna(latest_date):
             # Define the one_year_ago based on the latest data point, as before.
             # The example text about "last year" is just explanatory text for the legend.
             one_year_ago = latest_date - pd.DateOffset(years=1)
             logger.info(f"Applying ReviewStatus: Dates > {one_year_ago.strftime('%Y-%m-%d')} marked as 'Raw'.")
             df['ReviewStatus'] = 'Reviewed'
             df.loc[df['Date'] > one_year_ago, 'ReviewStatus'] = 'Raw'
             logger.info(f"ReviewStatus Counts: Reviewed={(df['ReviewStatus'] == 'Reviewed').sum()}, Raw={(df['ReviewStatus'] == 'Raw').sum()}")
        else:
             logger.warning("Could not determine latest date. Defaulting ReviewStatus to 'Reviewed'.")
             df['ReviewStatus'] = 'Reviewed'
    else: logger.warning("DataFrame empty or 'Date' missing. Skipping ReviewStatus.");
    if not df.empty and 'ReviewStatus' not in df.columns: df['ReviewStatus'] = 'Unknown'

    # 6. Apply Flagging Logic
    df = apply_flagging(df, site_thresholds, logger)

    # --- Prepare Hover Text for Main Discharge Lines ---
    logger.debug("Preparing hover text for discharge lines...")
    def format_qualifier_hover(q_dict):
        if not isinstance(q_dict, dict) or not q_dict: return ""
        items = [f"&nbsp;&nbsp;<i>{k}:</i> {v}" for k, v in q_dict.items()]
        return "<br>--- Qualifiers ---<br>" + "<br>".join(items)
    base_hover_series = df.apply(lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><b>Discharge:</b> {row['DISCHARGE']:.2f} {units}" if pd.notna(row['DISCHARGE']) else f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><b>Discharge:</b> N/A", axis=1)
    qualifier_hover_series = df['Qualifiers'].apply(format_qualifier_hover)
    df['line_hovertext'] = base_hover_series + qualifier_hover_series + ""

    # --- Prepare Data for Segmented Line Plotting ---
    logger.debug("Preparing data segments for discharge lines...")
    df['HasQualifier'] = df['Qualifiers'].notna()
    mask_rev_noqual = (df['ReviewStatus'] == 'Reviewed') & (~df['HasQualifier']); mask_rev_qual = (df['ReviewStatus'] == 'Reviewed') & (df['HasQualifier'])
    mask_raw_noqual = (df['ReviewStatus'] == 'Raw') & (~df['HasQualifier']); mask_raw_qual = (df['ReviewStatus'] == 'Raw') & (df['HasQualifier'])
    mask_unknown = (df['ReviewStatus'] == 'Unknown')
    df['D_Rev_NoQual'] = df['DISCHARGE'].where(mask_rev_noqual); df['D_Rev_Qual'] = df['DISCHARGE'].where(mask_rev_qual)
    df['D_Raw_NoQual'] = df['DISCHARGE'].where(mask_raw_noqual); df['D_Raw_Qual'] = df['DISCHARGE'].where(mask_raw_qual)
    df['D_Unknown'] = df['DISCHARGE'].where(mask_unknown)

    # 7. Create Plot Figure
    logger.info("Creating Plotly figure...")
    plot_title = f"Data from {actual_start_date_str} to {actual_end_date_str}"
    fig = go.Figure()

    # --- Define Trace Order Here ---
    # 1. Discharge Lines
    # 2. Flagged Points
    # 3. Threshold Lines (Min, then Max)
    # 4. Dummy Header for Status
    # 5. Status Bars

    # --- 1. Add Segmented Discharge Line Traces ---
    logger.debug("Adding segmented discharge line traces...")
    # Normal (Reviewed)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['D_Rev_NoQual'], mode='lines', name='Discharge',
        line=dict(color=COLOR_NORMAL, width=1.5), connectgaps=False,
        legendgroup=LG_DISCHARGE, showlegend=True,
        hovertext=df['line_hovertext'], hoverinfo='text'
    ))
    # Normal (Raw - hidden, grouped with Normal)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['D_Raw_NoQual'], mode='lines', name='Discharge (Raw Normal - hidden)',
        line=dict(color=COLOR_NORMAL, width=1.5), connectgaps=False,
        legendgroup=LG_DISCHARGE, showlegend=False, # Hidden but part of the group
        hovertext=df['line_hovertext'], hoverinfo='text'
    ))
    # Qualified (Reviewed)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['D_Rev_Qual'], mode='lines', name='Discharge (Qualified)',
        line=dict(color=COLOR_QUALIFIED, width=1.5), connectgaps=False,
        legendgroup=LG_DISCHARGE, showlegend=True,
        hovertext=df['line_hovertext'], hoverinfo='text'
    ))
    # Qualified (Raw - hidden, grouped with Qualified)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['D_Raw_Qual'], mode='lines', name='Discharge (Raw Qualified - hidden)',
        line=dict(color=COLOR_QUALIFIED, width=1.5), connectgaps=False,
        legendgroup=LG_DISCHARGE, showlegend=False, # Hidden but part of the group
        hovertext=df['line_hovertext'], hoverinfo='text'
    ))
    logger.debug("Finished adding segmented discharge line traces.")


    # --- 2. Add Flagged Points as Markers ---
    min_val_thresh = site_thresholds.get("min_val", float('nan')); max_val_thresh = site_thresholds.get("max_val", float('nan')); spike_unusual_thresh = site_thresholds.get("spike_unusual", float('nan'))
    repeated_thresh_val = site_thresholds.get("repeated_values_threshold", DEFAULT_REPEATED_THRESHOLD)
    fmt_spike = f"{spike_unusual_thresh:.2f}" if pd.notna(spike_unusual_thresh) else "N/A"; fmt_max = f"{max_val_thresh:.2f}" if pd.notna(max_val_thresh) else "N/A"; fmt_min = f"{min_val_thresh:.2f}" if pd.notna(min_val_thresh) else "N/A"
    flag_plot_info = {
        'FLAG_BELOW_CAPACITY': ('red', 'Flag: Below Capacity (< 0)'),
        'FLAG_ZERO': ('blue', 'Flag: Zero Discharge'),
        'FLAG_REPEATED': ('green', f'Flag: Repeated Value (>{repeated_thresh_val} readings)'),
        'FLAG_GREATER_THAN_MaxValue': ('purple', f'Flag: Over Estimated Capacity ({fmt_max})'),
        'UNUSUAL_SPIKE': ('orange', f'Flag: Unusual Spike (Change > {fmt_spike})'),
        'FLAG_LESS_THAN_Min._Value': ('darkred', f'Flag: Below Min Threshold ({fmt_min}, >0)')
    }
    hover_tmpl_flags = (f'<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Discharge:</b> %{{y:.2f}} {units}<br><b>%{{meta}}</b><extra></extra>')
    logger.debug("Adding flagged points markers...")
    for flag_col, (color, label) in flag_plot_info.items():
        if flag_col in df.columns and df[flag_col].any():
            subset = df.loc[df[flag_col]];
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset['Date'], y=subset['DISCHARGE'], mode='markers',
                    marker=dict(color=color, size=7, symbol='circle'), name=label,
                    meta=label, hovertemplate=hover_tmpl_flags,
                    showlegend=True, legendgroup=LG_FLAGS # Assign to flags group
                ));
                logger.debug(f" - Added markers for '{flag_col}' ({len(subset)} points)")
            else: logger.debug(f" - No points to plot for flag '{flag_col}'.")

    # --- 3. Add Threshold Lines and Buffer ---
    if not df.empty:
        min_plot_dt, max_plot_dt = df["Date"].min(), df["Date"].max()
        if pd.notna(min_plot_dt) and pd.notna(max_plot_dt):
            plot_date_range = [min_plot_dt, max_plot_dt]
            # Min Threshold Line
            if pd.notna(min_val_thresh) and min_val_thresh > 0:
                fig.add_trace(go.Scatter(
                    x=plot_date_range, y=[min_val_thresh]*2, mode='lines',
                    name=f"Min Threshold ({fmt_min})",
                    line=dict(color="gray", dash="dash", width=1),
                    hoverinfo='skip', showlegend=True, legendgroup=LG_THRESHOLDS # Assign group
                ));
                logger.debug(f"Added Min Thr line at y={min_val_thresh}")
            # Max Threshold Line
            if pd.notna(max_val_thresh):
                max_threshold_label = f"Estimated Max Capacity ({fmt_max})"
                fig.add_trace(go.Scatter(
                    x=plot_date_range, y=[max_val_thresh]*2, mode='lines',
                    name=max_threshold_label,
                    line=dict(color="purple", dash="dash", width=1.5),
                    hoverinfo='skip', showlegend=True, legendgroup=LG_THRESHOLDS # Assign group
                ));
                logger.debug(f"Added Max Thr line (as '{max_threshold_label}') at y={max_val_thresh}")
                # Gradient Buffer (associated with Max Threshold visually)
                if max_val_thresh > 0:
                    buffer = max_val_thresh * BUFFER_PERCENTAGE
                    if len(df['Date']) >= 2:
                        logger.debug(f"Adding gradient buffer (Amount: {buffer:.2f})");
                        add_gradient_buffer(fig, df['Date'], max_val_thresh, buffer, BUFFER_START_COLOR_RGBA, BUFFER_END_COLOR_RGBA, BUFFER_NUM_BANDS, logger) # add_gradient_buffer handles showlegend=False


    # --- Prepare Data for Review Status Progress Bar ---
    # (Data preparation remains the same)
    logger.info("Preparing data for review status progress bar...")
    review_periods = [];
    if 'ReviewStatus' in df.columns and not df.empty: df_sorted = df.sort_values('Date'); df_sorted['status_group'] = (df_sorted['ReviewStatus'] != df_sorted['ReviewStatus'].shift()).cumsum(); grouped = df_sorted.groupby('status_group'); [(review_periods.append((group['Date'].min(), group['Date'].max(), group['ReviewStatus'].iloc[0]))) for name, group in grouped if not group.empty]; logger.info(f"Identified {len(review_periods)} review status periods.")
    else: logger.warning("Cannot generate review status bar: 'ReviewStatus' column missing or DataFrame empty.")
    reviewed_bar_x, reviewed_bar_widths, reviewed_bar_hover, reviewed_bar_text = [], [], [], []; raw_bar_x, raw_bar_widths, raw_bar_hover, raw_bar_text = [], [], [], []; unknown_bar_x, unknown_bar_widths, unknown_bar_hover, unknown_bar_text = [], [], [], []; period_boundaries = []
    if review_periods:
        period_boundaries.append(review_periods[0][0])
        for i, (start_time, end_time, status) in enumerate(review_periods):
            if end_time == start_time: mid_point = start_time; delta_seconds = 86400
            else: mid_point = start_time + (end_time - start_time) / 2; delta_seconds = (end_time - start_time).total_seconds()
            width_ms = max(86400000, delta_seconds * 1000); hover_text = (f"Status: {status}<br>From: {start_time.strftime('%Y-%m-%d')}<br>To: {end_time.strftime('%Y-%m-%d')}"); bar_label = status
            if status == 'Reviewed': reviewed_bar_x.append(mid_point); reviewed_bar_widths.append(width_ms); reviewed_bar_hover.append(hover_text); reviewed_bar_text.append(bar_label)
            elif status == 'Raw': raw_bar_x.append(mid_point); raw_bar_widths.append(width_ms); raw_bar_hover.append(hover_text); raw_bar_text.append(bar_label)
            else: unknown_bar_x.append(mid_point); unknown_bar_widths.append(width_ms); unknown_bar_hover.append(hover_text); unknown_bar_text.append(bar_label)
            if i < len(review_periods) - 1: period_boundaries.append(end_time)
            elif i == len(review_periods) - 1: period_boundaries.append(end_time)

    # --- 4. Add Dummy Trace for "Data Quality Status" Header ---
    logger.debug("Adding dummy trace for 'Data Quality Status' legend header")
    # *** Update dummy trace name with italic subtext ***
    status_header_text = "<b>Data Quality Status</b><br><i>Review status 'raw' is set for the last year</b> as an example</i>"
    fig.add_trace(go.Scatter(
        mode='markers', # Needs a mode, markers is fine
        x=[None], y=[None], # No actual data point
        marker=dict(opacity=0), # Make the marker invisible
        name=status_header_text, # <-- Use the multi-line HTML formatted name
        showlegend=True,
        legendgroup=LG_REVIEW_STATUS # Group with the bars that follow
    ))

    # --- 5. Add Review Status Bar Traces ---
    logger.debug("Adding review status bar traces with text...")
    if reviewed_bar_x:
        fig.add_trace(go.Bar(
            x=reviewed_bar_x, y=[REVIEW_BAR_Y_VALUE] * len(reviewed_bar_x), width=reviewed_bar_widths, base=0,
            marker_color=REVIEW_BAR_COLORS['Reviewed'], name='Reviewed', # Simplified name
            legendgroup=LG_REVIEW_STATUS, # Assign group
            hovertext=reviewed_bar_hover, hoverinfo='text', text=reviewed_bar_text,
            textposition='inside', insidetextanchor='middle', textfont=dict(color='white', size=PROGRESS_BAR_TEXT_SIZE),
            yaxis='y2', showlegend=True
        ))
        logger.debug(f" - Added 'Reviewed' bar trace ({len(reviewed_bar_x)} segments).")
    if raw_bar_x:
        fig.add_trace(go.Bar(
            x=raw_bar_x, y=[REVIEW_BAR_Y_VALUE] * len(raw_bar_x), width=raw_bar_widths, base=0,
            marker_color=REVIEW_BAR_COLORS['Raw'], name='Raw', # Simplified name
            legendgroup=LG_REVIEW_STATUS, # Assign group
            hovertext=raw_bar_hover, hoverinfo='text', text=raw_bar_text,
            textposition='inside', insidetextanchor='middle', textfont=dict(color='white', size=PROGRESS_BAR_TEXT_SIZE),
            yaxis='y2', showlegend=True
        ))
        logger.debug(f" - Added 'Raw' bar trace ({len(raw_bar_x)} segments).")
    if unknown_bar_x:
        fig.add_trace(go.Bar(
            x=unknown_bar_x, y=[REVIEW_BAR_Y_VALUE] * len(unknown_bar_x), width=unknown_bar_widths, base=0,
            marker_color=REVIEW_BAR_COLORS['Unknown'], name='Unknown', # Simplified name
            legendgroup=LG_REVIEW_STATUS, # Assign group
            hovertext=unknown_bar_hover, hoverinfo='text', text=unknown_bar_text,
            textposition='inside', insidetextanchor='middle', textfont=dict(color='black', size=PROGRESS_BAR_TEXT_SIZE),
            yaxis='y2', showlegend=True
        ))
        logger.debug(f" - Added 'Unknown Status' bar trace ({len(unknown_bar_x)} segments).")

    review_dividing_lines = [];
    if len(period_boundaries) > 2:
        internal_boundaries = period_boundaries[1:-1];
        logger.debug(f"Adding {len(internal_boundaries)} divider lines to review bar.");
        [review_dividing_lines.append(go.layout.Shape(type='line', xref='x', yref='y2 domain', x0=t, x1=t, y0=0, y1=1, line=dict(color='white', width=2), layer='above')) for t in internal_boundaries]

    # 8. Calculate Basic Statistics
    logger.debug("Calculating basic statistics...")
    stats_dict = None; discharge_numeric = df['DISCHARGE'].dropna()
    if not discharge_numeric.empty: mean_val, min_val, max_val = discharge_numeric.mean(), discharge_numeric.min(), discharge_numeric.max(); stats_dict = {"count": f"{discharge_numeric.count():,}", "mean": f"{mean_val:.2f}" if pd.notna(mean_val) else "N/A", "min": f"{min_val:.2f}" if pd.notna(min_val) else "N/A", "max": f"{max_val:.2f}" if pd.notna(max_val) else "N/A", "units": units}; logger.info(f"Calculated Stats: {stats_dict}")
    else: logger.warning("No numeric discharge data for stats."); stats_dict = {"count": "0", "mean": "N/A", "min": "N/A", "max": "N/A", "units": units}

    # 9. Finalize Figure Layout
    logger.debug("Finalizing plot layout...")
    progress_bar_height = 0.1; main_plot_bottom_margin = progress_bar_height + 0.1

    # --- Legend Configuration ---
    legend_main_title = "<b>Data Quality Flags</b><br><i>Qualified data is set for Sept as an example</i><br>" # Italic, no parens, extra <br> for space

    fig.update_layout(
        title=dict(text=plot_title, x=0.5, y=0.97, font_size=40),
        xaxis=dict(title_text="", title_font_size=32, tickfont_size=24, showline=False, zeroline=True, zerolinewidth=1.5, zerolinecolor='darkgrey'),
        yaxis=dict(title_text=f"{units}", title_font_size=32, tickfont_size=24, showline=False, zeroline=True, zerolinewidth=1.5, zerolinecolor='darkgrey', domain=[main_plot_bottom_margin, 1.0]),
        yaxis2=dict(domain=[0, progress_bar_height], visible=False, showticklabels=False, showgrid=False, zeroline=False, fixedrange=True),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.98, xanchor="left", x=1.01,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1,
            font_size=22, # Font size for legend items
            tracegroupgap=25, # Add space between legend groups (Thresholds -> Review Status)
            title=dict(
                text=legend_main_title, # Only the main title section here
                font=dict(size=24) # Font size for the overall legend title section
            ),
            # traceorder="normal" # Default, order based on trace addition
        ),
        template="plotly_white",
        margin=dict(t=80, r=350, b=60, l=100), # Kept increased right margin
        height=700,
        hovermode='closest',
        shapes=review_dividing_lines
    )


    logger.info(f"--- Plot generation successful for SiteID {site_id} ---")
    logger.info(f"Final Plot Range: {actual_start_date_str} to {actual_end_date_str}")
    logger.info(f"Station Name Used (from API): '{station_name}'")

    return fig, None, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds, stats_dict
# --- END plot_generator.py ---