# plot_generator.py
# -*- coding: utf-8 -*-
# --- Imports ---
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import traceback # Keep for potential future detailed error logging
import logging # Logger type hint comes from here
import json
from typing import Dict, Any, Tuple, Optional, Sequence, List # Keep this import
import sys # Added for importerror handling

# --- Import functions and variables from threshold_manager ---
# Assumes threshold_manager.py is in the same directory or Python path
try:
    from threshold_manager import (
        get_site_thresholds,
        load_thresholds,          # Now potentially called by generate_plot_for_site
        thresholds_df_global,     # Checked by generate_plot_for_site
        THRESHOLDS_CSV_PATH,      # Path needed for potential reload
        # DEFAULT_REPEATED_DAYS   # Removed, no longer used
    )
except ImportError as e:
     print(f"FATAL ERROR: Could not import from threshold_manager.py. Ensure it exists and is accessible. Error: {e}", file=sys.stderr)
     # Define placeholders to avoid immediate crashes if import fails,
     # but functionality will be broken.
     def get_site_thresholds(site_id, logger): logger.error("threshold_manager import failed"); return None
     def load_thresholds(path, logger): logger.error("threshold_manager import failed"); return None
     thresholds_df_global = None
     THRESHOLDS_CSV_PATH = "thresholds.csv" # Placeholder


# --- Constants for Plotting ---
BUFFER_PERCENTAGE = 0.10
BUFFER_NUM_BANDS = 8
BUFFER_START_COLOR_RGBA = (128, 0, 128, 0.2) # Purple, semi-transparent
BUFFER_END_COLOR_RGBA = (128, 0, 128, 0.0)   # Purple, fully transparent

# --- Utility Functions ---
# (validate_date, interpolate_color, add_gradient_buffer functions remain the same)
def validate_date(date_str: Optional[str]) -> Optional[datetime]:
    """Converts a YYYY-MM-DD string to a datetime object, returns None if invalid."""
    if not date_str: return None
    try: return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError: return None

def interpolate_color(c1, c2, fr):
    """Linearly interpolates between two RGBA colors."""
    r1,g1,b1,a1=c1; r2,g2,b2,a2=c2; fr=max(0.0,min(1.0,fr)); r=int(r1+(r2-r1)*fr); g=int(g1+(g2-g1)*fr); b=int(b1+(b2-b1)*fr); a=a1+(a2-a1)*fr; r=max(0,min(255,r)); g=max(0,min(255,g)); b=max(0,min(255,b)); return f'rgba({r},{g},{b},{a:.4f})'

def add_gradient_buffer(fig, dates, mean_value, buffer, start_color, end_color, num_bands, logger):
    """Adds a gradient shaded buffer area around a line on a Plotly figure."""
    if buffer <= 0 or num_bands <= 0: logger.warning("Skipping gradient: invalid buffer/bands."); return
    # Ensure dates is a suitable type (list, Series, array) and has length
    if dates is None or len(dates) < 2: logger.warning("Skipping gradient: not enough points."); return
    n=len(dates);
    x_poly = list(dates) + list(dates)[::-1] # Create polygon shape for x-axis
    for i in range(num_bands - 1, -1, -1):
        outer_f=(i+1)/num_bands; inner_f=i/num_bands; band_color=interpolate_color(start_color, end_color, outer_f)
        # Upper buffer band
        y_low_u=mean_value+inner_f*buffer; y_high_u=mean_value+outer_f*buffer
        if np.isfinite(y_low_u) and np.isfinite(y_high_u):
            y_upper = [y_low_u]*n + [y_high_u]*n # Polygon shape for y-axis
            fig.add_trace(go.Scatter(x=x_poly, y=y_upper, fill='toself', fillcolor=band_color, line=dict(width=0), hoverinfo="skip", showlegend=False, mode='lines'))
        # Lower buffer band
        y_high_l=mean_value-inner_f*buffer; y_low_l=mean_value-outer_f*buffer
        if np.isfinite(y_high_l) and np.isfinite(y_low_l):
            y_lower = [y_low_l]*n + [y_high_l]*n
            fig.add_trace(go.Scatter(x=x_poly, y=y_lower, fill='toself', fillcolor=band_color, line=dict(width=0), hoverinfo="skip", showlegend=False, mode='lines'))


# --- Data Flagging Function ---
def apply_flagging(df: pd.DataFrame, thresholds: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Applies various data quality flags to the DataFrame based on provided thresholds."""
    logger.info("Applying flagging logic to the DataFrame...")

    # Define expected threshold keys needed for flagging (Removed 'repeated_days')
    required_keys = ['min_val', 'max_val', 'spike_unusual']
    # Define standard flag column names (Removed 'FLAG_REPEATED')
    flag_cols = [
        'FLAG_LESS_THAN_Min._Value',
        'FLAG_ZERO',
        # 'FLAG_REPEATED', # Removed
        'FLAG_GREATER_THAN_MaxValue',
        'UNUSUAL_SPIKE',
        'FLAG_BELOW_CAPACITY'
    ]

    # Helper to ensure flag columns exist, even if no flags are triggered
    def ensure_flag_columns(dframe):
        dframe['FLAGGED'] = False # Master flag column
        for col in flag_cols:
            if col not in dframe.columns:
                 dframe[col] = False
        return dframe

    # Validate thresholds input
    if not thresholds or not all(k in thresholds for k in required_keys):
        missing = [k for k in required_keys if not thresholds or k not in thresholds]
        logger.error(f"Invalid or incomplete thresholds provided for flagging. Missing keys: {missing}. Cannot apply flags accurately.")
        # Return dataframe with empty flag columns
        return ensure_flag_columns(df.copy())

    # Extract threshold values
    min_v = thresholds["min_val"]
    max_v = thresholds["max_val"]
    spike_thresh = thresholds["spike_unusual"]

    logger.info(f"Flagging parameters: Min={min_v}, Max={max_v}, Spike Threshold={spike_thresh}")

    df_processed = df.copy() # Work on a copy

    if 'DISCHARGE' not in df_processed.columns:
        logger.warning("Cannot apply flags: 'DISCHARGE' column not found in DataFrame.")
        return ensure_flag_columns(df_processed)

    df_processed['DISCHARGE'] = pd.to_numeric(df_processed['DISCHARGE'], errors='coerce')

    # Apply flags
    df_processed['FLAG_BELOW_CAPACITY'] = (df_processed['DISCHARGE'] < 0) & (df_processed['DISCHARGE'].notna())
    df_processed['FLAG_ZERO'] = df_processed['DISCHARGE'] == 0
    df_processed['FLAG_LESS_THAN_Min._Value'] = (df_processed['DISCHARGE'] < min_v) & (df_processed['DISCHARGE'].notna()) & (df_processed['DISCHARGE'] > 0)
    df_processed['FLAG_GREATER_THAN_MaxValue'] = (df_processed['DISCHARGE'] > max_v) & (df_processed['DISCHARGE'].notna())

    # Calculate Rate of Change for spike detection
    if 'Date' in df_processed.columns:
         df_processed = df_processed.sort_values('Date')
    else:
         logger.warning("Cannot reliably calculate rate of change: 'Date' column missing.")

    df_processed['RATE_OF_CHANGE'] = df_processed['DISCHARGE'].diff().abs()
    df_processed['UNUSUAL_SPIKE'] = (df_processed['RATE_OF_CHANGE'] > spike_thresh) & (df_processed['RATE_OF_CHANGE'].notna())

    # Update the master 'FLAGGED' column
    existing_flags = [c for c in flag_cols if c in df_processed.columns]
    if existing_flags:
        df_processed['FLAGGED'] = df_processed[existing_flags].any(axis=1)
    else:
        df_processed['FLAGGED'] = False

    num_flagged = df_processed['FLAGGED'].sum()
    logger.info(f"Flagging complete. Total points flagged: {num_flagged}")
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
    logger: logging.Logger # Use the logger passed from the main app/caller
) -> Tuple[Optional[go.Figure], Optional[str], str, str, str, str, Optional[Dict], Optional[Dict]]:
    """
    Generates a Plotly figure for a given site ID and date range.
    Station name is derived ONLY from the API call.
    Flagging logic uses thresholds from threshold_manager.
    Will attempt to load thresholds if global variable is not set in current context.
    """
    # Initialize return values
    station_name: str = "N/A"; units: str = 'Unknown Units'; site_thresholds: Optional[Dict] = None
    stats_dict: Optional[Dict] = None; actual_start_date_str: str = start_date_str_requested or ""
    actual_end_date_str: str = end_date_str_requested or ""

    logger.info(f"--- Plot Generation Initiated (Inside generate_plot_for_site) ---")
    logger.info(f"Parameters: SiteID='{site_id}', Req Start='{start_date_str_requested}', Req End='{end_date_str_requested}', Reset={is_reset}")

    # --- Check Prerequisite: Thresholds Loaded (and attempt reload if needed) ---
    # Check the state of the global variable *before* attempting load
    if thresholds_df_global is None or thresholds_df_global.empty:
        logger.warning("Global thresholds not loaded or empty in current process context. Attempting load now...")
        # --- Attempt to load thresholds within this context ---
        # Capture the return value directly from the function call
        loaded_df_in_context = load_thresholds(THRESHOLDS_CSV_PATH, logger)
        # load_thresholds() also updates the global variable in threshold_manager internally
        # ----------------------------------------------------

        # *** Check the RETURN VALUE of the load attempt ***
        if loaded_df_in_context is None or loaded_df_in_context.empty:
             # If the function explicitly returned None or empty, the load failed.
             err = ("CRITICAL ERROR: Threshold load attempt FAILED (returned None or empty). Threshold configuration file "
                   f"('{THRESHOLDS_CSV_PATH}') could not be loaded or is empty. "
                   "Check logs for permission/format errors. Cannot process site data.")
             logger.error(f"{err} (Attempted plot generation for SiteID: {site_id})")
             # Log the state of the actual global variable just for debugging comparison
             logger.error(f"Global threshold state after failed load attempt: {'None' if thresholds_df_global is None else ('Empty' if thresholds_df_global.empty else 'Not None/Empty?')}")
             return None, err, station_name, actual_start_date_str, actual_end_date_str, units, None, None
        else:
             # The load function returned a valid DataFrame.
             # We can now trust that the global variable in threshold_manager
             # was also updated correctly by the load_thresholds function itself.
             logger.info("Successfully loaded thresholds within generate_plot_for_site context (based on return value).")
             # No need to check the global variable again here; proceed using it via get_site_thresholds.
    # --- End Prerequisite Check ---

    # If we reach here, thresholds_df_global should be populated.

    # 1. Get Site-Specific Thresholds
    logger.info(f"Retrieving thresholds for SiteID: {site_id}")
    # This function reads from the global variable, which should now be correctly populated
    site_thresholds = get_site_thresholds(site_id, logger)

    # Handle case where global loaded, but this specific site isn't in the file or has invalid data
    if site_thresholds is None:
        err = (f"Error: Threshold data required by the application is missing or invalid "
               f"for SiteID {site_id} within the loaded configuration file "
               f"('{THRESHOLDS_CSV_PATH}'). Check if SiteID exists and has valid values in the file. Cannot generate plot for this site.")
        logger.error(err)
        # Return default station_name="N/A", as we haven't called API yet
        return None, err, station_name, actual_start_date_str, actual_end_date_str, units, None, None

    # 2. Fetch Data from API
    api_end_date = datetime.now().strftime('%Y-%m-%d') if (is_reset or not validate_date(end_date_str_requested)) else end_date_str_requested
    api_start_date = "1900-01-01" if (is_reset or not validate_date(start_date_str_requested)) else start_date_str_requested
    logger.info(f"Determined API Call Date Range: Start='{api_start_date}', End='{api_end_date}'")
    try:
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&start_date={api_start_date}&end_date={api_end_date}&f=json"
        logger.info(f"Fetching data from API: {api_url}")
        response = requests.get(api_url, timeout=45); response.raise_for_status(); data = response.json()
        logger.info("API data fetched successfully.")
    except requests.exceptions.Timeout:
         err = f"API request timed out for SiteID {site_id}. URL: {api_url}"
         logger.error(err, exc_info=False)
         # Return thresholds found, even if API fails
         return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None
    except requests.exceptions.RequestException as e:
        err = f"API Error fetching data for SiteID {site_id}: {e}. URL: {api_url}"
        logger.error(err, exc_info=True)
        return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None
    except (json.JSONDecodeError, ValueError) as e:
        err = (f"JSON Decode Error for SiteID {site_id}: {e}. Response snippet: '{response.text[:200]}...' URL: {api_url}")
        logger.error(err)
        return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None

    # 3. Process API Data
    metadata = { f: data.get(f, "N/A") for f in ["station_name", "units"] }
    api_units = metadata.get('units', 'Unknown Units')
    units = api_units if api_units and api_units != 'N/A' else units # Use default if API units invalid
    logger.info(f"Units from API: '{api_units}' -> Using: '{units}'")

    # --- Get station name ONLY from API ---
    api_station_name = metadata.get('station_name')
    if api_station_name and api_station_name != 'N/A':
        station_name = api_station_name # Assign API name
        logger.info(f"Station name from API: '{station_name}'")
    else:
        logger.warning(f"API did not provide a valid station name. Using default: '{station_name}'")
    # ------------------------------------

    if "data" not in data or not isinstance(data.get("data"), list) or not data["data"]:
        err = f"No time series 'data' found in API response for SiteID {site_id}."
        logger.warning(err)
        return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None

    try:
        df = pd.DataFrame(data["data"], columns=["date", "value"]); df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce'); df.dropna(subset=['Date'], inplace=True); df = df.sort_values('Date').reset_index(drop=True)
        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')
        logger.info(f"API data processed into DataFrame. Shape: {df.shape}.")
    except Exception as e:
        err = f"Error processing API data into DataFrame for SiteID {site_id}: {e}"; logger.error(err, exc_info=True); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None
    if df.empty:
        err = f"No valid data points remained after processing API response for SiteID {site_id}."; logger.warning(err); return None, err, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds, None

    # 4. Filter Dates
    min_data_dt, max_data_dt = df['Date'].min(), df['Date'].max(); start_req_dt_obj = validate_date(start_date_str_requested); end_req_dt_obj = validate_date(end_date_str_requested)
    start_dt_final = min_data_dt if (is_reset or not start_req_dt_obj) else max(start_req_dt_obj, min_data_dt); end_dt_final = max_data_dt if (is_reset or not end_req_dt_obj) else min(end_req_dt_obj, max_data_dt)
    if pd.isna(start_dt_final) or pd.isna(end_dt_final) or start_dt_final > end_dt_final: logger.warning("Date range invalid/no overlap. Using full data range."); start_dt_final, end_dt_final = min_data_dt, max_data_dt
    actual_start_date_str = start_dt_final.strftime('%Y-%m-%d') if pd.notna(start_dt_final) else ""; actual_end_date_str = end_dt_final.strftime('%Y-%m-%d') if pd.notna(end_dt_final) else ""
    logger.info(f"Final Plot Date Range: {actual_start_date_str} to {actual_end_date_str}")
    if pd.notna(start_dt_final) and pd.notna(end_dt_final):
        df_filtered = df.loc[(df['Date'] >= start_dt_final) & (df['Date'] <= end_dt_final)].copy().reset_index(drop=True)
    else: df_filtered = pd.DataFrame(columns=df.columns) # Empty df with same columns
    if df_filtered.empty:
        err = f"No data for site {site_id} in range [{actual_start_date_str} to {actual_end_date_str}]."; logger.warning(err); return None, err, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds, None
    df = df_filtered; logger.info(f"Processing {len(df)} points after date filtering.")

    # 5. Add ReviewStatus column
    if not df.empty and 'Date' in df.columns:
        latest_date = df['Date'].max()
        if pd.notna(latest_date):
            one_year_ago = latest_date - pd.DateOffset(years=1)
            logger.info(f"Applying ReviewStatus: Dates > {one_year_ago.strftime('%Y-%m-%d')} marked as 'Raw'.")
            df['ReviewStatus'] = 'Reviewed'; df.loc[df['Date'] > one_year_ago, 'ReviewStatus'] = 'Raw'
            logger.info(f"ReviewStatus Counts: Reviewed={(df['ReviewStatus'] == 'Reviewed').sum()}, Raw={(df['ReviewStatus'] == 'Raw').sum()}")
        else: logger.warning("Could not determine latest date. Defaulting ReviewStatus to 'Reviewed'."); df['ReviewStatus'] = 'Reviewed'
    else: logger.warning("DataFrame empty or 'Date' missing. Skipping ReviewStatus.");
    if not df.empty and 'ReviewStatus' not in df.columns: df['ReviewStatus'] = 'Unknown'

    # 6. Apply Flagging Logic (uses updated apply_flagging function)
    df = apply_flagging(df, site_thresholds, logger)

    # 7. Create Plot Figure
    logger.info("Creating Plotly figure...")
    plot_title = f"Data from {actual_start_date_str} to {actual_end_date_str}"
    fig = go.Figure()

    # --- Add Discharge Line Traces (Reviewed vs. Raw) ---
    if 'ReviewStatus' in df.columns:
        df_reviewed = df[df['ReviewStatus'] == 'Reviewed']
        if not df_reviewed.empty: fig.add_trace(go.Scatter(x=df_reviewed['Date'], y=df_reviewed['DISCHARGE'], mode='lines', name='Discharge (Reviewed)', line=dict(color='lightgray', width=1.5), connectgaps=False, hoverinfo='skip', showlegend=True)); logger.debug(f"Added 'Reviewed' trace ({len(df_reviewed)} pts).")
        else: logger.info("No 'Reviewed' data points.")
        df_raw = df[df['ReviewStatus'] == 'Raw']
        if not df_raw.empty: fig.add_trace(go.Scatter(x=df_raw['Date'], y=df_raw['DISCHARGE'], mode='lines', name='Discharge (Raw)', line=dict(color='blue', width=1.5), connectgaps=False, hoverinfo='skip', showlegend=True)); logger.debug(f"Added 'Raw' trace ({len(df_raw)} pts).")
        else: logger.info("No 'Raw' data points.")
    else:
        logger.warning("ReviewStatus missing. Plotting single gray line.")
        if not df.empty: fig.add_trace(go.Scatter(x=df['Date'], y=df['DISCHARGE'], mode='lines', name='Discharge', line=dict(color='lightgray', width=1.5), connectgaps=False, hoverinfo='skip', showlegend=False)); logger.debug(f"Added single fallback trace ({len(df)} pts).")

    # --- Add Flagged Points as Markers ---
    min_val_thresh = site_thresholds.get("min_val", float('nan'))
    max_val_thresh = site_thresholds.get("max_val", float('nan'))
    spike_unusual_thresh = site_thresholds.get("spike_unusual", float('nan'))

    fmt_spike = f"{spike_unusual_thresh:.2f}" if pd.notna(spike_unusual_thresh) else "N/A"
    fmt_max = f"{max_val_thresh:.2f}" if pd.notna(max_val_thresh) else "N/A"
    fmt_min = f"{min_val_thresh:.2f}" if pd.notna(min_val_thresh) else "N/A"

    # Define marker properties (Removed 'FLAG_REPEATED')
    flag_plot_info = {
        'FLAG_BELOW_CAPACITY': ('red', 'Flag: Below Sensor Capacity (< 0)'),
        'FLAG_ZERO': ('blue', 'Flag: Zero Discharge'),
        'FLAG_GREATER_THAN_MaxValue': ('purple', f'Flag: Over Max Threshold ({fmt_max})'),
        'UNUSUAL_SPIKE': ('orange', f'Flag: Unusual Spike (Change > {fmt_spike})'),
        'FLAG_LESS_THAN_Min._Value': ('darkred', f'Flag: Below Min Threshold ({fmt_min}, >0)')
    }

    hover_tmpl = (f'<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Discharge:</b> %{{y:.2f}} {units}<br><b>%{{meta}}</b><extra></extra>')
#                                                                                        ^^  ^^   <-- Doubled braces
    logger.debug("Adding flagged points markers...")
    for flag_col, (color, label) in flag_plot_info.items():
        if flag_col in df.columns and df[flag_col].any():
            subset = df.loc[df[flag_col]]
            fig.add_trace(go.Scatter(
                x=subset['Date'], y=subset['DISCHARGE'], mode='markers',
                marker=dict(color=color, size=7, symbol='circle'),
                name=label, meta=label, hovertemplate=hover_tmpl, showlegend=True
            ))
            logger.debug(f" - Added markers for '{flag_col}' ({len(subset)} points, Color: {color})")

    # --- Add Threshold Lines and Buffer ---
    if not df.empty:
        min_plot_dt, max_plot_dt = df["Date"].min(), df["Date"].max()
        if pd.notna(min_plot_dt) and pd.notna(max_plot_dt):
            plot_date_range = [min_plot_dt, max_plot_dt]
            if pd.notna(min_val_thresh) and min_val_thresh > 0:
                fig.add_trace(go.Scatter(x=plot_date_range, y=[min_val_thresh]*2, mode='lines', name=f"Min Threshold ({fmt_min})", line=dict(color="gray", dash="dash", width=1), hoverinfo='skip', showlegend=True)); logger.debug(f"Added Min Threshold line at y={min_val_thresh}")
            if pd.notna(max_val_thresh):
                fig.add_trace(go.Scatter(x=plot_date_range, y=[max_val_thresh]*2, mode='lines', name=f"Max Threshold ({fmt_max})", line=dict(color="purple", dash="dash", width=1.5), hoverinfo='skip', showlegend=True)); logger.debug(f"Added Max Threshold line at y={max_val_thresh}")
                if max_val_thresh > 0:
                    buffer = max_val_thresh * BUFFER_PERCENTAGE
                    if len(df['Date']) >= 2: logger.debug(f"Adding gradient buffer (Amount: {buffer:.2f})"); add_gradient_buffer(fig, df['Date'], max_val_thresh, buffer, BUFFER_START_COLOR_RGBA, BUFFER_END_COLOR_RGBA, BUFFER_NUM_BANDS, logger)
                    else: logger.warning("Skipping gradient buffer: Not enough data points.")
        else: logger.warning("Could not determine plot date range for threshold lines.")

    # 8. Calculate Basic Statistics
    logger.debug("Calculating basic statistics...")
    stats_dict = None; discharge_numeric = df['DISCHARGE'].dropna()
    if not discharge_numeric.empty:
        mean_val, min_val, max_val = discharge_numeric.mean(), discharge_numeric.min(), discharge_numeric.max()
        stats_dict = {"count": f"{discharge_numeric.count():,}", "mean": f"{mean_val:.2f}" if pd.notna(mean_val) else "N/A", "min": f"{min_val:.2f}" if pd.notna(min_val) else "N/A", "max": f"{max_val:.2f}" if pd.notna(max_val) else "N/A", "units": units}; logger.info(f"Calculated Stats: {stats_dict}")
    else: logger.warning("No numeric discharge data for stats."); stats_dict = {"count": "0", "mean": "N/A", "min": "N/A", "max": "N/A", "units": units}

    # 9. Finalize Figure Layout
    logger.debug("Finalizing plot layout...")
    fig.update_layout(
        title=dict(text=plot_title, x=0.5, y=0.97, font_size=20),
        xaxis=dict(title_text="Date", title_font_size=16, tickfont_size=12, showline=False, zeroline=True, zerolinewidth=1.5, zerolinecolor='darkgrey'),
        yaxis=dict(title_text=f"Discharge ({units})", title_font_size=16, tickfont_size=12, showline=False, zeroline=True, zerolinewidth=1.5, zerolinecolor='darkgrey'),
        showlegend=True, legend=dict(yanchor="top", y=0.98, xanchor="left", x=1.01, bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1, font_size=11),
        template="plotly_white", margin=dict(t=60, r=200, b=50, l=80), height=550, hovermode='closest'
    )

    logger.info(f"--- Plot generation successful for SiteID {site_id} ---")
    logger.info(f"Final Plot Range: {actual_start_date_str} to {actual_end_date_str}")
    logger.info(f"Station Name Used (from API): '{station_name}'")

    # Return results including the station_name obtained ONLY from API
    return fig, None, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds, stats_dict
# --- END plot_generator.py ---