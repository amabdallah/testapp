# plot_table_generator.py
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
# (Assuming imports are correct as before)
try:
    from threshold_manager_dash import (
        get_site_thresholds,
        load_thresholds,
        thresholds_df_global, # Assuming this global might be used or can be removed if get_site_thresholds handles it
        THRESHOLDS_CSV_PATH,
        DEFAULT_REPEATED_THRESHOLD
    )
    print("INFO: Successfully imported from threshold_manager_dash.py in plot_table_generator.", file=sys.stderr)
except ImportError as e:
     print(f"FATAL ERROR: plot_table_generator could not import from threshold_manager_dash.py. Error: {e}", file=sys.stderr)
     # Define dummy functions/variables if import fails to allow script structure to remain
     def get_site_thresholds(site_id, logger): logger.error("threshold_manager_dash import failed"); return None
     def load_thresholds(path, logger): logger.error("threshold_manager_dash import failed"); return None
     thresholds_df_global = None
     THRESHOLDS_CSV_PATH = "thresholds_placeholder.csv"
     DEFAULT_REPEATED_THRESHOLD = 4


# --- Constants for Plotting ---
BUFFER_PERCENTAGE = 0.10
BUFFER_NUM_BANDS = 8
BUFFER_START_COLOR_RGBA = (128, 0, 128, 0.2) # Purple, semi-transparent start
BUFFER_END_COLOR_RGBA = (128, 0, 128, 0.0) # Purple, fully transparent end
REVIEW_BAR_COLORS = {'Reviewed': 'green', 'Raw': 'blue', 'Unknown': 'orange'}
COLOR_NORMAL = 'lightgray' # Color for normal (non-qualified) discharge line segments
COLOR_QUALIFIED = 'red' # Color for qualified discharge line segments
COLOR_UNKNOWN = 'orange' # Color for discharge where status is unknown
REVIEW_BAR_Y_VALUE = 1 # Y-value position for the review status bar
# PROGRESS_BAR_TEXT_SIZE = 10 # Original value
PROGRESS_BAR_TEXT_SIZE = 20 # Doubled text size for progress bar labels

# --- Utility Functions ---
def validate_date(date_str: Optional[str]) -> Optional[datetime]:
    """Validates if a string is in 'YYYY-MM-DD' format and returns a datetime object or None."""
    if not date_str: return None
    try: return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError: return None

def interpolate_color(c1: Tuple[int, int, int, float], c2: Tuple[int, int, int, float], fr: float) -> str:
    """Linearly interpolates between two RGBA colors."""
    r1,g1,b1,a1=c1; r2,g2,b2,a2=c2
    fr=max(0.0,min(1.0,fr)) # Clamp fraction between 0 and 1
    r=int(r1+(r2-r1)*fr); g=int(g1+(g2-g1)*fr); b=int(b1+(b2-b1)*fr)
    a=a1+(a2-a1)*fr
    # Ensure RGB values are within valid range
    r=max(0,min(255,r)); g=max(0,min(255,g)); b=max(0,min(255,b))
    return f'rgba({r},{g},{b},{a:.4f})'

def add_gradient_buffer(fig: go.Figure, dates: pd.Series, mean_value: float, buffer: float,
                        start_color: Tuple[int, int, int, float], end_color: Tuple[int, int, int, float],
                        num_bands: int, logger: logging.Logger):
    """Adds a gradient buffer around a line (typically max threshold) to the figure."""
    if buffer <= 0 or num_bands <= 0:
        logger.warning("Skipping gradient buffer: invalid buffer amount or number of bands.")
        return
    if dates is None or not isinstance(dates, pd.Series) or len(dates) < 2:
        logger.warning(f"Skipping gradient buffer: not enough date points ({len(dates) if dates is not None else 0}).")
        return

    try:
        date_list = list(dates)
        n = len(date_list)
        # Create polygon x-coordinates (go out along dates, then back)
        x_poly = date_list + date_list[::-1]

        # Add gradient bands from outermost to innermost
        for i in range(num_bands - 1, -1, -1):
            outer_fraction = (i + 1) / num_bands
            inner_fraction = i / num_bands

            # Interpolate color for this band
            band_color = interpolate_color(start_color, end_color, outer_fraction)

            # Calculate y-coordinates for the band polygon
            y_low_u = np.full(n, mean_value) + inner_fraction * buffer
            y_high_u = np.full(n, mean_value) + outer_fraction * buffer

            # Ensure calculated y-values are finite before plotting
            if np.all(np.isfinite(y_low_u)) and np.all(np.isfinite(y_high_u)):
                y_upper_poly = list(y_low_u) + list(y_high_u)[::-1]
                fig.add_trace(go.Scatter(
                    x=x_poly, y=y_upper_poly, fill='toself', fillcolor=band_color,
                    line=dict(width=0), # No border line for the fill polygons
                    hoverinfo="skip", # Don't show hover info for the buffer
                    showlegend=False,
                    mode='lines' # Ensures it's treated as a line for filling
                ))
            else:
                logger.debug(f"Skipping gradient buffer band {i}: contains non-finite y-values.")

    except Exception as e:
        logger.error(f"Error adding gradient buffer: {e}", exc_info=True)


# --- Data Flagging Function ---
def apply_flagging(df: pd.DataFrame, thresholds: Optional[Dict[str, Any]], logger: logging.Logger) -> pd.DataFrame:
    """Applies various data quality flags to the discharge data based on site thresholds."""
    logger.info("Applying flagging logic to the DataFrame...")

    df_processed = df.copy() # Work on a copy

    # --- Ensure Standard Flag Columns Exist ---
    flag_cols = ['FLAG_LESS_THAN_Min._Value','FLAG_ZERO','FLAG_REPEATED','FLAG_GREATER_THAN_MaxValue','UNUSUAL_SPIKE','FLAG_BELOW_CAPACITY']
    if 'FLAGGED' not in df_processed.columns: df_processed['FLAGGED'] = False
    for col in flag_cols:
        if col not in df_processed.columns: df_processed[col] = False
    # Ensure all flag columns start as False for this run
    df_processed[flag_cols + ['FLAGGED']] = False

    # --- Validate Thresholds ---
    if not thresholds or not isinstance(thresholds, dict):
        logger.error("Thresholds are missing or invalid. Cannot apply flags.")
        return df_processed # Return with flags initialized to False

    required_keys = ['min_val', 'max_val', 'spike_unusual', 'repeated_values_threshold']
    missing_keys = [k for k in required_keys if k not in thresholds or pd.isna(thresholds[k])]
    if missing_keys:
        logger.error(f"Threshold dictionary is missing required keys or has NaN values: {missing_keys}. Cannot apply all flags accurately.")
        # Proceed with available thresholds, but flags depending on missing ones won't be set.

    # --- Get Threshold Values (handle missing/NaN safely) ---
    min_v = thresholds.get("min_val", float('nan'))
    max_v = thresholds.get("max_val", float('nan'))
    spike_thresh = thresholds.get("spike_unusual", float('nan'))
    rep_thresh = thresholds.get("repeated_values_threshold", DEFAULT_REPEATED_THRESHOLD)
    if pd.isna(rep_thresh) or rep_thresh < 2: rep_thresh = DEFAULT_REPEATED_THRESHOLD # Ensure valid repeat threshold

    logger.info(f"Using Flagging parameters: Min={min_v}, Max={max_v}, Spike Threshold={spike_thresh}, Repeated Threshold={rep_thresh}")

    # --- Check for Discharge Column ---
    if 'DISCHARGE' not in df_processed.columns:
        logger.error("'DISCHARGE' column missing. Cannot perform flagging.")
        return df_processed

    # Ensure Discharge is numeric
    df_processed['DISCHARGE'] = pd.to_numeric(df_processed['DISCHARGE'], errors='coerce')
    discharge_col = df_processed['DISCHARGE'] # Convenience variable

    # --- Apply Individual Flags ---
    # Note: Flags are only applied where discharge is not NaN.
    not_na_mask = discharge_col.notna()

    df_processed.loc[not_na_mask, 'FLAG_BELOW_CAPACITY'] = (discharge_col[not_na_mask] < 0)
    df_processed.loc[not_na_mask, 'FLAG_ZERO'] = (discharge_col[not_na_mask] == 0)

    if pd.notna(min_v):
        df_processed.loc[not_na_mask, 'FLAG_LESS_THAN_Min._Value'] = (discharge_col[not_na_mask] > 0) & (discharge_col[not_na_mask] < min_v)
    else: logger.warning("Min threshold is NaN, FLAG_LESS_THAN_Min._Value cannot be applied.")

    if pd.notna(max_v):
        df_processed.loc[not_na_mask, 'FLAG_GREATER_THAN_MaxValue'] = (discharge_col[not_na_mask] > max_v)
    else: logger.warning("Max threshold is NaN, FLAG_GREATER_THAN_MaxValue cannot be applied.")

    # --- Rate of Change / Spike Flag ---
    # Sort by date temporarily for diff() calculation, preserve original index
    original_index = df_processed.index
    if 'Date' in df_processed.columns:
        df_processed = df_processed.sort_values('Date')
    else:
        logger.warning("Date column missing. Rate of change/spike and repeated value detection might be inaccurate.")

    # Calculate absolute difference from the previous non-NaN value
    df_processed['RATE_OF_CHANGE'] = discharge_col.diff().abs()

    if pd.notna(spike_thresh):
        # Apply flag where RoC exceeds threshold and RoC is not NaN
        df_processed['UNUSUAL_SPIKE'] = (df_processed['RATE_OF_CHANGE'] > spike_thresh) & df_processed['RATE_OF_CHANGE'].notna()
    else:
        logger.warning("Spike threshold is NaN, UNUSUAL_SPIKE flag will not be applied.")
        df_processed['UNUSUAL_SPIKE'] = False # Ensure column exists

    # --- Repeated Value Flag ---
    # Consider only non-zero, non-NaN values for repetition check
    discharge_for_repeat = discharge_col.where((discharge_col != 0) & not_na_mask)

    if not discharge_for_repeat.isna().all():
        # Identify groups of consecutive identical values
        group_ids = (discharge_for_repeat != discharge_for_repeat.shift()).cumsum()
        # Count size of each group
        repeat_counts = discharge_for_repeat.groupby(group_ids).transform('size')
        # Flag rows where the count meets or exceeds the threshold
        df_processed.loc[discharge_for_repeat.notna(), 'FLAG_REPEATED'] = repeat_counts >= rep_thresh
    else:
        logger.debug("No non-zero, non-NaN discharge values found for repeated value check.")
        df_processed['FLAG_REPEATED'] = False # Ensure column exists

    # Restore original sort order
    df_processed = df_processed.loc[original_index]

    # --- Aggregate Flag ---
    # Check which flag columns actually exist after processing
    existing_flags_in_df = [c for c in flag_cols if c in df_processed.columns]
    if existing_flags_in_df:
        # Set FLAGGED to True if *any* individual flag is True for that row
        df_processed['FLAGGED'] = df_processed[existing_flags_in_df].any(axis=1)
    else:
        df_processed['FLAGGED'] = False # Should not happen if columns are initialized

    # --- Log Summary ---
    num_flagged = df_processed['FLAGGED'].sum()
    logger.info(f"Flagging complete. Total points flagged: {num_flagged}")
    for flag in existing_flags_in_df:
        count = df_processed[flag].sum()
        if count > 0: logger.debug(f" - {flag}: {count} points")

    return df_processed


# --- Core Plot Generation Function (MODIFIED FOR DASH and thresholds_override) ---
def generate_plot_for_site(
    site_id: str,
    start_date_str_requested: Optional[str],
    end_date_str_requested: Optional[str],
    is_reset: bool,
    logger: logging.Logger,
    # --- NEW ARGUMENT ---
    thresholds_override: Optional[Dict[str, Any]] = None # Accept optional threshold dictionary
    # --- ---
) -> Tuple[Optional[go.Figure], Optional[pd.DataFrame], Optional[str], str, str, str, str, Optional[Dict], Optional[Dict]]:
    """
    Generates a Plotly time series figure for site data and returns the processed DataFrame.
    Accepts an optional threshold dictionary to override loaded thresholds.
    Legend items are individually togglable. Includes example September qualifiers.
    Hover shows data only for the closest point.

    Includes review status progress bar, qualifiers, flags, thresholds, and
    color-coded discharge line based on qualifiers in the time series plot.
    The processed DataFrame is returned for use in a Dash DataTable.

    Returns:
        Tuple containing:
        - Main Plotly Figure (Time Series Plot, or None on error)
        - Processed Pandas DataFrame (or None on error/empty data)
        - Error message string (or None if successful)
        - Station Name string
        - Actual Start Date string used for plot/table ('YYYY-MM-DD')
        - Actual End Date string used for plot/table ('YYYY-MM-DD')
        - Units string
        - Site Thresholds dictionary used (or None if error occurred before loading)
        - Basic Statistics dictionary (or None if no data)
    """
    # Initialize return values
    station_name: str = "N/A"
    units: str = 'Unknown Units'
    site_thresholds: Optional[Dict] = None # This will hold the thresholds actually used
    stats_dict: Optional[Dict] = None
    actual_start_date_str: str = start_date_str_requested or ""
    actual_end_date_str: str = end_date_str_requested or ""
    fig: Optional[go.Figure] = None # Initialize main plot fig to None
    df_processed: Optional[pd.DataFrame] = None # Initialize df for return
    error_message: Optional[str] = None # Initialize error message

    logger.info(f"--- Plot Generation & Data Prep Initiated (Dash version) ---")
    logger.info(f"Parameters: SiteID='{site_id}', Req Start='{start_date_str_requested}', Req End='{end_date_str_requested}', Reset={is_reset}")
    if thresholds_override is not None:
        logger.info("Threshold override provided.")

    # --- Check Prerequisite: Can we load thresholds if needed? ---
    # Check if threshold module failed to import (only relevant if override NOT provided)
    if thresholds_override is None and 'threshold_manager_dash' not in sys.modules:
          err = "CRITICAL ERROR: Threshold module (threshold_manager_dash) failed to import. Cannot proceed without thresholds."
          logger.error(err)
          return None, None, err, station_name, actual_start_date_str, actual_end_date_str, units, None, None
    # Also check if the file load previously failed (if relying on global state)
    if thresholds_override is None and (thresholds_df_global is None or thresholds_df_global.empty):
          if load_thresholds(THRESHOLDS_CSV_PATH, logger) is None: # Attempt load again just in case
               err = f"CRITICAL ERROR: Threshold load FAILED. File '{THRESHOLDS_CSV_PATH}' could not be loaded or is empty. Cannot proceed."
               logger.error(err)
               return None, None, err, station_name, actual_start_date_str, actual_end_date_str, units, None, None

    # 1. Determine Thresholds to Use
    # This logic replaces the previous block #1
    if thresholds_override is not None and isinstance(thresholds_override, dict):
        logger.info(f"Using provided threshold override for site {site_id}.")
        site_thresholds = thresholds_override
        # Optional: Validate the override dictionary structure here if needed
        # Example validation:
        # required_keys = ['min_val', 'max_val', 'spike_unusual', 'repeated_values_threshold']
        # if not all(k in site_thresholds for k in required_keys):
        #     logger.warning("Provided threshold override is missing required keys.")
        #     # Decide how to handle: use defaults, return error, etc.
    else:
        logger.info(f"Loading thresholds via get_site_thresholds for site {site_id}.")
        site_thresholds = get_site_thresholds(site_id, logger) # Call the function to get thresholds

    # Check if thresholds were successfully loaded or provided
    if site_thresholds is None or not isinstance(site_thresholds, dict):
        err = (f"Error: Thresholds could not be loaded or provided in a valid format for SiteID {site_id}. Cannot generate plot or apply flags correctly.")
        logger.error(err)
        # Return None for thresholds if they are invalid/missing
        return None, None, err, station_name, actual_start_date_str, actual_end_date_str, units, None, None
    elif not site_thresholds: # Check if the dictionary is empty (might indicate site not found in thresholds file)
         logger.warning(f"No specific thresholds found for SiteID {site_id}. Proceeding without threshold-based flagging/plotting.")
         # Allow processing to continue, but threshold features will be missing
         # Ensure site_thresholds is an empty dict for downstream code
         site_thresholds = {}
    else:
        logger.info(f"Thresholds determined for SiteID {site_id}: {site_thresholds}")


    # 2. Fetch and Process API Data Block
    df = None # Explicitly initialize df here
    try:
        # --- Define API Date Range ---
        # MODIFICATION: Use current date from datetime for consistency
        current_date_dt = datetime.now()
        current_date_str = current_date_dt.strftime('%Y-%m-%d')

        api_end_dt = current_date_dt # Default to now
        v_end_date = validate_date(end_date_str_requested)
        if not is_reset and v_end_date:
             api_end_dt = v_end_date # Use requested end date if valid and not reset

        # Ensure API end date is not in the future
        api_end_dt = min(api_end_dt, current_date_dt)
        api_end_date_str = api_end_dt.strftime('%Y-%m-%d')

        api_start_dt = datetime(1900, 1, 1) # Default very old start
        v_start_date = validate_date(start_date_str_requested)
        if not is_reset and v_start_date:
            api_start_dt = v_start_date # Use requested start date if valid and not reset
        api_start_date_str = api_start_dt.strftime('%Y-%m-%d')

        # Swap if start > end after validation/defaults
        if api_start_dt > api_end_dt:
             logger.warning(f"API date range invalid ({api_start_date_str} > {api_end_date_str}). Swapping dates.")
             api_start_date_str, api_end_date_str = api_end_date_str, api_start_date_str
             # Also swap the datetime objects for consistency in later logic if needed
             api_start_dt, api_end_dt = api_end_dt, api_start_dt


        logger.info(f"API Call Date Range: Start='{api_start_date_str}', End='{api_end_date_str}'")

        # --- Fetch ---
        # Updated on 2025-04-28: Assuming the API endpoint might change, check documentation. Keeping original for now.
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&start_date={api_start_date_str}&end_date={api_end_date_str}&f=json"
        logger.info(f"Fetching data from API: {api_url}")
        response = requests.get(api_url, timeout=45) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        logger.info("API data fetched successfully.")

        # --- Process Metadata ---
        metadata = { f: data.get(f, "N/A") for f in ["station_name", "units"] }
        api_units = metadata.get('units', 'Unknown Units');
        units = api_units if api_units and api_units != 'N/A' else 'Unknown Units';
        logger.info(f"Units from API: '{units}'")

        api_station_name = metadata.get('station_name');
        if api_station_name and api_station_name != 'N/A': station_name = api_station_name; logger.info(f"Station name from API: '{station_name}'")
        else: logger.warning(f"API response missing valid station name. Using default: '{station_name}'")

        # --- Process Time Series Data ---
        if "data" not in data or not isinstance(data.get("data"), list) or not data["data"]:
            err = f"No time series 'data' list found or list is empty in API response for SiteID {site_id}."
            logger.warning(err)
            # Return thresholds found, but no plot/data
            return None, None, err, station_name, start_date_str_requested or "", end_date_str_requested or "", units, site_thresholds, None

        # --- Create DataFrame ---
        df = pd.DataFrame(data["data"], columns=["date", "value"])
        df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True) # Remove rows where date conversion failed
        if df.empty:
             err = f"DataFrame became empty after date processing for SiteID {site_id}."
             logger.warning(err)
             return None, None, err, station_name, start_date_str_requested or "", end_date_str_requested or "", units, site_thresholds, None

        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
        logger.info(f"API data processed into DataFrame. Shape: {df.shape}. Date range: {df['Date'].min()} to {df['Date'].max()}")

    # --- Handle API/Processing Errors ---
    except requests.exceptions.Timeout:
        err = f"API request timed out (>45s) for SiteID {site_id}."; logger.error(f"{err} URL: {api_url}"); return None, None, err, station_name, start_date_str_requested or "", end_date_str_requested or "", units, site_thresholds, None
    except requests.exceptions.HTTPError as e:
         err = f"API HTTP Error for SiteID {site_id}: {e}. Status Code: {e.response.status_code}."; logger.error(f"{err} URL: {api_url}"); return None, None, err, station_name, start_date_str_requested or "", end_date_str_requested or "", units, site_thresholds, None
    except requests.exceptions.RequestException as e:
        err = f"API Request Error fetching data for SiteID {site_id}: {e}."; logger.error(f"{err} URL: {api_url}", exc_info=True); return None, None, err, station_name, start_date_str_requested or "", end_date_str_requested or "", units, site_thresholds, None
    except (json.JSONDecodeError, ValueError) as e:
        response_text_snippet = response.text[:200] if 'response' in locals() else "N/A"
        err = (f"JSON Decode or Value Error processing data for SiteID {site_id}: {e}. Response snippet: '{response_text_snippet}...'."); logger.error(f"{err} URL: {api_url}"); return None, None, err, station_name, start_date_str_requested or "", end_date_str_requested or "", units, site_thresholds, None
    except Exception as e:
        err = f"Unexpected error processing API data for SiteID {site_id}: {e}"; logger.error(err, exc_info=True); return None, None, err, station_name, start_date_str_requested or "", end_date_str_requested or "", units, site_thresholds, None

    # --- If df was successfully created and is not empty, proceed ---
    if df is None or df.empty:
        logger.error("DataFrame is unexpectedly None or empty before date filtering step.")
        err = error_message or "Failed to create valid DataFrame from API." # Use existing error if available
        return None, None, err, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds, None


    # 4. Filter Dates based on Request (or use full range if reset/invalid)
    min_data_dt, max_data_dt = df['Date'].min(), df['Date'].max()
    # Use validated date objects if they exist from API date setup
    start_req_dt_obj = v_start_date if 'v_start_date' in locals() and v_start_date is not None else validate_date(start_date_str_requested)
    end_req_dt_obj = v_end_date if 'v_end_date' in locals() and v_end_date is not None else validate_date(end_date_str_requested)

    # Determine the final start date for the plot/table
    if not is_reset and start_req_dt_obj and start_req_dt_obj >= min_data_dt:
        start_dt_final = start_req_dt_obj
    else: # Use data minimum if reset, request is invalid, or before data start
        start_dt_final = min_data_dt

    # Determine the final end date for the plot/table
    if not is_reset and end_req_dt_obj and end_req_dt_obj <= max_data_dt:
        end_dt_final = end_req_dt_obj
    else: # Use data maximum if reset, request is invalid, or after data end
        end_dt_final = max_data_dt

    # Final check: ensure start <= end (might happen if requests were weird)
    if pd.isna(start_dt_final) or pd.isna(end_dt_final) or start_dt_final > end_dt_final:
        logger.warning(f"Determined final date range invalid ({start_dt_final} to {end_dt_final}). Defaulting to full data range: {min_data_dt} to {max_data_dt}.")
        start_dt_final, end_dt_final = min_data_dt, max_data_dt

    # Format final dates for return, handle potential NaT
    actual_start_date_str = start_dt_final.strftime('%Y-%m-%d') if pd.notna(start_dt_final) else ""
    actual_end_date_str = end_dt_final.strftime('%Y-%m-%d') if pd.notna(end_dt_final) else ""
    logger.info(f"Final Plot/Table Date Range determined: {actual_start_date_str} to {actual_end_date_str}")

    # Filter the DataFrame to the final determined range
    if pd.notna(start_dt_final) and pd.notna(end_dt_final):
        df_filtered = df.loc[(df['Date'] >= start_dt_final) & (df['Date'] <= end_dt_final)].copy()
        if df_filtered.empty:
            err = f"No data available for site {site_id} in the final selected range [{actual_start_date_str} to {actual_end_date_str}]."
            logger.warning(err)
            # Return thresholds found, but no plot/data for this range
            return None, None, err, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds, None
        df = df_filtered.reset_index(drop=True) # Use filtered data from now on
        logger.info(f"DataFrame filtered to {len(df)} points for the range {actual_start_date_str} to {actual_end_date_str}.")
    else:
        # This case should ideally not be reached due to the fallback above
        err = f"Could not determine a valid final date range for filtering. Using original unfiltered data."
        logger.error(err)
        df = df.reset_index(drop=True) # Use original if filtering failed

    df_processed = df # Assign the final DataFrame to df_processed


    # --- Initialize Qualifiers Column ---
    if 'Qualifiers' not in df_processed.columns:
        logger.debug("Initializing 'Qualifiers' column as object type.")
        df_processed['Qualifiers'] = pd.Series(dtype='object')
    else:
        logger.debug("'Qualifiers' column already exists. Ensuring dtype is object.")
        df_processed['Qualifiers'] = df_processed['Qualifiers'].astype('object')


    # --- Apply Specific Qualifiers for September ---
    if 'Date' in df_processed.columns:
        try:
            # Use datetime.now() for the qualifier timestamp
            # Using Salt Lake City timezone as per context (MDT/MST) - requires 'pytz'
            # If pytz is not available, fall back to naive local time
            try:
                import pytz
                slc_tz = pytz.timezone('America/Denver')
                now_dt_slc = datetime.now(slc_tz)
                qualifier_datetime_str = now_dt_slc.strftime('%Y-%m-%d %H:%M %Z%z') # Include timezone info
            except ImportError:
                logger.warning("pytz library not found. Using naive local time for September qualifier timestamp.")
                now_dt = datetime.now()
                qualifier_datetime_str = now_dt.strftime('%Y-%m-%d %H:%M') # Naive local time

            september_qualifier = {
                "Qualifier#1": "Qualifier-Edits Introduced",
                "Person": "Adel Abdallah", # Or appropriate name/system identifier
                "DateTime": qualifier_datetime_str
            }
            september_mask = df_processed['Date'].dt.month == 9
            num_sept_rows = september_mask.sum()

            if num_sept_rows > 0:
                logger.info(f"Applying September qualifiers to {num_sept_rows} rows...")
                indices_to_update = df_processed.loc[september_mask].index
                # Use .at with index to assign the dictionary; handles single cell assignment correctly
                for index in indices_to_update:
                     # current_qual = df_processed.at[index, 'Qualifiers'] # Get current if merging needed
                     # Decide how to handle existing qualifiers: overwrite or merge? Overwriting for now.
                     # Use .at for setting single value by label - FIX for "Incompatible indexer" error
                     df_processed.at[index, 'Qualifiers'] = september_qualifier.copy() # Assign copy using .at
                logger.info("September qualifiers applied.")
            else:
                logger.info("No data points found in September within the current date range. No example qualifiers applied.")
        except AttributeError as e:
             logger.error(f"Error accessing '.dt' accessor, likely 'Date' column is not datetime type: {e}")
        except Exception as e:
             # Log the specific error from the traceback provided by user
             logger.error(f"An unexpected error occurred during September qualifier application: {e}", exc_info=True)
             # Add specific logging for the known previous error if it occurs again
             if "Incompatible indexer with Series" in str(e):
                 logger.error("Specifically caught the 'Incompatible indexer' error during September qualifier assignment.")
    else:
        logger.warning("Cannot apply September qualifiers because 'Date' column is missing in df_processed.")


    # 5. Add ReviewStatus column (Example Logic: Last Year is 'Raw')
    if 'Date' in df_processed.columns and not df_processed.empty:
        latest_date = df_processed['Date'].max()
        if pd.notna(latest_date):
             # Use DateOffset for reliable year subtraction
             one_year_ago = latest_date - pd.DateOffset(years=1)
             logger.info(f"Applying ReviewStatus: Dates > {one_year_ago.strftime('%Y-%m-%d')} marked as 'Raw', others 'Reviewed'.")
             df_processed['ReviewStatus'] = 'Reviewed' # Default
             df_processed.loc[df_processed['Date'] > one_year_ago, 'ReviewStatus'] = 'Raw'
             logger.info(f"ReviewStatus Counts: Reviewed={(df_processed['ReviewStatus'] == 'Reviewed').sum()}, Raw={(df_processed['ReviewStatus'] == 'Raw').sum()}")
        else:
             logger.warning("Could not determine latest date. Defaulting all ReviewStatus to 'Unknown'.")
             df_processed['ReviewStatus'] = 'Unknown'
    else:
       logger.warning("DataFrame empty or 'Date' column missing after filtering. Defaulting ReviewStatus to 'Unknown'.")
       if 'ReviewStatus' not in df_processed.columns:
            df_processed['ReviewStatus'] = pd.Series(['Unknown'] * len(df_processed), index=df_processed.index)


    # 6. Apply Flagging Logic using the determined thresholds
    # Ensure site_thresholds is a dict, even if empty
    df_processed = apply_flagging(df_processed, site_thresholds if site_thresholds else {}, logger)


    # --- Prepare Hover Text for Main Discharge Lines ---
    logger.debug("Preparing hover text for discharge lines...")
    def format_qualifier_hover(q_dict):
        """Formats dictionary qualifiers for hover text."""
        if not isinstance(q_dict, dict) or not q_dict:
            return "" # Return empty string if no qualifier or not a dict
        try:
            # Format each key-value pair
            items = [f"&nbsp;&nbsp;<i>{k}:</i> {v}" for k, v in q_dict.items()]
            return "<br>--- Qualifiers ---<br>" + "<br>".join(items)
        except Exception as e:
             logger.error(f"Error formatting qualifier hover text for dict {q_dict}: {e}")
             return "<br>--- Qualifiers (Error) ---"

    # Create hover text, handling potential NaNs and formatting
    base_hover_series = df_processed.apply(lambda row:
        f"<b>Date:</b> {row.get('Date', pd.NaT).strftime('%Y-%m-%d')}<br>" +
        (f"<b>Discharge:</b> {row.get('DISCHARGE', float('nan')):.2f} {units}" if pd.notna(row.get('DISCHARGE')) else "<b>Discharge:</b> N/A"),
        axis=1
    )
    # Apply formatting function to the Qualifiers column (if it exists)
    qualifier_hover_series = df_processed.get('Qualifiers', pd.Series(dtype='object')).apply(format_qualifier_hover)

    # Combine base hover text with qualifier text and add <extra> tag
    df_processed['line_hovertext'] = base_hover_series + qualifier_hover_series + "<extra></extra>" # <extra> hides trace name


    # --- Prepare Data for Segmented Line Plotting ---
    logger.debug("Preparing data segments for discharge lines based on status and qualifiers...")
    # Check if 'Qualifiers' column exists before using it
    if 'Qualifiers' in df_processed.columns:
         df_processed['HasQualifier'] = df_processed['Qualifiers'].apply(lambda x: isinstance(x, dict) and bool(x))
    else:
         df_processed['HasQualifier'] = False # No qualifiers if column doesn't exist

    df_processed['ReviewStatus'] = df_processed.get('ReviewStatus', 'Unknown') # Ensure column exists

    # Create masks for different segments
    mask_rev_noqual = (df_processed['ReviewStatus'] == 'Reviewed') & (~df_processed['HasQualifier'])
    mask_rev_qual = (df_processed['ReviewStatus'] == 'Reviewed') & (df_processed['HasQualifier'])
    mask_raw_noqual = (df_processed['ReviewStatus'] == 'Raw') & (~df_processed['HasQualifier'])
    mask_raw_qual = (df_processed['ReviewStatus'] == 'Raw') & (df_processed['HasQualifier'])
    mask_unknown = (df_processed['ReviewStatus'] == 'Unknown') # Assuming we plot Unknown status

    # Create separate discharge columns for each segment using .where()
    df_processed['D_Rev_NoQual'] = df_processed['DISCHARGE'].where(mask_rev_noqual)
    df_processed['D_Rev_Qual'] = df_processed['DISCHARGE'].where(mask_rev_qual)
    df_processed['D_Raw_NoQual'] = df_processed['DISCHARGE'].where(mask_raw_noqual)
    df_processed['D_Raw_Qual'] = df_processed['DISCHARGE'].where(mask_raw_qual)
    df_processed['D_Unknown'] = df_processed['DISCHARGE'].where(mask_unknown) # Segment for Unknown status


    # 7. Create Plot Figure
    logger.info("Creating main Plotly time series figure...")
    plot_title = f"Data for {station_name} ({site_id}) | {actual_start_date_str} to {actual_end_date_str}"
    fig = go.Figure() # Initialize the figure object


    # --- Add Traces (using df_processed) ---
    # --- 1. Add Segmented Discharge Line Traces ---
    # NOTE: Using hoverinfo='text' relies on the pre-formatted 'line_hovertext' column.
    logger.debug("Adding segmented discharge line traces...")
    # Trace for Reviewed, No Qualifier (Primary "Discharge" legend entry)
    fig.add_trace(go.Scatter(
        x=df_processed['Date'], y=df_processed['D_Rev_NoQual'], mode='lines', name='Discharge',
        line=dict(color=COLOR_NORMAL, width=1.5), connectgaps=False,
        showlegend=True, legendgroup="discharge", # Group for legend toggling
        hovertext=df_processed['line_hovertext'].where(mask_rev_noqual), hoverinfo='text'
    ))
    # Trace for Raw, No Qualifier (Same appearance, linked to main legend entry)
    fig.add_trace(go.Scatter(
        x=df_processed['Date'], y=df_processed['D_Raw_NoQual'], mode='lines', name='Discharge (Raw)',
        line=dict(color=COLOR_NORMAL, width=1.5), connectgaps=False,
        showlegend=False, legendgroup="discharge", # Hide but link to main group
        hovertext=df_processed['line_hovertext'].where(mask_raw_noqual), hoverinfo='text'
    ))
    # Trace for Reviewed, Qualified
    fig.add_trace(go.Scatter(
        x=df_processed['Date'], y=df_processed['D_Rev_Qual'], mode='lines', name='Discharge (Qualified)',
        line=dict(color=COLOR_QUALIFIED, width=1.5), connectgaps=False,
        showlegend=True, legendgroup="discharge_qual",
        hovertext=df_processed['line_hovertext'].where(mask_rev_qual), hoverinfo='text'
    ))
    # Trace for Raw, Qualified
    fig.add_trace(go.Scatter(
        x=df_processed['Date'], y=df_processed['D_Raw_Qual'], mode='lines', name='Discharge (Raw Qualified)',
        line=dict(color=COLOR_QUALIFIED, width=1.5), connectgaps=False,
        showlegend=False, legendgroup="discharge_qual",
        hovertext=df_processed['line_hovertext'].where(mask_raw_qual), hoverinfo='text'
    ))
    # Trace for Unknown Status
    fig.add_trace(go.Scatter(
         x=df_processed['Date'], y=df_processed['D_Unknown'], mode='lines', name='Discharge (Unknown Status)',
         line=dict(color=COLOR_UNKNOWN, width=1.5), connectgaps=False,
         showlegend=True, legendgroup="discharge_unknown",
         hovertext=df_processed['line_hovertext'].where(mask_unknown), hoverinfo='text'
      ))
    logger.debug("Finished adding segmented discharge line traces.")


    # --- 2. Add Flagged Points as Markers (with customdata) ---
    # Get threshold values again for flag labels, handle NaN
    min_val_thresh = site_thresholds.get("min_val", float('nan'))
    max_val_thresh = site_thresholds.get("max_val", float('nan'))
    spike_unusual_thresh = site_thresholds.get("spike_unusual", float('nan'))
    repeated_thresh_val = site_thresholds.get("repeated_values_threshold", DEFAULT_REPEATED_THRESHOLD)
    if pd.isna(repeated_thresh_val) or repeated_thresh_val < 2: repeated_thresh_val = DEFAULT_REPEATED_THRESHOLD

    # Format threshold values for labels, handle NaN
    fmt_spike = f"{spike_unusual_thresh:.2f}" if pd.notna(spike_unusual_thresh) else "N/A"
    fmt_max = f"{max_val_thresh:.2f}" if pd.notna(max_val_thresh) else "N/A"
    fmt_min = f"{min_val_thresh:.2f}" if pd.notna(min_val_thresh) else "N/A"

    flag_plot_info = {
        'FLAG_BELOW_CAPACITY': ('black', 'Flag: Below Capacity (< 0)'),
        'FLAG_ZERO': ('blue', 'Flag: Zero Discharge'),
        'FLAG_REPEATED': ('green', f'Flag: Repeated Value (>{int(repeated_thresh_val) -1} consecutive)'),
        'FLAG_GREATER_THAN_MaxValue': ('purple', f'Flag: Over Estimated Capacity ({fmt_max})'),
        'UNUSUAL_SPIKE': ('orange', f'Flag: Unusual Spike (Change > {fmt_spike})'),
        'FLAG_LESS_THAN_Min._Value': ('darkred', f'Flag: Below Min Threshold ({fmt_min}, >0)')
    }
    # Define hover template for flagged markers
    hover_tmpl_flags = (f'<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Discharge:</b> %{{y:.2f}} {units}<br><b>%{{meta}}</b><br>Click for actions<extra></extra>')

    logger.debug("Adding flagged points markers...")
    # Iterate through defined flags and add traces if points exist
    for flag_col, (color, label) in flag_plot_info.items():
        if flag_col in df_processed.columns and df_processed[flag_col].any():
            subset = df_processed.loc[df_processed[flag_col]].copy() # Get rows where flag is True
            if not subset.empty:
                 fig.add_trace(go.Scatter(
                     x=subset['Date'], y=subset['DISCHARGE'], mode='markers',
                     marker=dict(color=color, size=7, symbol='circle'), name=label,
                     meta=[label] * len(subset), # Store flag label for hovertemplate %{meta}
                     customdata=subset.index, # Store original DataFrame index
                     hovertemplate=hover_tmpl_flags, # Use template for flags
                     showlegend=True
                 ))
                 logger.debug(f" - Added markers for '{flag_col}' ({len(subset)} points)")


    # --- 3. Add Threshold Lines and Buffer ---
    # Add lines only if thresholds are valid numbers
    if not df_processed.empty and 'Date' in df_processed.columns:
        min_plot_dt, max_plot_dt = df_processed["Date"].min(), df_processed["Date"].max()
        if pd.notna(min_plot_dt) and pd.notna(max_plot_dt):
            plot_date_range = [min_plot_dt, max_plot_dt] # Define range for horizontal lines

            # Min Threshold Line (only if > 0)
            if pd.notna(min_val_thresh) and min_val_thresh > 0:
                 fig.add_trace(go.Scatter(
                     x=plot_date_range, y=[min_val_thresh]*2, mode='lines',
                     name=f"Min Threshold ({fmt_min})", line=dict(color="gray", dash="dash", width=1),
                     hoverinfo='skip', showlegend=True # Skip hover for threshold lines
                 ))

            # Max Threshold Line and Buffer
            if pd.notna(max_val_thresh):
                 max_threshold_label = f"Estimated Max Capacity ({fmt_max})"
                 fig.add_trace(go.Scatter(
                     x=plot_date_range, y=[max_val_thresh]*2, mode='lines',
                     name=max_threshold_label, line=dict(color="purple", dash="dash", width=1.5),
                     hoverinfo='skip', showlegend=True # Skip hover for threshold lines
                 ))
                 # Add gradient buffer above max threshold if valid
                 if max_val_thresh > 0 and len(df_processed['Date']) >= 2:
                     buffer_amount = max_val_thresh * BUFFER_PERCENTAGE
                     logger.debug(f"Adding gradient buffer above Max Thr (Amount: {buffer_amount:.2f})")
                     # Gradient buffer already uses hoverinfo='skip'
                     add_gradient_buffer(fig, df_processed['Date'], max_val_thresh, buffer_amount,
                                         BUFFER_START_COLOR_RGBA, BUFFER_END_COLOR_RGBA, BUFFER_NUM_BANDS, logger)


    # --- 4 & 5. Add Review Status Bar and Legend Header ---
    logger.info("Preparing data for review status progress bar...")
    review_periods = []
    if 'ReviewStatus' in df_processed.columns and 'Date' in df_processed.columns and not df_processed.empty:
        # Sort by date and group by consecutive identical status values
        df_sorted_for_status = df_processed[['Date', 'ReviewStatus']].sort_values('Date')
        df_sorted_for_status['status_group'] = (df_sorted_for_status['ReviewStatus'] != df_sorted_for_status['ReviewStatus'].shift()).cumsum()
        grouped = df_sorted_for_status.groupby('status_group')
        for _, group in grouped:
            if not group.empty:
                 # Get start date, end date, and status for the period
                 review_periods.append((group['Date'].min(), group['Date'].max(), group['ReviewStatus'].iloc[0]))
        logger.info(f"Identified {len(review_periods)} distinct review status periods.")
    else:
        logger.warning("Cannot generate review status bar: 'ReviewStatus' or 'Date' column missing or DataFrame empty.")

    # Prepare data for Plotly Bar traces
    reviewed_bar_x, reviewed_bar_widths, reviewed_bar_hover, reviewed_bar_text = [], [], [], []
    raw_bar_x, raw_bar_widths, raw_bar_hover, raw_bar_text = [], [], [], []
    unknown_bar_x, unknown_bar_widths, unknown_bar_hover, unknown_bar_text = [], [], [], []
    period_boundaries = [] # To draw dividing lines

    if review_periods:
        period_boundaries.append(review_periods[0][0]) # Overall start boundary
        for i, (start_time, end_time, status) in enumerate(review_periods):
            if pd.isna(start_time) or pd.isna(end_time): continue # Skip if dates are invalid

            # Calculate midpoint and width for the bar
            if end_time == start_time: # Handle single-day period
                mid_point = start_time
                # Use approx 1 day width in milliseconds for hover target
                width_ms = 86400 * 1000 * 0.9
            else:
                mid_point = start_time + (end_time - start_time) / 2
                delta_seconds = (end_time - start_time).total_seconds()
                # Ensure minimum width, use actual duration otherwise
                width_ms = max(86400 * 1000 * 0.9, delta_seconds * 1000)

            hover_text = f"Status: {status}<br>From: {start_time.strftime('%Y-%m-%d')}<br>To: {end_time.strftime('%Y-%m-%d')}<extra></extra>"
            bar_label = status # Text inside the bar

            # Append data to the correct list based on status
            if status == 'Reviewed':
                reviewed_bar_x.append(mid_point); reviewed_bar_widths.append(width_ms); reviewed_bar_hover.append(hover_text); reviewed_bar_text.append(bar_label)
            elif status == 'Raw':
                raw_bar_x.append(mid_point); raw_bar_widths.append(width_ms); raw_bar_hover.append(hover_text); raw_bar_text.append(bar_label)
            else: # Assume 'Unknown' or any other status
                unknown_bar_x.append(mid_point); unknown_bar_widths.append(width_ms); unknown_bar_hover.append(hover_text); unknown_bar_text.append(bar_label)

            # Add end boundary for dividing lines (except for the very last period)
            if i < len(review_periods) - 1:
                period_boundaries.append(end_time)
            elif i == len(review_periods) - 1: # Add final end boundary
                period_boundaries.append(end_time)


    # --- Add Status Bar Traces ---
    # Add a non-functional trace to act as a legend title/header for the status bar
    status_header_text = "<b>Data Quality Status</b><br><i>Last year shown as 'Raw' (example)</i>"
    fig.add_trace(go.Scatter(
        mode='markers', x=[None], y=[None], # Invisible trace
        marker=dict(opacity=0, size=0),
        name=status_header_text,
        showlegend=True,
        hoverinfo='skip' # Skip hover for legend header
    ))

    # Add actual status bar traces (using hoverinfo='text' to show hover_text)
    # Assign to secondary y-axis (y2)
    # Define fonts using the (potentially updated) constant size
    text_font_reviewed_raw = dict(color='white', size=PROGRESS_BAR_TEXT_SIZE) # Will use updated size
    text_font_unknown = dict(color='black', size=PROGRESS_BAR_TEXT_SIZE) # Will use updated size

    if reviewed_bar_x: fig.add_trace(go.Bar(
        x=reviewed_bar_x, y=[REVIEW_BAR_Y_VALUE] * len(reviewed_bar_x), width=reviewed_bar_widths, base=0,
        marker_color=REVIEW_BAR_COLORS['Reviewed'], name='Reviewed',
        hovertext=reviewed_bar_hover, hoverinfo='text', text=reviewed_bar_text, textposition='inside',
        insidetextanchor='middle', textfont=text_font_reviewed_raw, yaxis='y2', showlegend=True
    ))
    if raw_bar_x: fig.add_trace(go.Bar(
        x=raw_bar_x, y=[REVIEW_BAR_Y_VALUE] * len(raw_bar_x), width=raw_bar_widths, base=0,
        marker_color=REVIEW_BAR_COLORS['Raw'], name='Raw',
        hovertext=raw_bar_hover, hoverinfo='text', text=raw_bar_text, textposition='inside',
        insidetextanchor='middle', textfont=text_font_reviewed_raw, yaxis='y2', showlegend=True
    ))
    if unknown_bar_x: fig.add_trace(go.Bar(
        x=unknown_bar_x, y=[REVIEW_BAR_Y_VALUE] * len(unknown_bar_x), width=unknown_bar_widths, base=0,
        marker_color=REVIEW_BAR_COLORS['Unknown'], name='Unknown Status',
        hovertext=unknown_bar_hover, hoverinfo='text', text=unknown_bar_text, textposition='inside',
        insidetextanchor='middle', textfont=text_font_unknown, yaxis='y2', showlegend=True
    ))

    # Add dividing lines shapes between status periods
    review_dividing_lines = []
    if len(period_boundaries) > 2: # Need at least start, one internal, end
        internal_boundaries = period_boundaries[1:-1] # Exclude overall start and end
        logger.debug(f"Adding {len(internal_boundaries)} divider lines to review status bar.")
        review_dividing_lines = [
            go.layout.Shape(
                type='line', xref='x', yref='y2 domain', # Refer to the y2 domain
                x0=t, x1=t, y0=0, y1=1, # Line spans the y2 domain height
                line=dict(color='white', width=2), layer='above' # Draw above bars
            ) for t in internal_boundaries if pd.notna(t) # Ensure boundary is valid
        ]


    # 8. Calculate Basic Statistics for the plotted range
    logger.debug("Calculating basic statistics for the plotted data range...")
    stats_dict = None
    if 'DISCHARGE' in df_processed.columns:
        discharge_numeric = df_processed['DISCHARGE'].dropna()
        if not discharge_numeric.empty:
            count = discharge_numeric.count()
            mean_val = discharge_numeric.mean()
            min_val = discharge_numeric.min()
            max_val = discharge_numeric.max()
            stats_dict = {
                "count": f"{count:,}", # Format count with comma
                "mean": f"{mean_val:.2f}" if pd.notna(mean_val) else "N/A",
                "min": f"{min_val:.2f}" if pd.notna(min_val) else "N/A",
                "max": f"{max_val:.2f}" if pd.notna(max_val) else "N/A",
                "units": units # Include units in stats dict
            }
            logger.info(f"Calculated Stats: {stats_dict}")
        else:
            stats_dict = {"count": "0", "mean": "N/A", "min": "N/A", "max": "N/A", "units": units}
            logger.warning("No numeric discharge data found in the final range for stats calculation.")
    else:
        stats_dict = {"count": "N/A", "mean": "N/A", "min": "N/A", "max": "N/A", "units": units}
        logger.warning("DISCHARGE column not found in final df_processed for stats calculation.")


    # 9. Finalize Main Plot Figure Layout
    logger.debug("Finalizing main plot figure layout...")
    # --- MODIFICATION: Increased Gap ---
    progress_bar_height = 0.08 # Relative height for the status bar domain
    # main_plot_domain_start = progress_bar_height + 0.02 # Original gap
    main_plot_domain_start = progress_bar_height + 0.04 # Start main plot above status bar + INCREASED gap
    legend_main_title = "<b>Data Quality Flags</b><br><i>Qualified data is set<br>for Sept as an example</i><br>" # Content remains same

    # --- MODIFICATION: Doubled Font Sizes ---
    fig.update_layout(
        # title=dict(text=plot_title, x=0.5, y=0.97, font_size=18), # Original title
        title=dict(text=plot_title, x=0.5, y=0.97, font_size=36), # Doubled title size

        xaxis=dict(
            title_text="", # No x-axis title
            # title_font_size=14, tickfont_size=12, # Original sizes
            tickfont_size=24, # Doubled tick size
            showline=False, zeroline=True, zerolinewidth=1.5, zerolinecolor='darkgrey',
            # rangeslider=dict(visible=True) # Optional: Add rangeslider
        ),
        yaxis=dict(
            title_text=f"Discharge ({units})",
            # title_font_size=14, tickfont_size=12, # Original sizes
            title_font_size=28, tickfont_size=24, # Doubled sizes
            showline=False, zeroline=True, zerolinewidth=1.5, zerolinecolor='darkgrey',
            domain=[main_plot_domain_start, 1.0] # Use updated domain start with larger gap
        ),
        yaxis2=dict( # Secondary y-axis for the status bar
            domain=[0, progress_bar_height], # Occupies bottom part (uses original height)
            visible=False, # Hide axis labels/ticks
            showticklabels=False, showgrid=False, zeroline=False,
            fixedrange=True # Prevent zooming/panning on status bar axis
        ),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.98, xanchor="left", x=1.01, # Position legend outside plot area
            bgcolor="rgba(255,255,255,0.8)", bordercolor="LightGrey", borderwidth=1,
            # font_size=11, # Original size
            font_size=22, # Doubled legend item size
            tracegroupgap=10, # Space between legend groups
            # title=dict(text=legend_main_title, font=dict(size=12)), # Original legend title size
            title=dict(text=legend_main_title, font=dict(size=24)), # Doubled legend title size
            # Allow toggling traces by clicking legend items
            itemclick='toggle',
            itemdoubleclick='toggle',
        ),
        template="plotly_white", # Use a clean template
        margin=dict(t=60, r=250, b=40, l=80), # Adjust margins (increased right for legend - might need more with large fonts)
        height=650, # Set plot height (might need increasing if large fonts make things cramped)
        hovermode='closest', # Show hover for the single closest point
        shapes=review_dividing_lines, # Add the divider lines to the layout
        clickmode='event+select' # Enable click events for interactivity
    )

    # --- Final Logging and Return ---
    log_end_message = "Plot generation successful."
    if df_processed is not None and not df_processed.empty:
        log_end_message += f" Processed DataFrame prepared ({len(df_processed)} rows)."
        if 'HasQualifier' in df_processed.columns:
             qual_count = df_processed['HasQualifier'].sum()
             log_end_message += f" {qual_count} points marked as qualified."
    else:
        log_end_message += " Processed DataFrame is empty or None."

    logger.info(f"--- {log_end_message} for SiteID {site_id} ---")
    logger.info(f"Final Plot Range Returned: {actual_start_date_str} to {actual_end_date_str}")
    logger.info(f"Station Name Returned: '{station_name}'")
    logger.info(f"Units Returned: '{units}'")
    logger.info(f"Thresholds Used: {site_thresholds}") # Log the thresholds actually used

    # Return the figure, the processed dataframe, error (if any), metadata, and stats
    return fig, df_processed, error_message, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds, stats_dict

# --- END plot_table_generator.py ---