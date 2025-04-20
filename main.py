# -*- coding: utf-8 -*-
# --- Imports ---
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
# Flask imports for web app, sessions, flashing messages
from flask import Flask, render_template_string, request, redirect, url_for, flash, session
import traceback
import logging
import json
from typing import Dict, Any, Tuple, Optional, Sequence # Type hinting
import sys
import os
from pathlib import Path
# File locking import (Unix-specific)
# Consider alternatives like portalocker library for cross-platform compatibility if needed.
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

import time  # For potential retry logic in file locking

# --- Pandas Option ---
pd.set_option('future.no_silent_downcasting', True) # Handle future pandas changes

# --- Flask App Setup ---
app = Flask(__name__)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO) # Ensure Flask logger uses INFO level
# Secret Key is required for session management (used by flash messages)
# Replace with a persistent, environment-specific key in production
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# --- Log fcntl status ---
# Log only once at startup if fcntl is not available
if not HAS_FCNTL:
    app.logger.warning("fcntl module not found. File locking disabled (not available on this OS, e.g., Windows).")


# --- Constants ---
# Core columns needed from thresholds CSV for basic flagging logic
CORE_REQUIRED_THRESHOLD_COLS = ["Over_Capacity", "Unusual_Spike"]
# All columns expected/used from thresholds CSV (including optional/adjustable)
EXPECTED_THRESHOLD_COLS = ["SiteID", "Over_Capacity", "Unusual_Spike", "Repeated_Days", "station_name"]
# Default value for repeated days if column/value is missing or invalid in CSV
DEFAULT_REPEATED_DAYS = 4

# Plotting constants
STATIC_MIN_THRESHOLD = 0 # Minimum value threshold (often zero)
BUFFER_PERCENTAGE = 0.10 # Width of gradient buffer as % of max capacity
BUFFER_NUM_BANDS = 8 # Number of bands for the gradient buffer effect
BUFFER_START_COLOR_RGBA = (128, 0, 128, 0.2) # Semi-transparent purple near the line
BUFFER_END_COLOR_RGBA = (128, 0, 128, 0.0)   # Fully transparent purple at the edge


# --- Path Definition (Using robust method with pathlib) ---
try:
    # Get directory containing the script
    script_dir = Path(__file__).resolve().parent
    csv_filename = "thresholds.csv"
    # Define the primary path relative to the script directory
    THRESHOLDS_CSV_PATH = script_dir / csv_filename
    # Check if the file exists at the primary path
    if not THRESHOLDS_CSV_PATH.is_file():
        app.logger.warning(f"Threshold file not found at '{THRESHOLDS_CSV_PATH}'. Falling back to relative path '{csv_filename}' in current working directory.")
        # Fallback to relative path in the current working directory
        THRESHOLDS_CSV_PATH = Path(csv_filename)
    else:
        app.logger.info(f"Using threshold file path: {THRESHOLDS_CSV_PATH}")
except NameError:
    # Fallback if __file__ is not defined (e.g., running interactively)
    csv_filename = "thresholds.csv"
    THRESHOLDS_CSV_PATH = Path(csv_filename) # Assume relative path in current working directory
    app.logger.warning(f"Could not determine script directory (__file__ not defined). Using relative path: {THRESHOLDS_CSV_PATH}")


# --- Threshold Loading ---
thresholds_df_global = None # Global variable to hold the loaded thresholds DataFrame

# --- Basic File Locking Functions (using fcntl if available) ---
def acquire_lock(file_handle, lock_type, timeout=5):
    """Tries to acquire a file lock (shared or exclusive). Returns True on success, False on timeout/error."""
    if not HAS_FCNTL:
        # app.logger.debug("File locking skipped (fcntl not available).")
        return True # Assume success if locking is not available

    start_time = time.time()
    lock_type_str = "Shared" if lock_type == fcntl.LOCK_SH else "Exclusive"
    while time.time() - start_time < timeout:
        try:
            # Attempt to acquire lock without blocking
            fcntl.flock(file_handle, lock_type | fcntl.LOCK_NB)
            app.logger.debug(f"{lock_type_str} lock acquired for {file_handle.name}")
            return True
        except BlockingIOError: # Error if lock is held by another process (non-blocking mode)
            # app.logger.debug(f"Waiting for {lock_type_str} lock on {file_handle.name}...")
            time.sleep(0.1) # Wait briefly before retrying
        except OSError as e: # Catch other potential OS errors
             # Common errors indicating lock contention or permissions
            if e.errno in [11, 13]: # Resource temporarily unavailable or Permission denied
                # app.logger.debug(f"Waiting for {lock_type_str} lock on {file_handle.name} (OS Error {e.errno})...")
                time.sleep(0.1)
            else:
                app.logger.error(f"Unexpected OSError ({e.errno}) acquiring lock on {file_handle.name}: {e}", exc_info=True)
                raise # Re-raise unexpected OS errors
    # If loop finishes without acquiring lock
    app.logger.error(f"Could not acquire {lock_type_str} lock on {file_handle.name} within {timeout}s.")
    return False

def release_lock(file_handle):
    """Releases a file lock if fcntl is available and handle is valid."""
    if HAS_FCNTL and file_handle and not file_handle.closed:
        try:
            fcntl.flock(file_handle, fcntl.LOCK_UN) # Release the lock
            app.logger.debug(f"Lock released for {file_handle.name}")
        except Exception as e:
            # Log error but don't crash the application for failing to release lock
            app.logger.error(f"Error releasing lock for {file_handle.name}: {e}", exc_info=True)


# --- Threshold Loading Function (Reads CSV with Locking) ---
def load_thresholds(file_path: Path) -> Optional[pd.DataFrame]:
    """Loads the thresholds CSV file into a DataFrame using file locking. Returns None on error."""
    global thresholds_df_global # Allow modification of the global variable
    file_path_str = str(file_path) # For logging and file operations
    app.logger.info(f"Attempting to load thresholds from: {file_path_str}")
    f = None # File handle
    lock_acquired = False
    try:
        # --- Acquire Shared Lock for Reading ---
        # Check existence first to provide a clearer error message
        if not file_path.is_file():
             raise FileNotFoundError(f"File not found at '{file_path_str}'")
        # Open file for reading
        f = open(file_path_str, 'r')
        # Try to acquire a shared lock (allows multiple readers, blocks writers)
        lock_acquired = acquire_lock(f, fcntl.LOCK_SH if HAS_FCNTL else 0) # Use 0 as dummy type if no fcntl
        if not lock_acquired:
            app.logger.error(f"Failed to acquire read lock for {file_path_str}. Aborting load.")
            # Ensure file is closed if lock fails but file was opened
            if f: f.close()
            return None

        # --- Read CSV Data ---
        thresholds_df = pd.read_csv(f) # Read data while holding the lock
        app.logger.info(f"Successfully read CSV file into DataFrame. Checking columns...")

        # --- Validate Required Columns ---
        # Check for absolutely essential columns for core logic
        missing_core_cols = [col for col in CORE_REQUIRED_THRESHOLD_COLS if col not in thresholds_df.columns]
        if missing_core_cols:
            found_cols = thresholds_df.columns.tolist()
            app.logger.error(f"Missing CORE required threshold columns in '{file_path_str}': {', '.join(missing_core_cols)}")
            app.logger.error(f"Columns *found* in CSV header: {found_cols}")
            app.logger.error("Please ensure the core column names exist (check capitalization).")
            return None # Fail loading if core columns are missing

        # Check for SiteID column, also essential
        if "SiteID" not in thresholds_df.columns:
            found_cols = thresholds_df.columns.tolist()
            app.logger.error(f"'SiteID' column not found in thresholds file: '{file_path_str}'.")
            app.logger.error(f"Columns *found* in CSV header: {found_cols}")
            return None # Fail loading if SiteID is missing

        # --- Handle Optional Columns (Log warnings if missing) ---
        if 'Repeated_Days' not in thresholds_df.columns:
             app.logger.warning(f"'Repeated_Days' column not found in '{file_path_str}'. Will use default value: {DEFAULT_REPEATED_DAYS}")
             # Note: Default application happens in get_site_thresholds

        if 'station_name' not in thresholds_df.columns:
            app.logger.warning(f"'station_name' column not found in '{file_path_str}'. Station names might show as 'N/A'.")
            thresholds_df['station_name'] = 'N/A' # Add default column if missing for consistency

        # --- Prepare DataFrame ---
        # Ensure SiteID is stored as string for reliable dictionary lookups later
        thresholds_df["SiteID_str"] = thresholds_df["SiteID"].astype(str)

        app.logger.info(f"Thresholds loaded and columns validated successfully from '{file_path_str}'.")
        thresholds_df_global = thresholds_df # Update the global variable
        return thresholds_df # Return the loaded DataFrame

    # --- Error Handling for Loading ---
    except FileNotFoundError:
        app.logger.error(f"!!! FileNotFoundError: Thresholds file not found at '{file_path_str}'. Please verify the path and file existence.")
        return None
    except pd.errors.EmptyDataError:
         app.logger.error(f"!!! pandas EmptyDataError: The file at '{file_path_str}' seems to be empty.")
         return None
    except pd.errors.ParserError as pe:
         app.logger.error(f"!!! pandas ParserError: Could not parse the CSV file at '{file_path_str}'. Check CSV format (delimiters, quoting, etc.). Error: {pe}", exc_info=False)
         return None
    except PermissionError:
         app.logger.error(f"!!! PermissionError: Do not have permission to read the file at '{file_path_str}'. Check file/folder permissions.")
         return None
    except Exception as e: # Catch any other unexpected errors during loading
        app.logger.error(f"!!! Unexpected Error during threshold loading/validation for '{file_path_str}'. Error Type: {type(e).__name__}, Message: {e}", exc_info=True)
        return None
    finally:
        # --- Release Lock and Close File ---
        # Ensure the lock is released and file is closed in all cases (success or error)
        if lock_acquired and f:
             release_lock(f)
        if f and not f.closed:
            f.close()
# --- END load_thresholds Function ---

# --- Initial Load of Thresholds at Application Startup ---
load_thresholds(THRESHOLDS_CSV_PATH)


# --- Helper Function: Get Thresholds for a Specific Site ---
def get_site_thresholds(thresholds_df: Optional[pd.DataFrame], site_id: str) -> Optional[Dict[str, Any]]:
    """
    Gets and validates thresholds for a specific site from the loaded DataFrame.
    Returns a dictionary including min_val, max_val, spike_unusual, and repeated_days,
    or None if the site is not found or core thresholds are invalid.
    """
    app.logger.info(f"Finding and validating thresholds for SiteID {site_id}...")
    # Check if global DataFrame is loaded
    if thresholds_df is None or thresholds_df.empty:
        app.logger.error("Thresholds DataFrame is not loaded or empty.")
        return None
    # Check if the SiteID_str column (used for lookup) exists
    if "SiteID_str" not in thresholds_df.columns:
        app.logger.error("'SiteID_str' column missing from thresholds DataFrame. Cannot perform lookup.")
        return None

    # Find the row for the given site_id (using the string version for reliable matching)
    site_thresholds_row = thresholds_df[thresholds_df["SiteID_str"] == str(site_id)]

    # Check if any row was found
    if site_thresholds_row.empty:
        app.logger.warning(f"SiteID {site_id} not found in thresholds data.")
        return None

    try:
        # Get the first (and should be only) row for the site
        threshold_row = site_thresholds_row.iloc[0]
        # Initialize dictionary to store validated thresholds
        validated_thresholds = {"min_val": float(STATIC_MIN_THRESHOLD)}
        missing_details = [] # Track missing core columns/values
        validation_errors = [] # Track non-fatal validation issues (e.g., defaults used)

        # --- Validate Core Required Numeric Thresholds ---
        for col_name in CORE_REQUIRED_THRESHOLD_COLS:
            raw_value = threshold_row.get(col_name) # Safely get value
            # Check if column value is missing or explicitly None/NaN
            if raw_value is None or pd.isna(raw_value):
                 missing_details.append(f"'{col_name}' (value missing or NaN)")
                 continue # Skip to next core column

            # Try converting to numeric, coerce errors to NaN
            numeric_value = pd.to_numeric(raw_value, errors='coerce')

            # Check if conversion failed
            if pd.isna(numeric_value):
                missing_details.append(f"'{col_name}' (value '{raw_value}' is not numeric)")
            else:
                # Assign validated numeric value to the correct key
                if col_name == "Over_Capacity":
                    validated_thresholds["max_val"] = float(numeric_value)
                elif col_name == "Unusual_Spike":
                    validated_thresholds["spike_unusual"] = float(numeric_value)
                # Add other core mappings here if needed

        # --- Validate Optional 'Repeated_Days' ---
        raw_repeated = threshold_row.get("Repeated_Days") # Safely get value
        # Check if missing or NaN
        if raw_repeated is None or pd.isna(raw_repeated):
            app.logger.warning(f"SiteID {site_id}: 'Repeated_Days' value missing or NaN. Using default: {DEFAULT_REPEATED_DAYS}")
            validated_thresholds["repeated_days"] = int(DEFAULT_REPEATED_DAYS)
        else:
            # Try converting to integer, raise error if not possible
            try:
                repeated_val_int = int(pd.to_numeric(raw_repeated, errors='raise'))
                # Apply logical check (must be at least 2 for a repeat sequence)
                if repeated_val_int >= 2:
                    validated_thresholds["repeated_days"] = repeated_val_int
                else:
                    # Value is integer but too small, use default
                    validation_errors.append(f"'Repeated_Days' (value: {repeated_val_int}) must be >= 2. Using default: {DEFAULT_REPEATED_DAYS}")
                    validated_thresholds["repeated_days"] = int(DEFAULT_REPEATED_DAYS)
            except (ValueError, TypeError):
                 # Conversion to int failed, use default
                 validation_errors.append(f"'Repeated_Days' (value: '{raw_repeated}') is not a valid integer. Using default: {DEFAULT_REPEATED_DAYS}")
                 validated_thresholds["repeated_days"] = int(DEFAULT_REPEATED_DAYS)

        # --- Final Checks and Return ---
        # If any core required value was missing or invalid, fail validation
        if missing_details:
            error_message = (f"Missing or invalid CORE required threshold value(s) for SiteID {site_id} "
                             f"in thresholds data: {', '.join(missing_details)}. Please check the source CSV.")
            app.logger.error(error_message)
            return None # Return None indicates failure to get valid core thresholds

        # Log any non-fatal validation warnings (e.g., if defaults were used)
        if validation_errors:
            for err in validation_errors:
                app.logger.warning(f"SiteID {site_id}: {err}")

        # Internal check: Ensure all required keys were actually populated in the dict
        if "max_val" not in validated_thresholds or "spike_unusual" not in validated_thresholds:
            app.logger.error(f"Internal logic error: Did not populate 'max_val' or 'spike_unusual' for SiteID {site_id} despite passing checks.")
            return None

        app.logger.info(f"Threshold values successfully validated for SiteID {site_id}. Using: {validated_thresholds}")
        return validated_thresholds # Return the dictionary of validated thresholds

    except Exception as e: # Catch any unexpected errors during row processing
        app.logger.error(f"Unexpected error extracting/validating thresholds for SiteID {site_id}: {e}", exc_info=True)
        return None
# --- END get_site_thresholds Function ---


# --- Helper Function: Apply Data Quality Flags ---
def apply_flagging(df: pd.DataFrame, thresholds: Dict[str, Any]) -> pd.DataFrame:
    """Applies data quality flags based on provided thresholds dictionary."""
    app.logger.info("Applying flagging logic...")
    # Check if the necessary threshold keys exist in the provided dictionary
    required_keys = ['min_val', 'max_val', 'spike_unusual', 'repeated_days']
    if not thresholds or not all(key in thresholds for key in required_keys):
        app.logger.error("Cannot apply flagging: Invalid or incomplete thresholds dictionary provided.")
        missing_keys = [k for k in required_keys if k not in thresholds] if thresholds else required_keys
        app.logger.error(f"Missing keys: {missing_keys}")
        # Ensure flag columns exist even if flagging fails, set all to False
        df['FLAGGED'] = False
        flag_cols_expected = ['FLAG_LESS_THAN_Min._Value', 'FLAG_ZERO', 'FLAG_REPEATED',
                              'FLAG_GREATER_THAN_MaxValue', 'UNUSUAL_SPIKE', 'FLAG_BELOW_CAPACITY']
        for col in flag_cols_expected:
            if col not in df.columns: df[col] = False
        return df

    # Extract validated thresholds from the dictionary
    min_val = thresholds["min_val"]
    max_val = thresholds["max_val"]
    spike_unusual = thresholds["spike_unusual"]
    repeated_days_threshold = thresholds["repeated_days"]

    app.logger.info(f"Using thresholds: Min={min_val}, Max={max_val}, Spike={spike_unusual}, RepeatedDays={repeated_days_threshold}")

    # Check if 'DISCHARGE' column exists before proceeding
    if 'DISCHARGE' in df.columns:
        # Ensure DISCHARGE is numeric, converting non-numeric to NaN
        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')

        # --- Apply Individual Flags ---
        # Note: .notna() check ensures NaNs don't get flagged incorrectly
        df['FLAG_LESS_THAN_Min._Value'] = (df['DISCHARGE'] < min_val) & (df['DISCHARGE'].notna()) & (df['DISCHARGE'] != 0) # Exclude zero from this specific flag
        df['FLAG_ZERO'] = df['DISCHARGE'] == 0 # Flag exact zeros
        df['FLAG_BELOW_CAPACITY'] = (df['DISCHARGE'] < 0) & (df['DISCHARGE'].notna()) # Flag negative values specifically
        df['FLAG_GREATER_THAN_MaxValue'] = (df['DISCHARGE'] > max_val) & (df['DISCHARGE'].notna()) # Flag values exceeding max capacity

        # --- Rate of Change Flag (Unusual Spike) ---
        # Calculate absolute difference between consecutive points
        df['RATE_OF_CHANGE'] = df['DISCHARGE'].diff().abs()
        # Flag if rate of change exceeds the unusual spike threshold
        df['UNUSUAL_SPIKE'] = (df['RATE_OF_CHANGE'] > spike_unusual) & (df['RATE_OF_CHANGE'].notna())
        app.logger.info(f"Applied Unusual Spike Threshold: {spike_unusual}. Found {df['UNUSUAL_SPIKE'].sum()} spikes.")

        # --- Repeated Value Flag ---
        df['FLAG_REPEATED'] = False # Initialize column
        # Consider only non-zero, non-NaN values for repeat check
        non_zero_discharge = df['DISCHARGE'].where((df['DISCHARGE'] != 0) & df['DISCHARGE'].notna())
        # Check if there are any values left to process
        if not non_zero_discharge.isna().all():
            # Identify groups of consecutive identical values
            # The cumsum() trick creates unique group IDs for each consecutive block
            group_ids = (non_zero_discharge != non_zero_discharge.shift()).cumsum()
            # Count the size of each consecutive group
            repeat_counts = non_zero_discharge.groupby(group_ids).transform('size')
            # Flag rows where the count meets or exceeds the threshold
            # Use .loc to ensure alignment, applying only where non_zero_discharge is not NaN
            df.loc[non_zero_discharge.notna(), 'FLAG_REPEATED'] = repeat_counts >= repeated_days_threshold

        app.logger.info(f"Found {df['FLAG_REPEATED'].sum()} instances of repeated non-zero values (>={repeated_days_threshold} days).")

        # --- Combine Flags ---
        # Define the list of individual flag columns
        flag_columns_list = ['FLAG_LESS_THAN_Min._Value', 'FLAG_ZERO', 'FLAG_REPEATED',
                             'FLAG_GREATER_THAN_MaxValue', 'UNUSUAL_SPIKE', 'FLAG_BELOW_CAPACITY']
        # Filter list to include only columns that actually exist in the DataFrame (safety check)
        existing_flag_columns = [col for col in flag_columns_list if col in df.columns]
        # Create the overall 'FLAGGED' column: True if *any* of the individual flags are True for a row
        if existing_flag_columns:
            df['FLAGGED'] = df[existing_flag_columns].any(axis=1)
        else:
            # Should not happen if columns are created above, but fallback
            app.logger.warning("No individual flag columns found to create combined 'FLAGGED' column.")
            df['FLAGGED'] = False

        app.logger.info(f"Total flagged points: {df['FLAGGED'].sum()}")

    else:
        # Handle case where 'DISCHARGE' column is missing entirely
        app.logger.warning("Cannot apply flagging: 'DISCHARGE' column not found in DataFrame.")
        df['FLAGGED'] = False
        # Ensure flag columns exist even if flagging fails
        flag_cols_expected = ['FLAG_LESS_THAN_Min._Value', 'FLAG_ZERO', 'FLAG_REPEATED',
                              'FLAG_GREATER_THAN_MaxValue', 'UNUSUAL_SPIKE', 'FLAG_BELOW_CAPACITY']
        for col in flag_cols_expected:
            if col not in df.columns: df[col] = False

    return df
# --- END apply_flagging Function ---


# --- Helper Function: Validate Date String ---
def validate_date(date_str):
    """Validate date string format YYYY-MM-DD. Returns datetime object or None."""
    if not date_str: # Handle empty string or None input
        return None
    try:
        # Attempt to parse the string into a datetime object
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        # Parsing failed, format is incorrect
        return None
# --- END validate_date Function ---


# --- Helper Function: Interpolate Color for Gradient ---
def interpolate_color(color1_rgba: Tuple[int, int, int, float],
                      color2_rgba: Tuple[int, int, int, float],
                      fraction: float) -> str:
    """ Interpolates between two RGBA colors based on a fraction (0.0 to 1.0). Returns rgba string."""
    r1, g1, b1, a1 = color1_rgba
    r2, g2, b2, a2 = color2_rgba
    # Clamp fraction to ensure it's within the valid range [0, 1]
    fraction = max(0.0, min(1.0, fraction))
    # Linear interpolation for each component (R, G, B, A)
    r = int(r1 + (r2 - r1) * fraction)
    g = int(g1 + (g2 - g1) * fraction)
    b = int(b1 + (b2 - b1) * fraction)
    a = a1 + (a2 - a1) * fraction
    # Clamp RGB values to valid 0-255 range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    # Return the result as a Plotly-compatible rgba string
    return f'rgba({r},{g},{b},{a:.4f})' # Format alpha to 4 decimal places
# --- END interpolate_color Function ---


# --- Helper Function: Add Gradient Buffer to Plot ---
def add_gradient_buffer(fig: go.Figure,
                        dates: Sequence, # Sequence of dates (e.g., list, pd.Series)
                        mean_value: float, # Center line value (e.g., max capacity)
                        buffer: float, # Width of the buffer (+/- from mean_value)
                        start_color_rgba: Tuple[int, int, int, float], # Color near mean_value
                        end_color_rgba: Tuple[int, int, int, float], # Color at buffer edge
                        num_bands: int = 15): # Number of gradient steps
    """ Adds gradient filled buffer bands around a central line value on a Plotly figure. """
    # Basic input validation
    if buffer <= 0 or num_bands <= 0:
        app.logger.warning("Buffer width or number of bands is non-positive, skipping gradient.")
        return
    n_points = len(dates)
    if n_points < 2: # Need at least two points to draw a polygon
         app.logger.warning("Not enough date points to draw gradient buffer.")
         return

    # Prepare x-coordinates for the filled polygon shape (forward then backward)
    x_coords_polygon = list(dates) + list(dates)[::-1]

    # Draw bands from outermost (most transparent) to innermost (most opaque)
    # This layering ensures correct visual appearance
    for i in range(num_bands - 1, -1, -1):
        # Calculate fractions representing the edges of the current band relative to the total buffer width
        outer_fraction = (i + 1) / num_bands # Outer edge fraction (further from mean)
        inner_fraction = i / num_bands       # Inner edge fraction (closer to mean)

        # Interpolate the color for this band based on its position
        # Using outer_fraction means color fades towards end_color_rgba at the edge
        band_color = interpolate_color(start_color_rgba, end_color_rgba, outer_fraction)

        # --- Upper Buffer Band (Above mean_value) ---
        # Calculate y-coordinates for the inner and outer edges of this band
        band_lower_y_upper = mean_value + inner_fraction * buffer # Inner edge (closer to mean)
        band_upper_y_upper = mean_value + outer_fraction * buffer # Outer edge (further from mean)

        # Check for non-finite values (NaN, inf) which Plotly cannot plot
        if not (np.isfinite(band_lower_y_upper) and np.isfinite(band_upper_y_upper)):
            app.logger.debug(f"Skipping upper gradient band {i} due to non-finite y-values.")
            continue # Skip this band if values are invalid

        # Prepare y-coordinates for the polygon (inner edge forward, outer edge backward)
        y_coords_upper = [band_lower_y_upper] * n_points + [band_upper_y_upper] * n_points
        # Add the filled scatter trace for this band
        fig.add_trace(go.Scatter(
            x=x_coords_polygon, y=y_coords_upper, fill='toself', fillcolor=band_color,
            line=dict(color='rgba(0,0,0,0)', width=0), # No visible line for the band itself
            hoverinfo="skip", showlegend=False, # Hide from hover and legend
            mode='lines' # Necessary for 'toself' fill to work
        ))

        # --- Lower Buffer Band (Below mean_value) ---
        # Calculate y-coordinates for the inner and outer edges
        band_upper_y_lower = mean_value - inner_fraction * buffer # Inner edge (closer to mean)
        band_lower_y_lower = mean_value - outer_fraction * buffer # Outer edge (further from mean)

        # Check for non-finite values
        if not (np.isfinite(band_upper_y_lower) and np.isfinite(band_lower_y_lower)):
            app.logger.debug(f"Skipping lower gradient band {i} due to non-finite y-values.")
            continue # Skip this band

        # Prepare y-coordinates for the polygon (outer edge forward, inner edge backward)
        y_coords_lower = [band_lower_y_lower] * n_points + [band_upper_y_lower] * n_points
        # Add the filled scatter trace
        fig.add_trace(go.Scatter(
            x=x_coords_polygon, y=y_coords_lower, fill='toself', fillcolor=band_color,
            line=dict(color='rgba(0,0,0,0)', width=0),
            hoverinfo="skip", showlegend=False,
            mode='lines'
        ))
# --- END add_gradient_buffer Function ---


# --- HTML Template Definition (with JS Debugging) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ station_name | default('Data Quality Analysis', true) }} (Site ID {{ site_id | default('N/A', true) }}) {{ start_date }} to {{ end_date }}</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        h1 { text-align: center; margin-block-start: 0.67em; margin-block-end: 0.67em; line-height: 1.3; }
        .error { color: red; text-align: center; font-weight: bold; margin-top: 15px; border: 1px solid red; padding: 10px; background-color: #ffecec; }
        .success { color: green; text-align: center; font-weight: bold; margin-top: 15px; border: 1px solid green; padding: 10px; background-color: #e7ffe7; }
        #plot_container { margin-top: 20px; min-height: 100px; background-color: #f0f0f0; }
        /* Updated controls style */
        .controls {
            text-align: center; margin-bottom: 20px; padding: 15px;
            border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9;
            display: flex; flex-direction: column; /* Stack form and threshold controls */
            align-items: center; gap: 15px;
        }
        .date-controls-wrapper { /* Wrap date form and quick buttons */
             display: flex; flex-wrap: wrap; justify-content: center;
             align-items: center; gap: 10px; width: 100%;
        }
        .date-controls-wrapper form { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 10px; }
        .controls label, .controls input, .controls button { margin: 0 5px; vertical-align: middle; }
        .controls input[type="submit"], .controls button { padding: 5px 15px; cursor: pointer; font-size: 1em; }
        .quick-date-buttons button { font-size: 0.9em; padding: 4px 10px;} /* Style quick buttons */

        .threshold-controls { text-align: left; padding: 15px; border: 1px dashed #aaa; border-radius: 5px; background-color: #fafafa; width: 100%; max-width: 500px; /* Limit width */ box-sizing: border-box;}
        .threshold-controls h3 { margin-top: 0; text-align: center; font-size: 1.1em; margin-bottom: 10px;}
        .threshold-controls form div { margin-bottom: 8px; display: flex; align-items: center; justify-content: center; gap: 5px;} /* Flexbox for alignment */
        .threshold-controls label { display: inline-block; width: 150px; text-align: right; font-size: 0.9em;}
        .threshold-controls input[type="number"] { width: 100px; padding: 3px; font-size: 0.9em;}
        .threshold-controls small { font-size: 0.8em; color: #555; } /* Style for units/context */
        .threshold-controls input[type="submit"] { display: block; margin: 10px auto 0 auto; padding: 6px 18px; font-size: 0.95em; background-color: #e0e0e0; border: 1px solid #adadad;}
        .threshold-controls input[type="submit"]:hover { background-color: #d0d0d0; }
        .plot-title-info { text-align: center; font-size: 30px; margin-bottom: 10px; }
        .header-link { font-size: 30px; color: darkblue; text-decoration: none; }
        .header-link:hover { text-decoration: underline; }
        .header-text-no-link { font-size: 14px; color: darkblue; }
        .modal { display: none; position: fixed; z-index: 1000; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 300px; max-width: 90%; padding: 20px; background-color: #fefefe; border: 3px solid red; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); border-radius: 5px; text-align: center; }
        .modal-content { position: relative; } .modal-content h4 { margin-top: 0; } .modal-content p { margin: 10px 0; font-size: 14px; word-wrap: break-word; } .modal-close { color: #aaa; position: absolute; top: 5px; right: 10px; font-size: 24px; font-weight: bold; line-height: 1; cursor: pointer; padding: 0 5px; } .modal-close:hover, .modal-close:focus { color: black; text-decoration: none; } .modal-button { display: inline-block; padding: 5px 10px; font-size: 12px; font-weight: bold; font-family: sans-serif; margin: 5px 3px; cursor: pointer; background-color: #e7e7e7; color: #333; border: 1px solid #adadad; border-radius: 4px; text-decoration: none; text-align: center; line-height: 1.4; white-space: nowrap; box-shadow: 0 1px 1px rgba(0,0,0,0.1); -webkit-appearance: button; -moz-appearance: button; appearance: button; } .modal-button:hover { background-color: #dcdcdc; border-color: #999999; text-decoration: none; color: #000; box-shadow: 0 1px 1px rgba(0,0,0,0.2); } .modal-button:active { background-color: #cccccc; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <h1><span style="font-size: 24px;">Data Quality Analysis for Measurement Site</span><br>{% if site_id and site_id != 'N/A' and site_id is not none %}<a href="https://waterrights.utah.gov/cgi-bin/dvrtview.exe?Modinfo=StationView&STATION_ID={{ site_id }}" target="_blank" rel="noopener noreferrer" class="header-link">{% if units and units != 'Unknown Units' %}{{ units }} - {% endif %}{{ station_name | default('Unknown Station', true) }} (Site ID={{ site_id }})</a>{% else %}<span class="header-text-no-link">{% if units and units != 'Unknown Units' %}{{ units }} - {% endif %}{{ station_name | default('Unknown Station', true) }} (Site ID={{ site_id | default('N/A', true) }})</span>{% endif %}</h1>

    {# --- Display Flash Messages --- #}
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        <div style="text-align:center; margin-bottom: 15px;">
        {% for category, message in messages %}
          <div class="{{ category }}">{{ message }}</div>
        {% endfor %}
        </div>
      {% endif %}
    {% endwith %}
    {% if error %} <p class="error">Error: {{ error }}</p> {% endif %} {# Display other errors passed directly #}

    <div class="controls">
        <div class="date-controls-wrapper"> {# Wrapper for date form + quick buttons #}
            <form method="GET" action="/plot" id="plotForm">
                <label for="id">Site ID:</label>
                <input type="text" id="id" name="id" value="{{ site_id | default('', true) }}" required>
                <label for="start_date">Start Date:</label>
                <input type="date" id="start_date" name="start_date" value="{{ start_date | default('', true) }}" required>
                <label for="end_date">End Date:</label>
                <input type="date" id="end_date" name="end_date" value="{{ end_date | default('', true) }}" required>
                <input type="submit" value="Update Plot">
            </form>
            <div class="quick-date-buttons"> {# Container for quick buttons #}
                 <button type="button" onclick="resetDates()">Full Range (Reset)</button> {# Renamed Reset #}
                 <button type="button" onclick="setQuickRange('year')">Last Year</button>
                 <button type="button" onclick="setQuickRange('month')">Last Month</button>
            </div>
        </div>

        {# --- Threshold Adjustment Form --- #}
        {% if site_id and current_thresholds %}
        <div class="threshold-controls">
            <h3>Adjust Quality Control Thresholds for Site {{ site_id }}</h3>
            <form method="POST" action="/update_thresholds">
                {# Hidden fields to pass context back #}
                <input type="hidden" name="site_id" value="{{ site_id }}">
                <input type="hidden" name="start_date" value="{{ start_date | default('', true) }}">
                <input type="hidden" name="end_date" value="{{ end_date | default('', true) }}">

                <div>
                    <label for="max_val">Max Capacity:</label>
                    <input type="number" step="any" id="max_val" name="max_val" value="{{ current_thresholds.max_val }}" required>
                    <small>({{ units if units != 'Unknown Units' else 'units' }})</small>
                </div>
                <div>
                    <label for="spike_unusual">Unusual Spike RoC:</label>
                    <input type="number" step="any" id="spike_unusual" name="spike_unusual" value="{{ current_thresholds.spike_unusual }}" required>
                     <small>({{ units if units != 'Unknown Units' else 'units' }}/day)</small>
               </div>
                <div>
                    <label for="repeated_days">Repeated Value Days:</label>
                    <input type="number" step="1" min="2" id="repeated_days" name="repeated_days" value="{{ current_thresholds.repeated_days }}" required>
                    <small>(days, min 2)</small>
                </div>
                <input type="submit" value="Update Thresholds">
            </form>
        </div>
        {% elif site_id %}
        <div class="threshold-controls">
             <p style="text-align: center; color: #888;">Thresholds could not be loaded for editing for Site ID {{ site_id }}.</p>
        </div>
        {% endif %}
        {# --- End Threshold Adjustment Form --- #}

    </div> {# End controls div #}

    {# --- Plot Area and Modal --- #}
    {% if plot_div %}
        <div id='plot_container'>{{ plot_div | safe }}</div>
        <div id="pointActionModal" class="modal">
            <div class="modal-content">
                <span id="closeModal" class="modal-close" title="Close">&times;</span>
                <h4>Quality Control Decision</h4>
                <p id="modalPointInfo">Point details will appear here.</p>
                <div id="modalActions"></div>
            </div>
        </div>

        {# --- JavaScript Section with Debugging --- #}
        <script>
            // Wrap major parts in DOMContentLoaded to ensure elements exist
            document.addEventListener('DOMContentLoaded', function() {
                console.log("[Debug] DOM fully loaded and parsed.");

                // --- Date Formatting Helper ---
                function formatDate(date) {
                    // Formats a Date object into YYYY-MM-DD
                    try {
                        const year = date.getFullYear();
                        // Months are 0-indexed, add 1
                        const month = String(date.getMonth() + 1).padStart(2, '0');
                        const day = String(date.getDate()).padStart(2, '0');
                        const formatted = `${year}-${month}-${day}`;
                        console.log("[Debug] formatDate input:", date, "output:", formatted);
                        return formatted;
                    } catch (e) {
                        console.error("[Debug] Error in formatDate:", e);
                        return null; // Return null on error
                    }
                }

                // --- Get Site ID Helper ---
                function getSiteIdValue() {
                    console.log("[Debug] Running getSiteIdValue...");
                    const siteIdInput = document.getElementById('id');
                    if (!siteIdInput) {
                        console.error("[Debug] Site ID input field (#id) not found.");
                        alert("Internal error: Cannot find Site ID field.");
                        return null;
                    }
                    const siteId = siteIdInput.value.trim();
                     console.log("[Debug] Site ID input value found:", siteId);
                    if (!siteId) {
                        console.warn("[Debug] Site ID is empty.");
                        alert("Please enter a Site ID first before selecting a date range.");
                        return null;
                    }
                    console.log("[Debug] Site ID obtained:", siteId);
                    return siteId;
                }

                // --- Make functions globally accessible for onclick handlers ---
                // Alternatively, attach event listeners programmatically below
                window.setQuickRange = function(period) {
                     console.log(`[Debug] setQuickRange called with period: ${period}`);
                     const siteId = getSiteIdValue();
                     if (!siteId) {
                         console.log("[Debug] setQuickRange stopped: No valid site ID.");
                         return; // Stop if no valid site ID
                     }

                     const today = new Date();
                     console.log("[Debug] Today's date:", today);
                     const endDateStr = formatDate(today);
                     let startDate = new Date(); // Start from today

                     if (!endDateStr) {
                         console.error("[Debug] Could not format end date. Aborting.");
                         alert("Error formatting current date.");
                         return;
                     }

                     if (period === 'year') {
                         console.log("[Debug] Calculating start date for 'Last Year'");
                         startDate.setFullYear(startDate.getFullYear() - 1);
                     } else if (period === 'month') {
                         console.log("[Debug] Calculating start date for 'Last Month' (-30 days)");
                         startDate.setDate(startDate.getDate() - 30);
                     } else {
                         console.error("[Debug] Invalid period specified for setQuickRange:", period);
                         return;
                     }
                     console.log("[Debug] Calculated start date obj:", startDate);
                     const startDateStr = formatDate(startDate);

                     if (!startDateStr) {
                         console.error("[Debug] Could not format start date. Aborting.");
                         alert("Error formatting start date.");
                         return;
                     }

                     // Construct URL and redirect
                     const newUrl = `/plot?id=${encodeURIComponent(siteId)}&start_date=${startDateStr}&end_date=${endDateStr}`;
                     console.log("[Debug] Redirecting to (Quick Range):", newUrl);
                     try {
                        window.location.href = newUrl;
                     } catch (e) {
                        console.error("[Debug] Error during window.location.href assignment:", e);
                        alert("An error occurred while trying to redirect.");
                     }
                }

                window.resetDates = function() {
                     console.log("[Debug] resetDates called.");
                     const siteId = getSiteIdValue();
                     if (!siteId) {
                        console.log("[Debug] resetDates stopped: No valid site ID.");
                        return; // Also check siteId here
                     }
                     const newUrl = `/plot?id=${encodeURIComponent(siteId)}&reset=true`;
                     console.log("[Debug] Redirecting to (Reset):", newUrl);
                     try {
                        window.location.href = newUrl;
                     } catch (e) {
                         console.error("[Debug] Error during window.location.href assignment:", e);
                         alert("An error occurred while trying to redirect.");
                     }
                }

                // --- Modal/Plot Interaction Functions (Can remain as they were) ---
                 window.pointAction = function(action, index, date, value) {
                    console.log("Action:", action, "Index:", index, "Date:", date, "Value:", value);
                    let valueStr = typeof value === 'number' ? value.toFixed(2) : String(value);
                    alert(action + " clicked for point:\nDate: " + date + "\nValue: " + valueStr + "\n(Point Index: " + index + ")\n\nAction not yet implemented.");
                    var modal = document.getElementById("pointActionModal");
                    if (modal) {
                        modal.style.display = "none";
                    }
                }

                window.plotInteractionRetry = false; // Flag to prevent infinite loops if Plotly fails
                window.modalHandlersAttached = false; // Ensure flag exists

                function initializePlotInteraction() {
                    console.log("[Init] Starting interaction setup (include_plotlyjs='cdn' mode)...");
                    var plotContainer = document.getElementById('plot_container');
                    var plotDiv = null;

                    if (!plotContainer) {
                        console.error("[Init] Plot container (#plot_container) not found! Cannot find plot div.");
                        return;
                    }

                    // Wait briefly for Plotly to potentially render
                    setTimeout(function() {
                        console.log("[Init] Attempting to find plot div inside #plot_container after delay...");
                        plotDiv = plotContainer.querySelector("div.js-plotly-plot");
                        if (!plotDiv) { plotDiv = plotContainer.querySelector("div.plotly-graph-div"); }
                        if (!plotDiv) {
                            var potentialDivs = plotContainer.getElementsByTagName("div");
                            if (potentialDivs.length > 0 && potentialDivs[0].id && potentialDivs[0].id.startsWith("plotly-")) {
                                plotDiv = potentialDivs[0];
                                console.warn("[Init] Couldn't find plot div by class, using first child div with Plotly ID:", plotDiv);
                            }
                        }

                        if (!plotDiv) {
                            console.error("[Init] Plot div element NOT FOUND within #plot_container even after delay. Cannot attach listener.");
                            if(typeof Plotly !== 'undefined') { console.warn("[Init] Plotly library IS loaded, but the plot div element wasn't found by selectors."); }
                            else { console.error("[Init] Plotly library is ALSO not loaded (unexpected in this mode)."); }
                            return;
                        }

                        console.log("[Init] Found plotDiv element:", plotDiv);
                        attachPlotlyListeners(plotDiv);

                    }, 350); // Slightly longer delay just in case
                }

                function attachPlotlyListeners(plotDiv) {
                    console.log("[Attach] Attempting to attach listeners to:", plotDiv);
                    var modal = document.getElementById('pointActionModal');
                    var closeModal = document.getElementById('closeModal');
                    var modalPointInfo = document.getElementById('modalPointInfo');
                    var modalActions = document.getElementById('modalActions');

                    if (!modal || !closeModal || !modalPointInfo || !modalActions) {
                         console.error("[Attach] Modal elements missing.");
                         return;
                    }

                    if (!window.modalHandlersAttached) {
                         closeModal.onclick = function() { if (modal) modal.style.display = "none"; };
                        window.addEventListener('click', function(event) {
                            if (modal && modal.style.display === 'block' && event.target === modal) {
                                 if (!modal.querySelector('.modal-content').contains(event.target)) {
                                    modal.style.display = 'none';
                                 }
                            }
                        });
                         console.log("[Attach] Modal close handlers attached.");
                         window.modalHandlersAttached = true;
                    }

                    try {
                        if (typeof plotDiv.on !== 'function' && plotDiv.id) {
                             plotDiv = document.getElementById(plotDiv.id);
                             if (typeof plotDiv.on !== 'function') { throw new Error("plotDiv.on is still not a function after re-fetch by ID."); }
                             console.warn("[Attach] Re-fetched plotDiv to find .on method.");
                        }

                        console.log("[Attach] Attaching 'plotly_click' listener...");
                        plotDiv.on('plotly_click', function(data) {
                            console.log("==== Plotly CLICK Event ====", data);
                            if (!modal || !modalPointInfo || !modalActions) { console.error("[plotly_click] Modal elements missing inside handler."); return; }
                            if (!data || !data.points || data.points.length === 0) { console.log("[plotly_click] Click was not on a data point."); return; }

                            var point = data.points[0];
                            if (point.curveNumber > 0 && point.fullData && point.fullData.mode && point.fullData.mode.includes('markers')) {
                                 console.log("[plotly_click] Clicked on flagged point.");
                                 var pointIndex = point.pointNumber;
                                 var pointDate = point.x;
                                 var pointValue = point.y;
                                 var traceName = point.fullData ? point.fullData.name : "Unknown Trace";
                                 var dateStr = pointDate;
                                 var valueStr = typeof pointValue === 'number' ? pointValue.toFixed(2) : String(pointValue);
                                 var flagType = String(traceName).split('[')[0].trim();

                                 modalPointInfo.innerHTML = `<b>Date:</b> ${dateStr}<br><b>Value:</b> ${valueStr}<br><b>Flag:</b> ${flagType}`;
                                 modalActions.innerHTML = ''; // Clear previous

                                 // Create buttons... (omitted for brevity, same as before)
                                 ['Approve - Correct Value', 'Interpolate - Estimate', 'Delete: enter manual measurement'].forEach(actionText => {
                                     var btn = document.createElement('button');
                                     btn.className = 'modal-button';
                                     btn.innerText = actionText;
                                     let actionCode = actionText.split(' ')[0]; // Basic action code
                                     btn.onclick = () => pointAction(actionCode, pointIndex, pointDate, pointValue);
                                     modalActions.appendChild(btn);
                                 });

                                 modal.style.display = 'block';
                                 console.log("[plotly_click] Modal displayed.");
                            } else {
                                 console.log("[plotly_click] Clicked on base line or non-marker trace.");
                                 if (modal) modal.style.display = "none";
                            }
                        });
                        console.log("[Attach] 'plotly_click' listener attached.");

                        plotDiv.on('plotly_afterplot', function() { console.log("---- Plotly AFTERPLOT Event ----"); });
                        console.log("[Attach] 'plotly_afterplot' listener attached.");

                    } catch (error) {
                        console.error("[Attach] FAILED to attach listener:", error);
                    }
                } // end attachPlotlyListeners

                // --- Initialize Plot Interaction if plot exists ---
                if (document.getElementById('plot_container') && document.getElementById('plot_container').innerHTML.trim() !== '') {
                    initializePlotInteraction();
                } else {
                     console.log("[Debug] No plot container content found on initial load, skipping plot interaction setup.");
                }

            }); // End DOMContentLoaded listener
        </script>
    {% elif not error and not site_id %}
        <p style="text-align: center;">Please enter a Site ID and select a date range above.</p>
    {% elif not error and site_id and not plot_div %}
        <p style="text-align: center;">No plot generated. Check if data exists for the selected Site ID and date range, or if there was an error fetching/processing data or loading thresholds.</p>
    {% endif %}
</body>
</html>"""
# --- END HTML Template ---


# --- Core Data Processing and Plotting Function ---
# [ generate_plot_for_site function remains unchanged from the previous version ]
# [ It correctly uses the thresholds dictionary and returns necessary values ]
def generate_plot_for_site(site_id, start_date_str_requested, end_date_str_requested, is_reset=False):
    """
    Fetches data, applies flags, and generates the Plotly figure for a given site.
    Returns: tuple(fig, error_msg, station_name, start_date_actual, end_date_actual, units, site_thresholds_dict)
    """
    station_name = None
    actual_start_date_str = start_date_str_requested
    actual_end_date_str = end_date_str_requested
    df = pd.DataFrame()
    metadata = {}
    units = 'Unknown Units'
    site_thresholds = None # This will hold the dict from get_site_thresholds

    app.logger.info(f"Generating plot - Input: Site ID: {site_id}, Start Req: {start_date_str_requested}, End Req: {end_date_str_requested}, Reset Flag: {is_reset}")

    # --- 1. Load/Verify Thresholds ---
    global thresholds_df_global
    if thresholds_df_global is None or thresholds_df_global.empty:
        app.logger.warning("Thresholds not loaded globally. Attempting to load now...")
        if load_thresholds(THRESHOLDS_CSV_PATH) is None: # Try loading, check result
             err_msg = f"Threshold data could not be loaded. Cannot process site {site_id}. Check if '{THRESHOLDS_CSV_PATH}' exists and is valid."
             app.logger.error(err_msg)
             return None, err_msg, "Data Quality Analysis", start_date_str_requested, end_date_str_requested, units, None

    # Now global df should be populated (or loading failed)
    site_thresholds = get_site_thresholds(thresholds_df_global, site_id)
    if site_thresholds is None:
        err_msg = f"Could not find or validate required thresholds for SiteID {site_id} in '{THRESHOLDS_CSV_PATH}'. Plot generation aborted."
        app.logger.error(err_msg)
        name_from_thresh = 'N/A'
        if thresholds_df_global is not None and 'station_name' in thresholds_df_global.columns:
             # Ensure SiteID_str exists before trying to use it for lookup here
             if 'SiteID_str' in thresholds_df_global.columns:
                 site_row = thresholds_df_global[thresholds_df_global['SiteID_str'] == str(site_id)]
                 if not site_row.empty:
                     name_from_thresh = site_row['station_name'].iloc[0]
             else:
                 app.logger.warning("SiteID_str column missing in global thresholds df when getting name fallback.")

        return None, err_msg, name_from_thresh, start_date_str_requested, end_date_str_requested, units, None


    # --- 2. Fetch Data from API ---
    api_end_date_call = datetime.now().strftime('%Y-%m-%d')
    if not is_reset and validate_date(end_date_str_requested):
        api_end_date_call = end_date_str_requested
    elif not is_reset:
        app.logger.warning(f"Invalid/missing end date ('{end_date_str_requested}'). Using today '{api_end_date_call}' for API call.")

    api_start_date_call = "1900-01-01" # API default
    if not is_reset and validate_date(start_date_str_requested):
        api_start_date_call = start_date_str_requested
    elif not is_reset:
        app.logger.warning(f"Invalid/missing start date ('{start_date_str_requested}'). Using default '{api_start_date_call}' for API call.")

    app.logger.info(f"API call parameters - Start: {api_start_date_call}, End: {api_end_date_call}")

    try:
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&start_date={api_start_date_call}&end_date={api_end_date_call}&f=json"
        app.logger.info(f"Fetching data from: {api_url}")
        response = requests.get(api_url, timeout=45) # Increased timeout

        if response.status_code != 200:
            err_msg = f"API Error (Status {response.status_code}) for site {site_id}"
            app.logger.error(err_msg + f". URL: {api_url}. Response: {response.text[:200]}...")
            name_from_thresh = 'N/A'
            if thresholds_df_global is not None and 'station_name' in thresholds_df_global.columns and 'SiteID_str' in thresholds_df_global.columns:
                 site_row = thresholds_df_global[thresholds_df_global['SiteID_str'] == str(site_id)]
                 if not site_row.empty: name_from_thresh = site_row['station_name'].iloc[0]

            return None, err_msg, name_from_thresh, start_date_str_requested, end_date_str_requested, units, site_thresholds

        try:
            data = response.json()
            if not isinstance(data, dict):
                    raise ValueError("API response was not a JSON object (dictionary).")
        except (requests.exceptions.JSONDecodeError, ValueError) as json_err:
            snippet = response.text[:200] if hasattr(response, 'text') else '(No text)'
            err_msg = f"JSON Decode Error for site {site_id}. Error: {json_err}. Response snippet: {snippet}..."
            app.logger.error(err_msg + f" URL: {api_url}")
            name_from_thresh = 'N/A'
            if thresholds_df_global is not None and 'station_name' in thresholds_df_global.columns and 'SiteID_str' in thresholds_df_global.columns:
                 site_row = thresholds_df_global[thresholds_df_global['SiteID_str'] == str(site_id)]
                 if not site_row.empty: name_from_thresh = site_row['station_name'].iloc[0]
            return None, err_msg, name_from_thresh, start_date_str_requested, end_date_str_requested, units, site_thresholds

        metadata_fields = ["station_id", "station_name", "system_name", "units"]
        metadata = {f: data.get(f, "N/A") for f in metadata_fields}
        # Prioritize name from API if available and not 'N/A'
        api_station_name = metadata.get('station_name')
        if api_station_name and api_station_name != 'N/A':
            station_name = api_station_name
        else:
            # Fallback to name from thresholds if API name is missing/NA
            if thresholds_df_global is not None and 'station_name' in thresholds_df_global.columns and 'SiteID_str' in thresholds_df_global.columns:
                 site_row = thresholds_df_global[thresholds_df_global['SiteID_str'] == str(site_id)]
                 if not site_row.empty:
                     station_name = site_row['station_name'].iloc[0]
            if not station_name: # If still no name
                station_name = 'N/A'


        units = metadata.get('units')
        if not units or units == 'N/A':
            units = 'Unknown Units'
            app.logger.warning(f"Units missing or N/A in API response for site {site_id}. Using '{units}'.")
        else:
            app.logger.info(f"Units found in API response: {units}")


        # --- 3. Process Data into DataFrame ---
        if "data" not in data or not isinstance(data["data"], list) or not data["data"]:
            err_msg = f"No 'data' array found or empty in API response for site {site_id}. Station: {station_name}"
            app.logger.warning(err_msg + f" URL: {api_url}")
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds

        try:
            df = pd.DataFrame(data["data"], columns=["date", "value"])
        except Exception as df_err:
            err_msg = f"DataFrame creation error for site {site_id} from API data structure. Error: {df_err}"
            app.logger.error(err_msg, exc_info=True)
            app.logger.error(f"First few items in data['data']: {data.get('data', [])[:3]}")
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds

        if df.empty:
            err_msg = f"DataFrame created but is empty for site {site_id} despite non-empty API data list. Station: {station_name}"
            app.logger.warning(err_msg + f" URL: {api_url}")
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds

        if "date" in df.columns and "value" in df.columns:
            df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
        else:
            err_msg = f"Critical error: DataFrame created but columns 'date' or 'value' are missing. Site {site_id}."
            app.logger.error(err_msg + f" Actual Columns Found: {df.columns.tolist()}")
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df = df.sort_values('Date').reset_index(drop=True)

        if df.empty:
            err_msg = f"No valid dates found in the data after conversion for site {site_id}."
            app.logger.warning(err_msg)
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds

        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')

        # --- 4. Filter Data by Date Range ---
        if df.empty: # Should not happen if previous check passed, but safe check
             min_data_dt = None
             max_data_dt = None
             app.logger.warning("DataFrame became empty before date filtering unexpectedly.")
        else:
             min_data_dt = df['Date'].min()
             max_data_dt = df['Date'].max()
             if pd.notna(min_data_dt) and pd.notna(max_data_dt): # Check if dates are valid
                 app.logger.info(f"Full data range available from API: {min_data_dt:%Y-%m-%d} to {max_data_dt:%Y-%m-%d}")
             else:
                 app.logger.warning("Min or Max date from API data is invalid.")
                 min_data_dt = None # Ensure invalid dates are None
                 max_data_dt = None


        if is_reset or min_data_dt is None or max_data_dt is None:
             start_dt_final = min_data_dt
             end_dt_final = max_data_dt
             if is_reset: app.logger.info(f"Reset requested. Using full data range.")
             if start_dt_final is None or end_dt_final is None: # If no valid data range exists
                 app.logger.warning("No valid date range in fetched data. Cannot filter.")
                 err_msg = f"No date range found in data for site {site_id}."
                 # Return None for fig, but pass existing info back
                 return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units, site_thresholds
             else:
                  app.logger.info(f"Using full range: {start_dt_final:%Y-%m-%d} to {end_dt_final:%Y-%m-%d}")

        else:
            # We have a valid data range and are not resetting
            start_req_dt = validate_date(start_date_str_requested)
            end_req_dt = validate_date(end_date_str_requested)
            # Default to actual data bounds if requested dates are invalid/missing
            if not start_req_dt: start_req_dt = min_data_dt
            if not end_req_dt: end_req_dt = max_data_dt

            # Ensure requested dates don't go outside actual data range
            start_dt_final = max(start_req_dt, min_data_dt)
            end_dt_final = min(end_req_dt, max_data_dt)

            # Check if the resulting range is valid (start <= end)
            if start_dt_final > end_dt_final:
                app.logger.warning(f"Requested/Adjusted range [{start_dt_final:%Y-%m-%d} - {end_dt_final:%Y-%m-%d}] "
                                   f"is invalid or outside available data range [{min_data_dt:%Y-%m-%d} - {max_data_dt:%Y-%m-%d}]. "
                                   f"Plotting full available range instead.")
                start_dt_final = min_data_dt # Fallback to full range
                end_dt_final = max_data_dt
            else:
                app.logger.info(f"Using date range for plot: {start_dt_final:%Y-%m-%d} to {end_dt_final:%Y-%m-%d}")

        # Convert final dates to strings for display and potential use
        actual_start_date_str = start_dt_final.strftime('%Y-%m-%d') if pd.notna(start_dt_final) else ""
        actual_end_date_str = end_dt_final.strftime('%Y-%m-%d') if pd.notna(end_dt_final) else ""


        # Filter the DataFrame using the final determined date range
        # Need to handle cases where start/end might still be NaT if initial data was bad
        if pd.notna(start_dt_final) and pd.notna(end_dt_final):
            df_filtered = df.loc[(df['Date'] >= start_dt_final) & (df['Date'] <= end_dt_final)].copy().reset_index(drop=True)
        else:
             df_filtered = pd.DataFrame() # Empty dataframe if dates are invalid

        if df_filtered.empty:
            err_msg = f"No data available after filtering for site {site_id} in range [{actual_start_date_str} to {actual_end_date_str}]."
            app.logger.warning(err_msg)
            # Still return thresholds, just no plot data
            return None, err_msg, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds

        df = df_filtered # Work with the filtered data from now on
        app.logger.info(f"Processing {len(df)} data points for flagging and plotting.")

        # --- 5. Apply Flagging ---
        df = apply_flagging(df, site_thresholds)

        # --- 6. Create Plot ---
        plot_title = f"Data from {actual_start_date_str} to {actual_end_date_str}"
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['DISCHARGE'], mode='lines',
            line=dict(color='lightgray', width=1.5),
            name='Mean Daily Discharge',
            connectgaps=False, # Don't connect gaps in the base line
            hoverinfo='skip'
        ))

        # Extract thresholds for plotting lines/legends
        min_val_thresh = site_thresholds.get("min_val", float('nan'))
        max_val_thresh = site_thresholds.get("max_val", float('nan'))
        spike_unusual_thresh = site_thresholds.get("spike_unusual", float('nan'))
        repeated_days_thresh = site_thresholds.get("repeated_days", DEFAULT_REPEATED_DAYS)

        # Format thresholds for display in legends/annotations
        formatted_spike_threshold = f"{spike_unusual_thresh:.2f}" if pd.notna(spike_unusual_thresh) else "N/A"
        formatted_max_threshold = f"{max_val_thresh:.2f}" if pd.notna(max_val_thresh) else "N/A"
        formatted_min_threshold = f"{min_val_thresh:.2f}" if pd.notna(min_val_thresh) else "N/A"

        # Define flag types, colors, and legend formats (using f-strings and {{}} for count placeholders)
        flag_plot_info = {
            'FLAG_BELOW_CAPACITY': ('red', 'Below Measuring Capacity (Negative) [{}]'), # {} is fine
            'FLAG_ZERO': ('blue', 'Zero Discharge [{}]'), # {} is fine
            'FLAG_REPEATED': ('green', f'Repeated Value (>= {repeated_days_thresh} days, non-zero) [{{}}]'), # Use {{}}
            'FLAG_GREATER_THAN_MaxValue': ('purple', f'Over Max Capacity ({formatted_max_threshold}) [{{}}]'), # Use {{}}
            'UNUSUAL_SPIKE': ('orange', f"Unusual Spike (RoC > {formatted_spike_threshold}) [{{}}]") # Use {{}}
        }
        # Define hover template for flagged points
        hover_tmpl = (f'<b>Date:</b> %{{x|%Y-%m-%d}}<br>'
                      f'<b>Value:</b> %{{y:.2f}} {units}<br>'
                      f'<b>Flag Type:</b> %{{meta}}' # Use 'meta' for the clean flag name
                      f'<extra></extra>') # Hide extra hover info

        # Add traces for each flag type that has occurred
        for flag_col_name, (color, legend_format) in flag_plot_info.items():
             if flag_col_name in df.columns and df[flag_col_name].any():
                 subset = df.loc[df[flag_col_name]]
                 count = len(subset)
                 # Get the label part *before* formatting count, used for hover meta
                 flag_label_only = legend_format.split('[')[0].strip()
                 fig.add_trace(go.Scatter(
                     x=subset['Date'], y=subset['DISCHARGE'], mode='markers',
                     marker=dict(color=color, size=7, symbol='circle'),
                     name=legend_format.format(count), # Apply count format here for legend
                     meta=flag_label_only, # Pass label without count to meta for hover
                     hovertemplate=hover_tmpl,
                     showlegend=True
                 ))

        # Add threshold lines and gradient buffer if dates are valid
        min_plot_dt, max_plot_dt = df["Date"].min(), df["Date"].max()
        if pd.notna(min_plot_dt) and pd.notna(max_plot_dt):
            plot_date_range = [min_plot_dt, max_plot_dt] # Use actual data bounds

            # Add minimum value threshold line (if not zero)
            if pd.notna(min_val_thresh) and min_val_thresh != 0:
                fig.add_trace(go.Scatter(
                    x=plot_date_range, y=[min_val_thresh, min_val_thresh],
                    mode='lines', line=dict(color="gray", dash="dash", width=1),
                    name=f"Min Value Threshold ({formatted_min_threshold})", hoverinfo='skip'
                ))
            # Add max capacity threshold line and gradient buffer
            if pd.notna(max_val_thresh):
                fig.add_trace(go.Scatter(
                    x=plot_date_range, y=[max_val_thresh, max_val_thresh],
                    mode='lines', line=dict(color="purple", dash="dash", width=1),
                    name=f"Max Capacity Threshold ({formatted_max_threshold})", hoverinfo='skip'
                ))
                # Add gradient buffer only if max capacity is positive
                if max_val_thresh > 0:
                    buffer_width = max_val_thresh * BUFFER_PERCENTAGE
                    dates_for_buffer = df['Date'].tolist()
                    if len(dates_for_buffer) >= 2:
                        app.logger.info(f"Adding gradient buffer around max capacity ({max_val_thresh:.2f}) with width {buffer_width:.2f}")
                        add_gradient_buffer(
                            fig=fig, dates=dates_for_buffer, mean_value=max_val_thresh,
                            buffer=buffer_width, start_color_rgba=BUFFER_START_COLOR_RGBA,
                            end_color_rgba=BUFFER_END_COLOR_RGBA, num_bands=BUFFER_NUM_BANDS
                        )
                    else: app.logger.warning("Not enough data points to draw gradient buffer.")
                else: app.logger.info(f"Max capacity threshold is {max_val_thresh:.2f}, skipping gradient buffer.")
        else:
             app.logger.warning("Could not determine plot date range for threshold lines (min/max plot dates invalid).")


        # --- 6a. Calculate Statistics ---
        stats_text = "Statistics not available"
        discharge_data_numeric = df['DISCHARGE'].dropna() # Exclude NaNs for stats
        if not discharge_data_numeric.empty:
            count_rec = discharge_data_numeric.count()
            mean_discharge = discharge_data_numeric.mean()
            min_discharge = discharge_data_numeric.min()
            max_discharge = discharge_data_numeric.max()
            # Format numbers for display
            count_str = f"{count_rec:,}" if pd.notna(count_rec) else "N/A"
            mean_str = f"{mean_discharge:.2f}" if pd.notna(mean_discharge) else "N/A"
            min_str = f"{min_discharge:.2f}" if pd.notna(min_discharge) else "N/A"
            max_str = f"{max_discharge:.2f}" if pd.notna(max_discharge) else "N/A"
            # Create multi-line string for annotation
            stats_text = (
                f"<b>Statistics ({units}):</b><br>"
                f"--------------------<br>"
                f"Record Count: {count_str}<br>"
                f"Mean Daily: {mean_str}<br>"
                f"Min Value: {min_str}<br>"
                f"Max Value: {max_str}"
            )
            app.logger.info(f"Calculated statistics: Count={count_str}, Mean={mean_str}, Min={min_str}, Max={max_str}")
        else:
            app.logger.warning("No valid numeric discharge data found in the filtered range to calculate statistics.")


        # --- 7. Finalize Plot Layout ---
        fig.update_layout(
            title=dict(text=plot_title, x=0.5, y=0.95, font_size=24), # Centered title
            xaxis=dict(
                title_text="Date", title_font_size=18, tickfont_size=14,
                showline=False, zeroline=True, zerolinewidth=2, zerolinecolor='black'
            ),
            yaxis=dict(
                title_text=units, title_font_size=18, tickfont_size=16,
                showline=False, zeroline=True, zerolinewidth=2, zerolinecolor='black'
            ),
            legend=dict(
                orientation="v", x=1.0, y=1, xanchor="left", yanchor="top", # Top-right outside plot
                title=dict(text="Data Flagging Criteria:", font=dict(size=18)),
                font=dict(size=16), bgcolor='rgba(255,255,255,0.7)' # Semi-transparent background
            ),
            annotations=[ # Add statistics box as an annotation
                go.layout.Annotation(
                    text=stats_text, align='left', showarrow=False,
                    xref='paper', yref='paper', # Position relative to plot area
                    x=1.0, y=0.5, # Position below legend (adjust y as needed)
                    xanchor='left', yanchor='top',
                    bordercolor='black', borderwidth=1, bgcolor='rgba(255,255,255,0.8)',
                    font=dict(size=18)
                )
            ],
            template="plotly_white", # Clean background theme
            margin=dict(t=80, r=400, b=80, l=80), # Adjust margins (esp. right for legend/stats)
            height=700, # Set plot height
            hovermode='closest' # Show hover for nearest point
        )

        app.logger.info(f"Plot generated successfully for {site_id} [{actual_start_date_str} to {actual_end_date_str}]")
        # Return all results including the figure and validated thresholds
        return fig, None, station_name, actual_start_date_str, actual_end_date_str, units, site_thresholds

    # --- Error Handling for Plot Generation ---
    except requests.exceptions.RequestException as e: # Network errors during API call
        err = f"Network error fetching data: {e}"
        app.logger.error(f"API Request failed for site {site_id}: {e}", exc_info=True)
        name = 'N/A' # Try to get station name from thresholds as fallback
        if thresholds_df_global is not None and 'station_name' in thresholds_df_global.columns and 'SiteID_str' in thresholds_df_global.columns:
             site_row = thresholds_df_global[thresholds_df_global['SiteID_str'] == str(site_id)]
             if not site_row.empty: name = site_row['station_name'].iloc[0]
        # Return None for fig, pass back error and other info
        return None, err, name, start_date_str_requested, end_date_str_requested, units, site_thresholds
    except Exception as e: # Catch any other unexpected errors
        err = f"Unexpected error during plot generation process: {type(e).__name__}"
        name = 'N/A' # Fallback station name
        if thresholds_df_global is not None and 'station_name' in thresholds_df_global.columns and 'SiteID_str' in thresholds_df_global.columns:
             site_row = thresholds_df_global[thresholds_df_global['SiteID_str'] == str(site_id)]
             if not site_row.empty: name = site_row['station_name'].iloc[0]

        app.logger.error(f"Plot generation internal error for site {site_id}: {e}", exc_info=True)
        # Try to use determined actual dates if available, otherwise fall back to requested dates
        final_start = actual_start_date_str if 'actual_start_date_str' in locals() and actual_start_date_str else start_date_str_requested
        final_end = actual_end_date_str if 'actual_end_date_str' in locals() and actual_end_date_str else end_date_str_requested
        final_units = units if 'units' in locals() and units != 'Unknown Units' else 'Unknown Units'
        # Return None for fig, pass back error and other info
        return None, err, name, final_start, final_end, final_units, site_thresholds
# --- END generate_plot_for_site Function ---


# --- Flask Route: Update Thresholds (Handles POST request) ---
@app.route('/update_thresholds', methods=['POST'])
def update_thresholds():
    """Handles form submission for updating thresholds in the CSV file."""
    global thresholds_df_global # Allow modification and reload of global DataFrame

    # Get data from submitted form
    site_id = request.form.get('site_id')
    # Get dates from hidden fields to maintain state on redirect
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    # Basic check: Ensure Site ID was submitted
    if not site_id:
        flash("Error: Site ID was missing in the update request.", "error")
        return redirect(url_for('index')) # Redirect to index if no site ID

    # --- Get and Validate New Threshold Values ---
    try:
        # Get values from form, check if they exist
        max_val_str = request.form.get('max_val')
        spike_unusual_str = request.form.get('spike_unusual')
        repeated_days_str = request.form.get('repeated_days')
        if max_val_str is None or spike_unusual_str is None or repeated_days_str is None:
             raise ValueError("Missing one or more threshold values in form submission.")

        # Convert to appropriate types (float, float, int)
        new_max_val = float(max_val_str)
        new_spike_unusual = float(spike_unusual_str)
        new_repeated_days = int(repeated_days_str)

        # Apply domain-specific validation rules
        if new_repeated_days < 2:
             flash("Error: Repeated Value Days must be 2 or greater.", "error")
             return redirect(url_for('show_plot', id=site_id, start_date=start_date, end_date=end_date))
        if new_max_val < 0 or new_spike_unusual < 0:
             flash("Error: Max Capacity and Unusual Spike values cannot be negative.", "error")
             return redirect(url_for('show_plot', id=site_id, start_date=start_date, end_date=end_date))

    except (ValueError, TypeError) as e: # Catch errors during conversion
        app.logger.error(f"Invalid threshold format submitted for SiteID {site_id}: {e}")
        flash(f"Error: Invalid number format submitted for thresholds. Please check values.", "error")
        # Redirect back to plot page without saving
        return redirect(url_for('show_plot', id=site_id, start_date=start_date, end_date=end_date))

    app.logger.info(f"Received threshold update for SiteID {site_id}: Max={new_max_val}, Spike={new_spike_unusual}, Repeated={new_repeated_days}")

    # --- Update the CSV File (with file locking) ---
    f = None # File handle
    lock_acquired = False
    try:
        file_path_str = str(THRESHOLDS_CSV_PATH)
        # --- Acquire Exclusive Lock for Writing ---
        # Check existence before opening to avoid creating file if path is wrong
        if not THRESHOLDS_CSV_PATH.is_file():
             raise FileNotFoundError(f"Threshold file '{file_path_str}' not found during update.")

        # Open in 'r+' mode (read and write)
        f = open(file_path_str, 'r+')
        # Acquire exclusive lock (blocks other readers and writers)
        lock_acquired = acquire_lock(f, fcntl.LOCK_EX if HAS_FCNTL else 0)
        if not lock_acquired:
            app.logger.error(f"Failed to acquire write lock for {file_path_str}. Update aborted.")
            flash("Error: Could not save thresholds. File might be busy. Please try again.", "error")
            if f: f.close() # Close file if lock failed
            return redirect(url_for('show_plot', id=site_id, start_date=start_date, end_date=end_date))

        # --- Read, Modify, Write ---
        temp_df = pd.read_csv(f) # Read current content

        # Find the row index for the site using the original 'SiteID' column name
        site_id_col = "SiteID"
        # Ensure comparison uses string types for robustness against mixed types in CSV
        row_index = temp_df[temp_df[site_id_col].astype(str) == str(site_id)].index

        # Check if the site was found in the DataFrame
        if not row_index.empty:
            idx = row_index[0] # Get the index of the first match
            # Update the values in the DataFrame at the found index
            temp_df.loc[idx, 'Over_Capacity'] = new_max_val
            temp_df.loc[idx, 'Unusual_Spike'] = new_spike_unusual
            # Ensure the Repeated_Days column exists before assigning
            if 'Repeated_Days' not in temp_df.columns:
                temp_df['Repeated_Days'] = DEFAULT_REPEATED_DAYS # Add with default if missing
            temp_df.loc[idx, 'Repeated_Days'] = new_repeated_days

            # --- Write back to the file ---
            f.seek(0) # Go to the beginning of the file
            f.truncate() # Clear the existing content
            temp_df.to_csv(f, index=False) # Write the updated DataFrame back
            f.flush() # Ensure changes are written to the OS buffer
            # Force write from OS buffer to disk if possible (fsync)
            if hasattr(os, 'fsync'): # fsync might not be available on all OS
                 try:
                     os.fsync(f.fileno())
                 except OSError as fsync_err:
                      app.logger.warning(f"os.fsync failed for {file_path_str}: {fsync_err}")

            app.logger.info(f"Successfully updated thresholds for SiteID {site_id} in {file_path_str}")

            # --- Reload Global Thresholds DataFrame ---
            # Call load_thresholds to refresh the global variable with the saved data
            load_thresholds(THRESHOLDS_CSV_PATH)

            # Provide success feedback to the user
            flash(f"Thresholds for Site ID {site_id} updated successfully.", "success")
        else:
            # Site ID was not found in the CSV file
            app.logger.error(f"SiteID {site_id} not found in {file_path_str} during update attempt.")
            flash(f"Error: Site ID {site_id} not found in the thresholds file. Could not update.", "error")

    # --- Error Handling for File Operations ---
    except FileNotFoundError:
        app.logger.error(f"Threshold file '{THRESHOLDS_CSV_PATH}' not found during update.")
        flash(f"Error: Threshold file not found. Cannot save changes.", "error")
    except PermissionError:
        app.logger.error(f"Permission denied when trying to write to '{THRESHOLDS_CSV_PATH}'.")
        flash(f"Error: Permission denied. Cannot save threshold changes.", "error")
    except Exception as e: # Catch any other unexpected errors during file processing
        app.logger.error(f"Unexpected error updating thresholds CSV for {site_id}: {e}", exc_info=True)
        flash(f"Error: An unexpected error occurred while saving thresholds.", "error")
    finally:
        # --- Release Lock and Close File ---
        # Ensure lock release and file closing happens even if errors occurred
        if lock_acquired and f:
            release_lock(f)
        if f and not f.closed:
            f.close()

    # Redirect back to the plot page with the original date range (or updated if needed)
    return redirect(url_for('show_plot', id=site_id, start_date=start_date, end_date=end_date))
# --- END Route /update_thresholds ---


# --- Flask Route: Show Plot (Handles GET request) ---
@app.route('/plot')
def show_plot():
    """Handles requests to display the data quality plot."""
    # Get parameters from URL query string
    site_id = request.args.get('id')
    start_date_req = request.args.get('start_date')
    end_date_req = request.args.get('end_date')
    # Check for reset flag (used by "Full Range" button)
    is_reset = request.args.get('reset') == 'true'

    app.logger.info(f"Request received: id={site_id}, start={start_date_req}, end={end_date_req}, reset={is_reset}")

    # Initialize variables used in template rendering
    current_thresholds_for_template = None # Holds dict for threshold form
    err_msg = None # Holds error messages for display
    fig = None # Holds the generated Plotly figure
    plot_div = None # Holds the HTML representation of the figure
    st_name = "Data Quality Analysis" # Default station name
    status = 200 # Default HTTP status code
    start_proc = None # Start date passed to generate_plot
    end_proc = None # End date passed to generate_plot
    start_render = start_date_req # Start date displayed in the form input
    end_render = end_date_req # End date displayed in the form input
    units_val = 'Unknown Units' # Units for display

    # --- Handle Initial Load or Missing Site ID ---
    if not site_id:
        app.logger.info("No Site ID provided. Rendering initial form.")
        today = datetime.now()
        # Default to last 30 days for initial view when no site ID is given
        def_end = today.strftime('%Y-%m-%d')
        def_start = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        # Render template with default dates and no plot/thresholds
        return render_template_string(HTML_TEMPLATE,
                                      site_id=None,
                                      station_name=st_name,
                                      start_date=def_start,
                                      end_date=def_end,
                                      error=None, plot_div=None, units=units_val,
                                      current_thresholds=None)

    # --- Handle Empty Site ID String ---
    if not site_id.strip():
         app.logger.warning("Empty Site ID provided.")
         err_msg = "Site ID cannot be empty."
         status = 400 # Bad Request
         today = datetime.now() # Provide default dates for form context
         def_end = today.strftime('%Y-%m-%d')
         def_start = (today - timedelta(days=30)).strftime('%Y-%m-%d')
         start_render = start_date_req if start_date_req else def_start
         end_render = end_date_req if end_date_req else def_end
         # Render template with error message
         return render_template_string(HTML_TEMPLATE, site_id=site_id, station_name=st_name, start_date=start_render, end_date=end_render, error=err_msg, plot_div=None, units=units_val, current_thresholds=None), status

    # --- Determine if Reset is Needed (Implicitly if dates are missing) ---
    # If not explicitly reset, but dates are missing, treat as a reset request
    if not is_reset and (not start_date_req or not end_date_req):
        app.logger.info(f"Site ID '{site_id}' provided without full date range. Treating as reset.")
        is_reset = True
        start_date_req = None # Clear requested dates as they are ignored
        end_date_req = None

    # --- Validate Date Range (if not resetting) ---
    if not is_reset:
        start_dt = validate_date(start_date_req)
        end_dt = validate_date(end_date_req)
        if not start_dt or not end_dt:
            # Invalid date format
            err_msg = "Valid Start and End Dates required (YYYY-MM-DD format)."
            status = 400 # Bad Request
            app.logger.warning(f"Invalid date format provided: Start='{start_date_req}', End='{end_date_req}'")
            # Keep potentially invalid dates in the form for user correction
            start_render = start_date_req or ""
            end_render = end_date_req or ""
        elif start_dt > end_dt:
            # Start date is after end date
            err_msg = "Start date cannot be after end date."
            status = 400 # Bad Request
            app.logger.warning(f"Date range error: Start={start_dt:%Y-%m-%d}, End={end_dt:%Y-%m-%d}")
            # Keep dates in the form for user correction
            start_render = start_date_req
            end_render = end_date_req
        else:
            # Dates are valid and in correct order
            start_proc = start_dt.strftime('%Y-%m-%d')
            end_proc = end_dt.strftime('%Y-%m-%d')
            # Update render dates to reflect the validated format
            start_render = start_proc
            end_render = end_proc
            app.logger.info(f"Using specific date range: {start_proc} to {end_proc}")
    else:
        # Reset requested, dates will be determined by generate_plot
        app.logger.info("Reset requested or dates missing. Will use full data range.")
        # start_proc/end_proc remain None
        # start_render/end_render will be updated later by generate_plot result

    # --- Generate Plot (if date validation passed) ---
    if not err_msg: # Proceed only if date validation did not produce an error
        app.logger.info(f"Calling generate_plot_for_site: id={site_id}, start={start_proc}, end={end_proc}, is_reset={is_reset}")
        try:
            # Call the core function to fetch data, apply flags, and create figure
            # It returns the figure, error message (if any), and other metadata including thresholds
            fig, err_func, name_func, final_start, final_end, units_val, current_thresholds_for_template = generate_plot_for_site(
                site_id, start_proc, end_proc, is_reset=is_reset
            )

            # Update render dates based on what generate_plot actually used/determined
            # This handles the 'reset=true' case where dates are derived from data range
            start_render = final_start
            end_render = final_end
            # Update station name if provided by generate_plot
            if name_func and name_func != 'N/A': st_name = name_func

            # Handle errors returned *from* generate_plot
            if err_func:
                err_msg = err_func # Set the error message for display
                # Determine appropriate HTTP status based on the error type
                if "API Error" in err_msg or "Network error" in err_msg: status = 502 # Bad Gateway
                elif "JSON Decode Error" in err_msg: status = 500 # Internal Server Error (bad data format)
                elif "Threshold data could not be loaded" in err_msg: status = 500 # Server config issue
                elif "Could not find or validate" in err_msg: status = 404 # Not Found (bad SiteID or config)
                elif "Unexpected" in err_msg or "Critical error" in err_msg: status = 500 # Generic server error
                elif "No data available" in err_msg or "No date range found" in err_msg: status = 200 # OK, but no data/plot
                else: status = 400 # Default bad request for other functional errors

            app.logger.info(f"Plot generation function complete. Final render dates: {start_render} to {end_render}, Units: {units_val}")

        except Exception as e:
            # Catch unexpected errors *during the call* to generate_plot itself
            app.logger.error(f"Unhandled exception during plot generation call for {site_id}: {e}", exc_info=True)
            err_msg = "Unexpected server error during plot generation."
            status = 500 # Internal Server Error
            # Try to preserve original requested dates if possible
            start_render = start_date_req or ""
            end_render = end_date_req or ""
            units_val = 'Unknown Units'
            current_thresholds_for_template = None # Unlikely to have thresholds if call failed


    # --- Convert Plot Figure to HTML ---
    if fig: # Only if a figure object was successfully created
        try:
            # Generate HTML div for the plot, using CDN for Plotly.js
            plot_div = fig.to_html(
                full_html=False,          # Generate only the div, not full page
                include_plotlyjs='cdn',   # Use Plotly CDN
                config={'displayModeBar': True, 'scrollZoom': True} # Configure Plotly bar
            )
            app.logger.info(f"Plot converted to HTML div for {site_id}.")
        except Exception as plot_e:
            # Handle errors during HTML conversion
            app.logger.error(f"Error converting plot to HTML for {site_id}: {plot_e}", exc_info=True)
            # Append to existing error or set new one
            err_msg = (err_msg + "; Error preparing plot for display.") if err_msg else "Error preparing plot for display."
            status = 500 # Internal Server Error
            plot_div = None # Ensure no partial plot is shown
            fig = None # Clear the figure object


    # --- Handle Cases Where No Plot Was Generated but No Specific Error Was Set ---
    # E.g., generate_plot returned (None, None, ...) without setting err_func
    # Only add generic message if it wasn't already a known non-error (like 'No data') or a server/config error
    if not fig and not err_msg and status not in [200, 500, 502, 404]:
        app.logger.info(f"Final state: No plot generated for {site_id}, but no specific error reported (status={status}). Setting generic message.")
        err_msg = f"No plot generated. This might happen if data exists but all points were filtered out or another condition prevented plotting."
        status = 404 # Treat as "Not Found" if no specific reason given


    # --- Render Final Template ---
    # Ensure dates passed to template are non-None strings
    start_final = start_render if start_render is not None else ""
    end_final = end_render if end_render is not None else ""

    # Log the final set of parameters being passed to the template
    app.logger.debug(f"Rendering template with: site_id={site_id}, name={st_name}, start={start_final}, end={end_final}, error={err_msg is not None}, plot={plot_div is not None}, units={units_val}, thresholds={current_thresholds_for_template is not None}, status={status}")

    # Render the Jinja2 template with all collected data
    return render_template_string(HTML_TEMPLATE,
                                  site_id=site_id,
                                  station_name=st_name,
                                  start_date=start_final,
                                  end_date=end_final,
                                  error=err_msg, # Pass error message string
                                  plot_div=plot_div, # Pass plot HTML div
                                  units=units_val, # Pass units string
                                  current_thresholds=current_thresholds_for_template # Pass thresholds dict
                                  ), status # Return response with appropriate HTTP status code
# --- END Route /plot ---


# --- Flask Route: Index / Root ---
@app.route('/')
def index():
    """Renders the initial page with default date range."""
    today_dt = datetime.now()
    today_str = today_dt.strftime('%Y-%m-%d')
    # Default to last 30 days for the initial index page view
    one_month_ago_str = (today_dt - timedelta(days=30)).strftime('%Y-%m-%d')
    app.logger.info("Rendering index page with default date range (last 30 days).")
    # Render the template with no site ID, default dates, and no plot/thresholds
    return render_template_string(HTML_TEMPLATE,
                                  site_id=None,
                                  station_name="Data Quality Analysis",
                                  start_date=one_month_ago_str,
                                  end_date=today_str,
                                  error=None, plot_div=None, units='Unknown Units',
                                  current_thresholds=None)
# --- END Route / ---


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Pre-run Check: Ensure Thresholds Loaded ---
    # This prevents starting the server if the essential config file is missing/invalid
    if thresholds_df_global is None:
        app.logger.warning("Thresholds not loaded at import time. Attempting load again before starting server.")
        # Try loading one more time
        if load_thresholds(THRESHOLDS_CSV_PATH) is None:
             # If still fails, print fatal error and exit
             print(f"\nFATAL ERROR: Could not load required thresholds from '{THRESHOLDS_CSV_PATH}'. The application cannot run without them.", file=sys.stderr)
             print("Please ensure the file exists at the expected location, is readable, and contains required columns.", file=sys.stderr)
             print(f"Core columns expected: SiteID, {', '.join(CORE_REQUIRED_THRESHOLD_COLS)}", file=sys.stderr)
             print(f"Optional columns expected: {', '.join([c for c in EXPECTED_THRESHOLD_COLS if c not in ['SiteID'] + CORE_REQUIRED_THRESHOLD_COLS])}", file=sys.stderr)
             print("Check application logs for specific errors (e.g., FileNotFoundError, ParserError, PermissionError, missing columns).", file=sys.stderr)
             sys.exit(1) # Exit with a non-zero code indicates an error

    # --- Start Flask Development Server ---
    app.logger.info(f"Starting Flask server on http://127.0.0.1:5000")
    app.logger.info("Flask Debug Mode is ON")
    # Use threaded=True for basic concurrency handling, especially with file I/O
    # Set debug=False for production deployment
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
# --- END Main Execution ---
