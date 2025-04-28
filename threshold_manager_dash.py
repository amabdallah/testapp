# threshold_manager_dash.py (Modified for improved exception logging)
# -*- coding: utf-8 -*-
# --- Imports ---
import pandas as pd
import logging # Logger type hint comes from here
from typing import Dict, Any, Optional, Tuple # Keep this import
import os
from pathlib import Path
import time
import sys
import traceback # Import for potential fallback logging

# File locking import (Unix-specific)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# --- Pandas Option ---
# pd.set_option('future.no_silent_downcasting', True) # Keep commented out unless needed

# --- Constants ---
CORE_REQUIRED_THRESHOLD_COLS = ["Over_Capacity", "Unusual_Spike"]
EXPECTED_THRESHOLD_COLS = [
    "SiteID", "Below_Capacity", "Over_Capacity", "Min_IQR_Upper_Bound_Value",
    "Max_Value_95Perc", "Average_Rate_Of_Change", "Unusual_Change_90th_Perc.",
    "MaxRoC", "Unusual_Spike", "Repeated_values"
]
DEFAULT_REPEATED_THRESHOLD = 4
STATIC_MIN_THRESHOLD = 0 # This defines the 'min_val' returned by get_site_thresholds

# --- Path Definition ---
# Use a dedicated logger for setup issues if main logger isn't ready
setup_logger = logging.getLogger('threshold_setup')
if not setup_logger.hasHandlers(): # Basic setup if not configured elsewhere
     setup_handler = logging.StreamHandler(sys.stderr)
     setup_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     setup_handler.setFormatter(setup_formatter)
     setup_logger.addHandler(setup_handler)
     setup_logger.setLevel(logging.INFO)

THRESHOLDS_CSV_PATH = None # Initialize
try:
    script_dir = Path(__file__).resolve().parent
    primary_path = script_dir / "thresholds.csv"
    fallback_path = Path("thresholds.csv") # Relative path

    if primary_path.is_file():
        THRESHOLDS_CSV_PATH = primary_path
        setup_logger.info(f"Using primary threshold file path: {THRESHOLDS_CSV_PATH}")
    elif fallback_path.is_file():
        THRESHOLDS_CSV_PATH = fallback_path
        setup_logger.warning(f"Threshold file not found at primary '{primary_path}'. Falling back to relative path '{fallback_path}' ({THRESHOLDS_CSV_PATH}).")
    else:
        # Critical failure if neither exists
        setup_logger.critical(f"Threshold file not found at primary path '{primary_path}' or fallback relative path '{fallback_path}'. Thresholds cannot be loaded.")
        # Keep THRESHOLDS_CSV_PATH as None

except NameError:
    # Fallback if __file__ is not defined (e.g., interactive session)
    fallback_path = Path("thresholds.csv")
    setup_logger.warning(f"Could not determine script directory via __file__. Trying relative path: {fallback_path}")
    if fallback_path.is_file():
        THRESHOLDS_CSV_PATH = fallback_path
        setup_logger.info(f"Using relative threshold file path: {THRESHOLDS_CSV_PATH}")
    else:
        setup_logger.critical(f"Relative threshold file path '{fallback_path}' does not point to an existing file. Thresholds cannot be loaded.")
        # Keep THRESHOLDS_CSV_PATH as None
except Exception as path_e:
    setup_logger.critical(f"Unexpected error resolving threshold file path: {path_e}", exc_info=True)
    # Keep THRESHOLDS_CSV_PATH as None

# Add final debug print for resolved path
setup_logger.debug(f"Final THRESHOLDS_CSV_PATH resolved to: {THRESHOLDS_CSV_PATH}")
if isinstance(THRESHOLDS_CSV_PATH, Path):
    setup_logger.debug(f"Does the file exist at that path? {THRESHOLDS_CSV_PATH.is_file()}")
elif THRESHOLDS_CSV_PATH is None:
     setup_logger.debug("THRESHOLDS_CSV_PATH is None after setup attempt.")
else:
    setup_logger.debug(f"THRESHOLDS_CSV_PATH is not a valid Path object (Type: {type(THRESHOLDS_CSV_PATH)}).")


# --- Global Threshold Variable ---
# Stores the loaded thresholds DataFrame in memory
thresholds_df_global: Optional[pd.DataFrame] = None

# --- File Locking Functions ---
def acquire_lock(file_handle, lock_type: int, logger: logging.Logger, timeout: int = 5) -> bool:
    """Attempts to acquire a shared (SH) or exclusive (EX) lock on an open file."""
    if not HAS_FCNTL:
        logger.debug("fcntl not available, skipping file lock acquisition.")
        return True # Assume success if locking is disabled

    start_time = time.time()
    lock_type_str = "Shared" if lock_type == fcntl.LOCK_SH else "Exclusive"

    while time.time() - start_time < timeout:
        try:
            fcntl.flock(file_handle, lock_type | fcntl.LOCK_NB) # Non-blocking attempt
            logger.debug(f"{lock_type_str} lock acquired for {file_handle.name}")
            return True
        except (BlockingIOError, OSError) as e: # BlockingIOError or OSError (errno 11/EAGAIN) indicate lock held
            # Check if it's a known "lock held" error before retrying
            is_lock_held_error = isinstance(e, BlockingIOError) or (isinstance(e, OSError) and e.errno in [11, 13]) # 11=EAGAIN/EWOULDBLOCK, 13=EACCES (permission denied, might indicate lock conflict)
            if not is_lock_held_error:
                 # --- MODIFICATION: Log traceback for unexpected OS errors ---
                logger.error(f"Unexpected OSError ({getattr(e, 'errno', 'N/A')}) acquiring lock: {e}", exc_info=True)
                # Reraise unexpected errors? Or return False? Returning False for now.
                return False
            # If it is a lock held error, wait and retry
            time.sleep(0.1)
        except Exception as e_lock: # Catch any other unexpected exceptions
            # --- MODIFICATION: Ensure traceback is logged ---
            logger.error(f"Unexpected exception acquiring lock: {e_lock}", exc_info=True)
            # Reraise or return False? Returning False to indicate failure.
            return False

    logger.error(f"Could not acquire {lock_type_str} lock on {file_handle.name} within {timeout}s timeout.")
    return False

def release_lock(file_handle, logger: logging.Logger):
    """Releases the lock on the file handle if fcntl is available."""
    if not HAS_FCNTL:
        logger.debug("fcntl not available, skipping lock release.")
        return

    if file_handle and not file_handle.closed:
        try:
            fcntl.flock(file_handle, fcntl.LOCK_UN)
            logger.debug(f"Lock released for {file_handle.name}")
        except Exception as e_release: # Catch any error during release
            # --- MODIFICATION: Ensure traceback is logged ---
            logger.error(f"Error releasing lock for {file_handle.name}", exc_info=True)
    else:
        logger.debug("Skipping lock release: fcntl enabled but file handle invalid or closed.")


# --- Threshold Loading Function ---
def load_thresholds(file_path: Optional[Path], logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Loads thresholds from the CSV file into the global variable thresholds_df_global.
    Uses a shared read lock. Returns the loaded DataFrame or None on failure.
    """
    global thresholds_df_global

    # --- MODIFICATION: Check if file_path is None early ---
    if file_path is None:
        logger.error("Threshold file path is None. Cannot load thresholds.")
        thresholds_df_global = None
        return None
    # --- END MODIFICATION ---

    file_path_str = str(file_path)
    logger.info(f"Attempting to load thresholds from: {file_path_str}")

    f = None
    lock_acquired = False

    try:
        # Check existence before opening
        if not file_path.is_file():
            logger.error(f"Threshold file not found at '{file_path_str}'. Cannot load.")
            thresholds_df_global = None # Ensure global is None if file not found
            return None

        # Open with read mode ('r')
        f = open(file_path_str, 'r', encoding='utf-8') # Specify encoding
        lock_mode = fcntl.LOCK_SH if HAS_FCNTL else 0 # Shared lock for reading
        lock_acquired = acquire_lock(f, lock_mode, logger)

        if not lock_acquired:
            logger.error(f"Failed to acquire read lock for {file_path_str}. Aborting load.")
            thresholds_df_global = None
            return None

        # Read CSV - explicitly set dtype for SiteID
        try:
            thresholds_df = pd.read_csv(f, dtype={"SiteID": str})
            logger.info(f"Read CSV '{file_path_str}' with SiteID explicitly as string dtype.")
        except pd.errors.EmptyDataError:
            logger.warning(f"Threshold file '{file_path_str}' is empty. Proceeding with empty DataFrame.")
            # Ensure columns match expectations if possible, even if empty
            thresholds_df = pd.DataFrame(columns=EXPECTED_THRESHOLD_COLS)
        except Exception as read_err: # Catch any error during pandas read
            # --- MODIFICATION: Ensure traceback is logged ---
            logger.error(f"Failed to read CSV content from locked file '{file_path_str}'", exc_info=True)
            thresholds_df_global = None; return None

        logger.info(f"Successfully read CSV content from '{file_path_str}'. Validating...")
        if thresholds_df.empty:
            logger.warning(f"Threshold file '{file_path_str}' resulted in an empty DataFrame.")
            # Update global state even if empty, validation is skipped
            thresholds_df_global = thresholds_df
            return thresholds_df # Allow empty DataFrame

        # --- Validation ---
        if "SiteID" not in thresholds_df.columns:
            logger.error(f"'SiteID' column missing in '{file_path_str}'. Load failed.")
            thresholds_df_global = None; return None

        # Check for core required columns used for threshold logic
        missing_core = [c for c in CORE_REQUIRED_THRESHOLD_COLS if c not in thresholds_df.columns]
        if missing_core:
            logger.error(f"Missing CORE required threshold columns in '{file_path_str}': {missing_core}. Load failed.")
            thresholds_df_global = None; return None

        # Warn about other expected columns that might be missing but aren't core
        missing_expected = [c for c in EXPECTED_THRESHOLD_COLS if c not in thresholds_df.columns]
        if missing_expected:
            logger.warning(f"Optional/Expected columns missing in '{file_path_str}': {missing_expected}.")

        # Create the SiteID_str column for reliable matching (handles leading/trailing spaces)
        try:
             thresholds_df["SiteID_str"] = thresholds_df["SiteID"].astype(str).str.strip()
             logger.info("Created 'SiteID_str' column for matching.")
        except Exception as e_sid_str:
             logger.error("Failed to create 'SiteID_str' column", exc_info=True)
             thresholds_df_global = None; return None # Fail if this crucial step errors


        logger.info(f"Thresholds loaded and validated successfully from '{file_path_str}'. Shape: {thresholds_df.shape}")
        thresholds_df_global = thresholds_df # Update global variable
        return thresholds_df

    except Exception as e_load: # Catch any other unexpected error during loading
        # --- MODIFICATION: Ensure traceback is logged ---
        logger.error(f"Unexpected error during threshold loading from '{file_path_str}'", exc_info=True)
        thresholds_df_global = None # Ensure global is None on unexpected error
        return None
    finally:
        # Ensure lock is released and file is closed
        if lock_acquired and f:
            release_lock(f, logger)
        if f and not f.closed:
            try:
                f.close()
                logger.debug(f"File handle closed for {file_path_str} in load_thresholds.")
            except Exception as close_e:
                 # --- MODIFICATION: Log traceback if close fails ---
                 logger.error(f"Error closing file handle for {file_path_str}", exc_info=True)


# --- Get Site Thresholds Function ---
def get_site_thresholds(site_id: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Retrieves and validates thresholds *currently used by the application*
    for a specific site ID from the global DataFrame.

    Returns a dictionary with keys: 'min_val', 'max_val', 'spike_unusual', 'repeated_values_threshold'
    or None if the site is not found or thresholds are invalid.
    """
    logger.info(f"Getting thresholds for SiteID '{site_id}' from loaded global data...")

    # --- Access global safely ---
    # Use a local copy to avoid potential race conditions if reload happens mid-function
    current_thresholds_df = thresholds_df_global
    # ---

    if current_thresholds_df is None:
        logger.error("Cannot get site thresholds: Global thresholds DataFrame is None (not loaded successfully).")
        return None
    if current_thresholds_df.empty:
        logger.warning("Cannot get site thresholds: Global thresholds DataFrame is empty.")
        return None
    if "SiteID_str" not in current_thresholds_df.columns:
        logger.error("Cannot get site thresholds: 'SiteID_str' column missing from global thresholds DataFrame.")
        return None

    site_id_str = str(site_id).strip() # Ensure input is treated as stripped string for matching
    logger.debug(f"Searching for SiteID_str: '{site_id_str}'")

    try: # Wrap the main logic
        # Perform the filtering using the pre-calculated SiteID_str column
        site_row_df = current_thresholds_df[current_thresholds_df["SiteID_str"] == site_id_str]

        if site_row_df.empty:
            logger.warning(f"SiteID '{site_id_str}' not found in the loaded thresholds file.")
            # Log available IDs only at DEBUG level to avoid clutter
            if logger.isEnabledFor(logging.DEBUG):
                available_ids = current_thresholds_df['SiteID_str'].unique().tolist()
                logger.debug(f"Available SiteID_str values: {available_ids[:20]}..." if len(available_ids) > 20 else available_ids)
            return None

        # If multiple rows match (shouldn't happen with unique SiteIDs), use the first
        if len(site_row_df) > 1:
            logger.warning(f"Multiple rows found for SiteID '{site_id_str}'. Using the first row found.")
        row_data = site_row_df.iloc[0] # Get the first matching row as a Series

        # --- Initialize and Validate ---
        validated_thresholds = {
            "min_val": float(STATIC_MIN_THRESHOLD) # Static minimum threshold
        }
        missing_values = []
        validation_errors = []

        # Map and validate core numeric columns
        mapping = {
            "Over_Capacity": "max_val",
            "Unusual_Spike": "spike_unusual"
        }
        for csv_col, app_key in mapping.items():
            raw_value = row_data.get(csv_col)
            if pd.isna(raw_value): # Check specifically for NaN or None
                missing_values.append(f"'{csv_col}' (missing/NaN)")
                continue # Skip to next column if value is missing

            # Try converting to numeric
            numeric_value = pd.to_numeric(raw_value, errors='coerce')
            if pd.isna(numeric_value):
                missing_values.append(f"'{csv_col}' (value '{raw_value}' is not numeric)")
            else:
                validated_thresholds[app_key] = float(numeric_value) # Store as float

        # Validate 'Repeated_values' -> 'repeated_values_threshold'
        repeated_col_name = "Repeated_values"
        raw_repeated = row_data.get(repeated_col_name)
        if pd.isna(raw_repeated): # Check specifically for NaN or None
            logger.warning(f"Site '{site_id_str}': '{repeated_col_name}' missing/NaN in CSV. Using default: {DEFAULT_REPEATED_THRESHOLD}")
            validated_thresholds["repeated_values_threshold"] = int(DEFAULT_REPEATED_THRESHOLD)
        else:
            try:
                # Ensure it's treated as numeric first, then convert to int
                rep_numeric = pd.to_numeric(raw_repeated, errors='raise') # Raise error if not numeric
                repeated_int = int(rep_numeric) # Convert to integer
                if repeated_int >= 2:
                    validated_thresholds["repeated_values_threshold"] = repeated_int
                else:
                    validation_errors.append(f"'{repeated_col_name}' ({repeated_int}) is < 2. Using default {DEFAULT_REPEATED_THRESHOLD}.")
                    validated_thresholds["repeated_values_threshold"] = int(DEFAULT_REPEATED_THRESHOLD)
            except (ValueError, TypeError) as e_rep: # Catch conversion errors
                # --- MODIFICATION: Log traceback for conversion errors ---
                logger.warning(f"Site '{site_id_str}': Error converting '{repeated_col_name}' (value '{raw_repeated}') to int. Using default {DEFAULT_REPEATED_THRESHOLD}.", exc_info=True)
                validation_errors.append(f"'{repeated_col_name}' (value '{raw_repeated}') is not a valid integer. Using default {DEFAULT_REPEATED_THRESHOLD}.")
                validated_thresholds["repeated_values_threshold"] = int(DEFAULT_REPEATED_THRESHOLD)

        # --- Final Checks ---
        # Check if core values needed by the app were successfully populated
        if "max_val" not in validated_thresholds or "spike_unusual" not in validated_thresholds:
             # This implies one of the core columns was missing/invalid
             error_summary = ', '.join(missing_values) if missing_values else "Internal validation error."
             logger.error(f"Failed to retrieve essential thresholds for SiteID '{site_id_str}': {error_summary}")
             return None

        # Log any non-critical validation warnings
        if validation_errors:
             for e in validation_errors: logger.warning(f"SiteID '{site_id_str}' Threshold Validation: {e}")

        logger.info(f"Thresholds retrieved and validated successfully for SiteID '{site_id_str}': {validated_thresholds}")
        return validated_thresholds

    except Exception as e_get: # Catch any unexpected error during retrieval/validation
        # --- MODIFICATION: Ensure traceback is logged ---
        logger.error(f"Unexpected error getting thresholds for SiteID '{site_id_str}'", exc_info=True)
        return None


# --- Update Threshold in CSV Function ---
def update_threshold_in_csv(site_id: str, new_thresholds: Dict[str, Any], logger: logging.Logger) -> Tuple[bool, str]:
    """
    Updates thresholds for a specific site ID in the CSV file.
    Uses an exclusive write lock and reloads the global DataFrame on success.

    Args:
        site_id: The site ID (string) to update.
        new_thresholds: Dict containing keys 'max_val', 'spike_unusual', 'repeated_values_threshold'.
        logger: The logger instance.

    Returns:
        Tuple (bool, str): (success status, message).
    """
    global thresholds_df_global

    # --- MODIFICATION: Check file path validity early ---
    if THRESHOLDS_CSV_PATH is None:
        logger.error("Cannot update threshold CSV: File path was not resolved successfully during startup.")
        return False, "Error: Threshold file path not configured."
    # --- END MODIFICATION ---

    f = None
    lock_acquired = False
    success_status = False
    return_message = "An unknown issue occurred during threshold update."
    file_path_str = str(THRESHOLDS_CSV_PATH)

    try:
        # --- MODIFICATION: Add explicit check for Path object ---
        if not isinstance(THRESHOLDS_CSV_PATH, Path):
             raise TypeError(f"THRESHOLDS_CSV_PATH is not a valid Path object: {type(THRESHOLDS_CSV_PATH)}")
        # --- END MODIFICATION ---

        if not THRESHOLDS_CSV_PATH.is_file():
            # Raise specific error if file doesn't exist at update time
            raise FileNotFoundError(f"Threshold file '{file_path_str}' not found for update.")

        # Open with read/write mode ('r+'), ensure encoding
        f = open(file_path_str, 'r+', encoding='utf-8')
        lock_mode = fcntl.LOCK_EX if HAS_FCNTL else 0 # Exclusive lock for writing
        lock_acquired = acquire_lock(f, lock_mode, logger)

        if not lock_acquired:
            msg = f"Failed to acquire write lock for {file_path_str}. Update aborted."
            logger.error(msg)
            success_status = False
            return_message = "Error: Could not save thresholds (file busy or timeout)."
            # Close file if open before returning
            if f and not f.closed: f.close()
            return success_status, return_message # Return immediately
        else:
            # --- Read, Modify, Write (while holding lock) ---
            try:
                # Read the current content
                temp_df = pd.read_csv(f, dtype={"SiteID": str}) # Read with SiteID as string
            except Exception as read_update_err:
                 # --- MODIFICATION: Log traceback on read failure during update ---
                 logger.error(f"Failed to read CSV for update from locked file '{file_path_str}'", exc_info=True)
                 raise read_update_err # Reraise to be caught by outer block

            site_id_col = "SiteID" # The actual column name in the CSV
            if site_id_col not in temp_df.columns:
                raise ValueError(f"'{site_id_col}' column not found in {file_path_str}")

            # Create temporary column for matching (handle whitespace)
            temp_df["SiteID_match"] = temp_df[site_id_col].astype(str).str.strip()
            site_id_str_to_match = str(site_id).strip()

            # Find the row index matching the site ID
            row_index = temp_df[temp_df["SiteID_match"] == site_id_str_to_match].index
            temp_df = temp_df.drop(columns=["SiteID_match"]) # Drop temporary matching column

            if not row_index.empty:
                idx = row_index[0] # Use the first match
                logger.info(f"Updating thresholds for SiteID '{site_id_str_to_match}' at index {idx} in CSV...")

                # Map application keys back to CSV column names
                update_mapping = {
                    'max_val': 'Over_Capacity',
                    'spike_unusual': 'Unusual_Spike',
                    'repeated_values_threshold': 'Repeated_values'
                }

                updated = False
                for app_key, csv_col in update_mapping.items():
                    if app_key in new_thresholds:
                        # Ensure column exists
                        if csv_col not in temp_df.columns:
                            logger.warning(f"Column '{csv_col}' missing in CSV during update for site {site_id}. Adding it.")
                            temp_df[csv_col] = pd.NA
                        temp_df.loc[idx, csv_col] = new_thresholds[app_key]
                        updated = True
                    else:
                        logger.warning(f"Key '{app_key}' not found in new_thresholds dict for site {site_id}. Column '{csv_col}' not updated.")

                if updated:
                    # Overwrite the file content
                    f.seek(0) # Go to the beginning
                    f.truncate() # Clear existing content
                    temp_df.to_csv(f, index=False, encoding='utf-8') # Write updated DataFrame
                    f.flush() # Ensure buffer is written to OS

                    # Force write to disk if possible (OS dependent)
                    if hasattr(os, 'fsync'):
                        try:
                            os.fsync(f.fileno())
                            logger.debug(f"os.fsync completed for {file_path_str}")
                        except OSError as fsync_err:
                             # Log fsync errors as warnings, don't fail the whole update
                            logger.warning(f"os.fsync failed for {file_path_str}: {fsync_err}")

                    logger.info(f"Thresholds updated and file saved successfully for site '{site_id_str_to_match}' in {file_path_str}")
                    success_status = True
                    return_message = f"Thresholds for Site ID {site_id} updated successfully."
                else:
                    logger.warning(f"No values were updated for SiteID '{site_id_str_to_match}'. Check input dictionary keys.")
                    success_status = False # Didn't fail, but didn't update
                    return_message = f"No thresholds updated for Site ID {site_id} (check input)."

            else: # Site ID not found
                logger.error(f"SiteID '{site_id_str_to_match}' not found in {file_path_str} for update.")
                success_status = False
                return_message = f"Error: Site ID {site_id} not found in threshold file."

    except FileNotFoundError as e_fnf:
        # --- MODIFICATION: Ensure traceback is logged ---
        logger.error(f"Update failed: Threshold file not found at '{file_path_str}'", exc_info=True)
        success_status=False; return_message = "Error: Threshold file not found during update attempt."
    except PermissionError as e_perm:
        # --- MODIFICATION: Ensure traceback is logged ---
        logger.error(f"Permission denied attempting to write to: '{file_path_str}'", exc_info=True)
        success_status=False; return_message = "Error: Permission denied saving thresholds."
    except ValueError as e_val: # Catches issues like missing SiteID column or numeric conversion issues
        # Traceback already included from logger call below
        logger.error(f"Value error during CSV update process for site '{site_id}': {e_val}", exc_info=True)
        success_status=False; return_message = f"Error processing threshold file data: {e_val}"
    except Exception as e_upd: # Catch any other unexpected error
        # --- MODIFICATION: Ensure traceback is logged ---
        logger.error(f"Unexpected error updating CSV for site '{site_id}'", exc_info=True)
        success_status=False; return_message = "An unexpected server error occurred while saving thresholds."
    finally:
        # Always release lock and close file in finally block
        if lock_acquired and f:
            release_lock(f, logger)
        if f and not f.closed:
            try:
                f.close()
                logger.debug("File closed in finally block during update.")
            except Exception as close_e_upd:
                 # --- MODIFICATION: Log traceback if close fails ---
                 logger.error(f"Error closing file handle during update for {file_path_str}", exc_info=True)


    # --- Reload Global DataFrame After Successful Update ---
    if success_status:
        logger.info("Reloading thresholds into global DataFrame after successful update.")
        load_result = load_thresholds(THRESHOLDS_CSV_PATH, logger) # Use the module-level path
        if load_result is None:
             logger.error("CRITICAL: Threshold update succeeded, but failed to reload the updated data into memory!")
             # Keep success_status True but modify message
             return_message += " (Warning: Failed to reload data after save, UI may be outdated)."
        else:
             logger.info("Global thresholds successfully reloaded after update.")

    return success_status, return_message
# --- END threshold_manager_dash.py ---