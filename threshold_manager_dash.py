# threshold_manager_dash.py
# -*- coding: utf-8 -*-
# --- Imports ---
import pandas as pd
import logging # Logger type hint comes from here
from typing import Dict, Any, Optional, Tuple # Keep this import
import os
from pathlib import Path
import time
import sys

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
try:
    script_dir = Path(__file__).resolve().parent
    csv_filename = "thresholds.csv"
    THRESHOLDS_CSV_PATH = script_dir / csv_filename
    if not THRESHOLDS_CSV_PATH.is_file():
        THRESHOLDS_CSV_PATH_FALLBACK = Path(csv_filename)
        if THRESHOLDS_CSV_PATH_FALLBACK.is_file():
            THRESHOLDS_CSV_PATH = THRESHOLDS_CSV_PATH_FALLBACK
            print(f"WARNING: Threshold file not found at primary '{script_dir / csv_filename}'. Falling back to relative path '{csv_filename}' ({THRESHOLDS_CSV_PATH}).", file=sys.stderr)
        else:
            print(f"ERROR: Threshold file not found at primary path '{script_dir / csv_filename}' or fallback relative path '{csv_filename}'.", file=sys.stderr)
    else:
        print(f"INFO: Using primary threshold file path: {THRESHOLDS_CSV_PATH}", file=sys.stderr)
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive session)
    csv_filename = "thresholds.csv"
    THRESHOLDS_CSV_PATH = Path(csv_filename)
    print(f"WARNING: Could not determine script directory via __file__. Using relative path: {THRESHOLDS_CSV_PATH}", file=sys.stderr)
    if not THRESHOLDS_CSV_PATH.is_file():
        print(f"ERROR: Relative threshold file path '{THRESHOLDS_CSV_PATH}' does not point to an existing file.", file=sys.stderr)

# Add debug print for final path resolution
print(f"DEBUG: Final THRESHOLDS_CSV_PATH resolved to: {THRESHOLDS_CSV_PATH}", file=sys.stderr)
if isinstance(THRESHOLDS_CSV_PATH, Path):
    print(f"DEBUG: Does the file exist at that path? {THRESHOLDS_CSV_PATH.is_file()}", file=sys.stderr)
else:
     print(f"DEBUG: THRESHOLDS_CSV_PATH is not a valid Path object.", file=sys.stderr)
sys.stderr.flush()


# --- Global Threshold Variable ---
# Stores the loaded thresholds DataFrame in memory
thresholds_df_global: Optional[pd.DataFrame] = None

# --- File Locking Functions ---
def acquire_lock(file_handle, lock_type: int, logger: logging.Logger, timeout: int = 5) -> bool:
    """Attempts to acquire a shared (SH) or exclusive (EX) lock on an open file."""
    if not HAS_FCNTL: logger.debug("fcntl not available, skipping file lock acquisition."); return True
    start_time = time.time(); lock_type_str = "Shared" if lock_type == fcntl.LOCK_SH else "Exclusive"
    while time.time() - start_time < timeout:
        try:
            fcntl.flock(file_handle, lock_type | fcntl.LOCK_NB) # Non-blocking attempt
            logger.debug(f"{lock_type_str} lock acquired for {file_handle.name}")
            return True
        except (BlockingIOError, OSError) as e: # BlockingIOError or OSError (errno 11/EAGAIN) indicate lock held
            # Only retry if it's a known "lock held" error
            if isinstance(e, OSError) and e.errno not in [11, 13]: # 11=EAGAIN, 13=EACCES (sometimes seen?)
                 logger.error(f"Unexpected OSError ({e.errno}) acquiring lock: {e}", exc_info=True); raise
            # Wait a bit before retrying
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Unexpected exception acquiring lock: {e}", exc_info=True); raise
    logger.error(f"Could not acquire {lock_type_str} lock on {file_handle.name} within {timeout}s timeout."); return False

def release_lock(file_handle, logger: logging.Logger):
    """Releases the lock on the file handle if fcntl is available."""
    if HAS_FCNTL and file_handle and not file_handle.closed:
        try:
            fcntl.flock(file_handle, fcntl.LOCK_UN)
            logger.debug(f"Lock released for {file_handle.name}")
        except Exception as e:
            logger.error(f"Error releasing lock for {file_handle.name}: {e}", exc_info=True)
    elif not HAS_FCNTL:
        logger.debug("fcntl not available, skipping lock release.")


# --- Threshold Loading Function ---
def load_thresholds(file_path: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Loads thresholds from the CSV file into the global variable thresholds_df_global.
    Uses a shared read lock. Returns the loaded DataFrame or None on failure.
    """
    global thresholds_df_global
    file_path_str = str(file_path)
    logger.info(f"Attempting to load thresholds from: {file_path_str}")
    f = None; lock_acquired = False
    try:
        # Check existence before opening
        if not file_path.is_file():
            logger.error(f"Threshold file not found at '{file_path_str}'. Cannot load.")
            thresholds_df_global = None # Ensure global is None if file not found
            return None # Return None, don't raise FileNotFoundError here directly

        # Open with read mode ('r')
        f = open(file_path_str, 'r')
        lock_mode = fcntl.LOCK_SH if HAS_FCNTL else 0 # Shared lock for reading
        lock_acquired = acquire_lock(f, lock_mode, logger)
        if not lock_acquired:
            logger.error(f"Failed to acquire read lock for {file_path_str}. Aborting load.")
            thresholds_df_global = None
            return None

        # Read CSV - explicitly set dtype for SiteID
        try:
            thresholds_df = pd.read_csv(f, dtype={"SiteID": str})
            logger.info("Read CSV with SiteID explicitly as string dtype.")
        except pd.errors.EmptyDataError:
            logger.warning(f"Threshold file '{file_path_str}' is empty. Proceeding with empty DataFrame.")
            thresholds_df = pd.DataFrame(columns=EXPECTED_THRESHOLD_COLS) # Create empty df with expected cols if possible
        except Exception as read_err:
            logger.error(f"Failed to read CSV content from locked file '{file_path_str}': {read_err}", exc_info=True)
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
        thresholds_df["SiteID_str"] = thresholds_df["SiteID"].astype(str).str.strip()
        logger.info("Created 'SiteID_str' column for matching.")

        logger.info(f"Thresholds loaded and validated successfully from '{file_path_str}'. Shape: {thresholds_df.shape}")
        thresholds_df_global = thresholds_df # Update global variable
        return thresholds_df

    # Removed FileNotFoundError catch block as the check is done earlier
    except Exception as e:
        logger.error(f"Unexpected error during threshold loading from '{file_path_str}': {e}", exc_info=True)
        thresholds_df_global = None # Ensure global is None on unexpected error
        return None
    finally:
        # Ensure lock is released and file is closed
        if lock_acquired and f:
            release_lock(f, logger)
        if f and not f.closed:
            f.close()
            logger.debug(f"File handle closed for {file_path_str} in load_thresholds.")

# --- Get Site Thresholds Function ---
def get_site_thresholds(site_id: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Retrieves and validates thresholds *currently used by the application*
    for a specific site ID from the global DataFrame.

    Returns a dictionary with keys: 'min_val', 'max_val', 'spike_unusual', 'repeated_values_threshold'
    or None if the site is not found or thresholds are invalid.
    """
    logger.info(f"Getting thresholds for SiteID '{site_id}' from loaded global data...")
    if thresholds_df_global is None:
        logger.error("Cannot get site thresholds: Global thresholds DataFrame is None (not loaded successfully).")
        return None
    if thresholds_df_global.empty:
        logger.warning("Cannot get site thresholds: Global thresholds DataFrame is empty.")
        return None
    if "SiteID_str" not in thresholds_df_global.columns:
        logger.error("Cannot get site thresholds: 'SiteID_str' column missing from global thresholds DataFrame.")
        return None

    site_id_str = str(site_id).strip() # Ensure input is treated as stripped string for matching
    logger.debug(f"Searching for SiteID_str: '{site_id_str}'")

    # Perform the filtering using the pre-calculated SiteID_str column
    site_row_df = thresholds_df_global[thresholds_df_global["SiteID_str"] == site_id_str]

    if site_row_df.empty:
        logger.warning(f"SiteID '{site_id_str}' not found in the loaded thresholds file.")
        available_ids = thresholds_df_global['SiteID_str'].unique().tolist()
        logger.debug(f"Available SiteID_str values: {available_ids[:20]}..." if len(available_ids) > 20 else available_ids) # Show some available IDs for debugging
        return None

    try:
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
        # ('Over_Capacity' -> 'max_val', 'Unusual_Spike' -> 'spike_unusual')
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
            except (ValueError, TypeError):
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

    except Exception as e:
        logger.error(f"Unexpected error during threshold validation for SiteID '{site_id_str}': {e}", exc_info=True)
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
    f = None; lock_acquired = False; success_status = False
    return_message = "An unknown issue occurred during threshold update."
    file_path_str = str(THRESHOLDS_CSV_PATH)

    try:
        if not THRESHOLDS_CSV_PATH.is_file():
            raise FileNotFoundError(f"Threshold file '{file_path_str}' not found for update.")

        # Open with read/write mode ('r+')
        f = open(file_path_str, 'r+')
        lock_mode = fcntl.LOCK_EX if HAS_FCNTL else 0 # Exclusive lock for writing
        lock_acquired = acquire_lock(f, lock_mode, logger)

        if not lock_acquired:
            msg = f"Failed to acquire write lock for {file_path_str}. Update aborted."
            logger.error(msg)
            success_status = False
            return_message = "Error: Could not save thresholds (file busy or timeout)."
        else:
            # --- Read, Modify, Write ---
            # Read the current content while holding the lock
            temp_df = pd.read_csv(f, dtype={"SiteID": str}) # Read with SiteID as string
            site_id_col = "SiteID" # The actual column name in the CSV

            if site_id_col not in temp_df.columns:
                raise ValueError(f"'{site_id_col}' column not found in {file_path_str}")

            # Create a temporary column for matching (handle potential whitespace)
            temp_df["SiteID_match"] = temp_df[site_id_col].astype(str).str.strip()
            site_id_str_to_match = str(site_id).strip()

            # Find the row index matching the site ID
            row_index = temp_df[temp_df["SiteID_match"] == site_id_str_to_match].index
            temp_df = temp_df.drop(columns=["SiteID_match"]) # Drop temporary matching column

            if not row_index.empty:
                idx = row_index[0] # Use the first match if multiple (shouldn't happen)
                logger.info(f"Updating thresholds for SiteID '{site_id_str_to_match}' at index {idx} in CSV...")

                # Map application keys back to CSV column names
                update_mapping = {
                    'max_val': 'Over_Capacity',
                    'spike_unusual': 'Unusual_Spike',
                    'repeated_values_threshold': 'Repeated_values'
                }

                # Update the DataFrame at the found index
                updated = False
                for app_key, csv_col in update_mapping.items():
                    if app_key in new_thresholds:
                         # Ensure the column exists in the DataFrame before assigning
                         if csv_col not in temp_df.columns:
                              logger.warning(f"Column '{csv_col}' missing in CSV during update for site {site_id}. Adding it.")
                              temp_df[csv_col] = pd.NA # Add column with missing values initially
                         temp_df.loc[idx, csv_col] = new_thresholds[app_key]
                         updated = True
                    else:
                         logger.warning(f"Key '{app_key}' not found in new_thresholds dict for site {site_id}. Column '{csv_col}' not updated.")

                if updated:
                    # Overwrite the file content
                    f.seek(0) # Go to the beginning of the file
                    f.truncate() # Clear existing content
                    temp_df.to_csv(f, index=False) # Write updated DataFrame
                    f.flush() # Ensure buffer is written to OS

                    # Force write to disk if possible (OS dependent)
                    if hasattr(os, 'fsync'):
                        try:
                            os.fsync(f.fileno())
                            logger.debug(f"os.fsync completed for {file_path_str}")
                        except OSError as fsync_err:
                            logger.warning(f"os.fsync failed for {file_path_str}: {fsync_err}")

                    logger.info(f"Thresholds updated and file saved successfully for site '{site_id_str_to_match}' in {file_path_str}")
                    success_status = True
                    return_message = f"Thresholds for Site ID {site_id} updated successfully."
                else:
                    logger.warning(f"No values were updated for SiteID '{site_id_str_to_match}'. Check input dictionary keys.")
                    success_status = False # Technically didn't fail, but didn't update either
                    return_message = f"No thresholds updated for Site ID {site_id} (check input)."

            else:
                # Site ID not found in the file
                logger.error(f"SiteID '{site_id_str_to_match}' not found in {file_path_str} for update.")
                success_status = False
                return_message = f"Error: Site ID {site_id} not found in threshold file."

    except FileNotFoundError as e:
        logger.error(f"Update failed: {e}")
        success_status=False; return_message = "Error: Threshold file not found during update attempt."
    except PermissionError:
        logger.error(f"Permission denied attempting to write to: '{file_path_str}'.")
        success_status=False; return_message = "Error: Permission denied saving thresholds."
    except ValueError as e: # Catches issues like missing SiteID column or numeric conversion issues
        logger.error(f"Value error during CSV update process for site '{site_id}': {e}", exc_info=True)
        success_status=False; return_message = f"Error processing threshold file data: {e}"
    except Exception as e:
        logger.error(f"Unexpected error updating CSV for site '{site_id}': {e}", exc_info=True)
        success_status=False; return_message = "An unexpected server error occurred while saving thresholds."
    finally:
        # Always release lock and close file
        if lock_acquired and f:
            release_lock(f, logger)
        if f and not f.closed:
            f.close()
            logger.debug("File closed in finally block during update.")

    # --- Reload Global DataFrame After Successful Update ---
    # Crucial step to ensure the application uses the latest values
    if success_status:
        logger.info("Reloading thresholds into global DataFrame after successful update.")
        load_result = load_thresholds(THRESHOLDS_CSV_PATH, logger)
        if load_result is None:
             logger.error("CRITICAL: Threshold update succeeded, but failed to reload the updated data into memory!")
             # Keep success_status True but modify message
             return_message += " (Warning: Failed to reload data after save, UI may be outdated)."
        else:
             logger.info("Global thresholds successfully reloaded after update.")

    return success_status, return_message
# --- END threshold_manager.py ---