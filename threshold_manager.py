# threshold_manager.py
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
pd.set_option('future.no_silent_downcasting', True)

# --- Constants ---
# Core thresholds currently *used* by the flagging logic
CORE_REQUIRED_THRESHOLD_COLS = ["Over_Capacity", "Unusual_Spike"]
# Reflects the actual columns expected in the CSV file based on user input
EXPECTED_THRESHOLD_COLS = [
    "SiteID", "Below_Capacity", "Over_Capacity", "Min_IQR_Upper_Bound_Value",
    "Max_Value_95Perc", "Average_Rate_Of_Change", "Unusual_Change_90th_Perc.",
    "MaxRoC", "Unusual_Spike"
]
# DEFAULT_REPEATED_DAYS = 4 # No longer used as Repeated_Days is not in CSV
STATIC_MIN_THRESHOLD = 0

# --- Path Definition ---
# (Path definition logic remains the same)
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
            # Set path to None if not found anywhere? Or let it fail later? Let it fail later.
    else:
        print(f"INFO: Using primary threshold file path: {THRESHOLDS_CSV_PATH}", file=sys.stderr)
except NameError:
    csv_filename = "thresholds.csv"
    THRESHOLDS_CSV_PATH = Path(csv_filename)
    print(f"WARNING: Could not determine script directory via __file__. Using relative path: {THRESHOLDS_CSV_PATH}", file=sys.stderr)
    if not THRESHOLDS_CSV_PATH.is_file():
        print(f"ERROR: Relative threshold file path '{THRESHOLDS_CSV_PATH}' does not point to an existing file.", file=sys.stderr)

print(f"DEBUG: Final THRESHOLDS_CSV_PATH resolved to: {THRESHOLDS_CSV_PATH}", file=sys.stderr)
if isinstance(THRESHOLDS_CSV_PATH, Path): # Check if it's a Path object before calling is_file()
    print(f"DEBUG: Does the file exist at that path? {THRESHOLDS_CSV_PATH.is_file()}", file=sys.stderr)
else:
     print(f"DEBUG: THRESHOLDS_CSV_PATH is not a valid Path object.", file=sys.stderr)
sys.stderr.flush()


# --- Global Threshold Variable ---
thresholds_df_global: Optional[pd.DataFrame] = None

# --- File Locking Functions ---
# (acquire_lock and release_lock functions remain the same)
def acquire_lock(file_handle, lock_type: int, logger: logging.Logger, timeout: int = 5) -> bool:
    """Attempts to acquire a lock on an open file handle."""
    if not HAS_FCNTL:
        logger.debug("fcntl not available, skipping file lock.")
        return True # Assume success if fcntl is not available

    start_time = time.time()
    lock_type_str = "Shared" if lock_type == fcntl.LOCK_SH else "Exclusive"
    while time.time() - start_time < timeout:
        try:
            fcntl.flock(file_handle, lock_type | fcntl.LOCK_NB) # LOCK_NB = non-blocking
            logger.debug(f"{lock_type_str} lock acquired for {file_handle.name}")
            return True
        except (BlockingIOError, OSError) as e:
            if isinstance(e, OSError) and e.errno not in [11, 13]: # EAGAIN/EWOULDBLOCK, EACCES
                 logger.error(f"Unexpected OSError ({e.errno}) acquiring lock: {e}", exc_info=True)
                 raise # Reraise unexpected errors
            time.sleep(0.1)
        except Exception as e:
             logger.error(f"Unexpected exception acquiring lock: {e}", exc_info=True)
             raise # Reraise unexpected errors

    logger.error(f"Could not acquire {lock_type_str} lock on {file_handle.name} within {timeout}s.")
    return False

def release_lock(file_handle, logger: logging.Logger):
    """Releases a lock on an open file handle."""
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
    """Loads thresholds from the CSV file with shared read lock."""
    global thresholds_df_global
    file_path_str = str(file_path)
    logger.info(f"Attempting to load thresholds from: {file_path_str}")

    f = None
    lock_acquired = False
    try:
        if not file_path.is_file():
            logger.error(f"Threshold file not found at '{file_path_str}' during load attempt.")
            raise FileNotFoundError(f"File not found at '{file_path_str}'")

        f = open(file_path_str, 'r')
        lock_mode = fcntl.LOCK_SH if HAS_FCNTL else 0
        lock_acquired = acquire_lock(f, lock_mode, logger)
        if not lock_acquired:
            logger.error(f"Failed to acquire read lock for {file_path_str}. Aborting load.")
            thresholds_df_global = None # Explicitly set global to None on failure
            return None

        thresholds_df = pd.read_csv(f)
        logger.info(f"Successfully read CSV content from '{file_path_str}'. Validating columns...")

        # --- Validation ---
        if thresholds_df.empty:
             logger.warning(f"Threshold file '{file_path_str}' was loaded but is empty (no data rows).")
             # Allow empty file to load, but get_site_thresholds will return None later.
             # Set global correctly.
             thresholds_df["SiteID_str"] = thresholds_df["SiteID"].astype(str) # Add if SiteID exists, even if empty
             thresholds_df_global = thresholds_df
             return thresholds_df # Return the empty DataFrame

        missing_core = [c for c in CORE_REQUIRED_THRESHOLD_COLS if c not in thresholds_df.columns]
        if missing_core:
            logger.error(f"Missing CORE required columns used by application logic in '{file_path_str}': {missing_core}")
            thresholds_df_global = None
            return None # Fail if core columns used by logic are missing

        if "SiteID" not in thresholds_df.columns:
            logger.error(f"'SiteID' column is missing in '{file_path_str}'. Cannot process thresholds.")
            thresholds_df_global = None
            return None # Fail if SiteID is missing

        # Check for all *expected* columns based on user list (informational)
        missing_expected = [c for c in EXPECTED_THRESHOLD_COLS if c not in thresholds_df.columns]
        if missing_expected:
             logger.warning(f"Columns expected based on configuration but missing in '{file_path_str}': {missing_expected}. Proceeding with available columns.")

        # Remove station_name check/addition
        # if 'station_name' not in thresholds_df.columns:
        #     logger.warning(f"'station_name' column missing in '{file_path_str}'. Adding 'N/A' default.")
        #     thresholds_df['station_name'] = 'N/A'

        # Ensure SiteID is string type for consistent matching
        thresholds_df["SiteID_str"] = thresholds_df["SiteID"].astype(str)

        logger.info(f"Thresholds loaded and validated successfully from '{file_path_str}'. Shape: {thresholds_df.shape}")
        thresholds_df_global = thresholds_df # Update global variable
        return thresholds_df # Return the loaded DataFrame

    except FileNotFoundError:
        thresholds_df_global = None
        return None
    except pd.errors.EmptyDataError:
         logger.error(f"Error loading thresholds from '{file_path_str}': File is empty or unreadable by pandas.")
         thresholds_df_global = None
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading thresholds from '{file_path_str}': {e}", exc_info=True)
        thresholds_df_global = None
        return None
    finally:
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
    """
    logger.info(f"Getting thresholds for SiteID {site_id} from loaded global data...")

    if thresholds_df_global is None: # Check None first
        logger.error("Global thresholds DataFrame is None (load may have failed). Cannot get site thresholds.")
        return None
    if thresholds_df_global.empty: # Then check empty
        logger.warning("Global thresholds DataFrame is empty (no data rows). Cannot get site thresholds.")
        return None

    if "SiteID_str" not in thresholds_df_global.columns:
         logger.error("'SiteID_str' column missing from loaded global thresholds. Load may have failed validation.")
         return None

    site_row = thresholds_df_global[thresholds_df_global["SiteID_str"] == str(site_id)]

    if site_row.empty:
        logger.warning(f"SiteID {site_id} not found in the loaded thresholds file.")
        return None

    try:
        row_data = site_row.iloc[0]
        validated_thresholds = {"min_val": float(STATIC_MIN_THRESHOLD)} # Start with static min
        missing_values = []
        validation_errors = [] # Keep for potential future use

        # Validate CORE required numeric columns used by flagging
        for col in CORE_REQUIRED_THRESHOLD_COLS: # Only loop through Over_Capacity, Unusual_Spike
            raw_value = row_data.get(col)
            if raw_value is None or pd.isna(raw_value):
                missing_values.append(f"'{col}' (is missing or NaN)")
                continue

            numeric_value = pd.to_numeric(raw_value, errors='coerce')
            if pd.isna(numeric_value):
                missing_values.append(f"'{col}' (value '{raw_value}' is not numeric)")
            else:
                if col == "Over_Capacity":
                    validated_thresholds["max_val"] = float(numeric_value)
                elif col == "Unusual_Spike":
                    validated_thresholds["spike_unusual"] = float(numeric_value)

        # Remove validation for 'Repeated_Days' as it's not in the CSV
        # raw_repeated = row_data.get("Repeated_Days")
        # ... (removed logic) ...

        # --- Final Checks ---
        if missing_values:
            logger.error(f"Missing or invalid CORE thresholds required by application logic for SiteID {site_id}: {', '.join(missing_values)}")
            return None # Cannot proceed without core thresholds

        # Ensure the essential keys were successfully populated
        if "max_val" not in validated_thresholds or "spike_unusual" not in validated_thresholds:
             logger.error(f"Internal error: Failed to populate required threshold keys ('max_val', 'spike_unusual') for SiteID {site_id} even after processing.")
             return None

        # Remove station name retrieval from CSV data
        # validated_thresholds["station_name"] = row_data.get('station_name', 'N/A')

        logger.info(f"Thresholds required by application retrieved successfully for SiteID {site_id}: {validated_thresholds}")
        return validated_thresholds

    except Exception as e:
        logger.error(f"Unexpected error during threshold validation for SiteID {site_id}: {e}", exc_info=True)
        return None

# --- Update Threshold in CSV Function ---
def update_threshold_in_csv(site_id: str, new_thresholds: Dict[str, Any], logger: logging.Logger) -> Tuple[bool, str]:
    """
    Updates thresholds for a specific site ID in the CSV file with exclusive write lock.
    NOTE: This currently only updates columns used by the get_site_thresholds logic.
    """
    global thresholds_df_global # To reload after successful update

    f = None
    lock_acquired = False
    success_status = False
    return_message = "An unknown issue occurred during threshold update."

    try:
        file_path_str = str(THRESHOLDS_CSV_PATH)
        if not THRESHOLDS_CSV_PATH.is_file():
            raise FileNotFoundError(f"Threshold file '{file_path_str}' not found for update.")

        f = open(file_path_str, 'r+')
        lock_mode = fcntl.LOCK_EX if HAS_FCNTL else 0
        lock_acquired = acquire_lock(f, lock_mode, logger)

        if not lock_acquired:
            msg = f"Failed to acquire write lock for {file_path_str}. Update aborted."
            logger.error(msg)
            success_status = False
            return_message = "Error: Could not save thresholds (file may be busy or locked)."
        else:
            temp_df = pd.read_csv(f)
            site_id_col = "SiteID"

            if site_id_col not in temp_df.columns:
                raise ValueError(f"'{site_id_col}' column not found in the threshold file: {file_path_str}")

            row_index = temp_df[temp_df[site_id_col].astype(str) == str(site_id)].index

            if not row_index.empty:
                idx = row_index[0]
                logger.info(f"Updating thresholds for SiteID {site_id} at index {idx} in {file_path_str}...")

                # --- Perform Updates ---
                # Ensure target columns exist before assigning
                # Only update columns currently used by get_site_thresholds
                cols_to_update = ['Over_Capacity', 'Unusual_Spike']
                for col in cols_to_update:
                    if col not in temp_df.columns:
                        logger.warning(f"Column '{col}' missing in CSV during update. Adding it.")
                        temp_df[col] = pd.Series(dtype='object') # Add with flexible type

                temp_df.loc[idx, 'Over_Capacity'] = new_thresholds['max_val']
                temp_df.loc[idx, 'Unusual_Spike'] = new_thresholds['spike_unusual']

                # Remove Repeated_Days update
                # if 'Repeated_Days' not in temp_df.columns: temp_df['Repeated_Days'] = DEFAULT_REPEATED_DAYS # No default needed
                # temp_df.loc[idx, 'Repeated_Days'] = new_thresholds['repeated_days']

                # --- Write Back to File ---
                f.seek(0); f.truncate(); temp_df.to_csv(f, index=False); f.flush()
                if hasattr(os, 'fsync'):
                    try: os.fsync(f.fileno()); logger.debug(f"os.fsync completed: {file_path_str}")
                    except OSError as fsync_err: logger.warning(f"os.fsync failed: {fsync_err}")

                logger.info(f"Thresholds updated and saved ok: {file_path_str}"); success_status = True; return_message = f"Thresholds for Site ID {site_id} updated."

            else:
                logger.error(f"SiteID {site_id} not found in {file_path_str} for update."); success_status = False; return_message = f"Error: Site ID {site_id} not found."

    except FileNotFoundError as e: logger.error(f"Update failed: {e}"); success_status=False; return_message = "Error: Threshold file not found during update."
    except PermissionError: logger.error(f"Permission denied writing to: '{THRESHOLDS_CSV_PATH}'."); success_status=False; return_message = "Error: Permission denied saving thresholds."
    except ValueError as e: logger.error(f"Value error during CSV update for {site_id}: {e}", exc_info=True); success_status=False; return_message = f"Error processing threshold file data: {e}"
    except Exception as e: logger.error(f"Unexpected error updating CSV {site_id}: {e}", exc_info=True); success_status=False; return_message = "Unexpected server error saving thresholds."
    finally:
        if lock_acquired and f: release_lock(f, logger)
        if f and not f.closed: f.close(); logger.debug("File closed in finally block during update.")

    if success_status and return_message.startswith("Thresholds for Site ID"):
        logger.info("Reloading thresholds into global DataFrame after successful update.")
        load_thresholds(THRESHOLDS_CSV_PATH, logger)

    return success_status, return_message
# --- END threshold_manager.py ---