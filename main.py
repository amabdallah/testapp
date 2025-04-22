# -*- coding: utf-8 -*-
# --- Imports ---
# (Keep all your imports)
from flask import Flask, render_template_string, request, redirect, url_for, flash, session, jsonify
import logging
import os
import sys
from datetime import datetime, timedelta
import uuid

# --- Import functions from data_handler ---
try:
    from data_handler import (
        load_thresholds,
        generate_plot_for_site,
        update_threshold_in_csv,
        THRESHOLDS_CSV_PATH, # Import path constant
        HAS_FCNTL # Import check result
    )
except ImportError as e:
    print(f"FATAL ERROR: Could not import from data_handler.py. Ensure it exists in the same directory.", file=sys.stderr)
    print(f"Error details: {e}", file=sys.stderr)
    sys.exit(1) # Exit if handler can't be imported

# --- Flask App Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO) # Configure app's logger
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# --- Log fcntl status ---
if not HAS_FCNTL:
    app.logger.warning("fcntl not available. File locking disabled.")
else:
    app.logger.info("fcntl available. File locking enabled.")

# --- Load Thresholds AT STARTUP ---  <<<<<<<<< MOVED HERE
# This code now runs when Gunicorn imports the module
app.logger.info(f"Attempting initial threshold load from: {THRESHOLDS_CSV_PATH}")
if load_thresholds(THRESHOLDS_CSV_PATH, app.logger) is None:
    # Log critical error, but allow app to potentially start to show the error message
    app.logger.critical(f"FATAL: Initial threshold load failed from '{THRESHOLDS_CSV_PATH}'. Application might not function correctly.")
    # Consider if you want the app to completely fail to start here or just log the error.
    # For now, we log and continue, the route handler will show the CRITICAL ERROR message.
else:
    app.logger.info("Initial threshold load successful.")
# --- END Load Thresholds AT STARTUP ---

# --- HTML Template Definition ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
 RENDER
# (Keep your entire HTML template string here)
</html>
"""
# --- END HTML Template ---


# --- Flask Route: Update Thresholds ---
@app.route('/update_thresholds', methods=['POST'])
def update_thresholds():
    # (Keep your route function code here)
    site_id = request.form.get('site_id')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    if not site_id: flash("Error: Site ID missing.", "error"); return redirect(url_for('index'))
    try:
        new_vals = {'max_val': float(request.form['max_val']), 'spike_unusual': float(request.form['spike_unusual']), 'repeated_days': int(request.form['repeated_days'])}
        if new_vals['repeated_days'] < 2: flash("Repeated Days must be >= 2.", "error"); raise ValueError("Repeated days < 2")
        if new_vals['max_val'] < 0 or new_vals['spike_unusual'] < 0: flash("Thresholds cannot be negative.", "error"); raise ValueError("Negative threshold")
    except (ValueError, TypeError, KeyError) as e:
        app.logger.error(f"Invalid threshold format/value for SiteID {site_id}: {e}")
        flash("Error: Invalid number format or value for thresholds.", "error")
        # Redirect back to plot page on error
        return redirect(url_for('show_plot', id=site_id, start_date=start_date, end_date=end_date))

    app.logger.info(f"Updating SiteID {site_id}: {new_vals}")
    success, message = update_threshold_in_csv(site_id, new_vals, app.logger)
    flash(message, "success" if success else "error")
    # Redirect back to plot page after update attempt
    return redirect(url_for('show_plot', id=site_id, start_date=start_date, end_date=end_date))


# --- Flask Route: Show Plot ---
@app.route('/plot')
def show_plot():
    # (Keep your route function code here)
    # Example start:
    site_id = request.args.get('id')
    start_date_req = request.args.get('start_date')
    end_date_req = request.args.get('end_date')
    is_reset = request.args.get('reset') == 'true'
    app.logger.info(f"Request: id={site_id}, start={start_date_req}, end={end_date_req}, reset={is_reset}")
    # ... rest of show_plot function ...

    # Call generate_plot_for_site which now has the initial check
    fig, err_func, name_func, final_start, final_end, units_val, current_thresholds_for_template, stats_dict_for_template = generate_plot_for_site(
        site_id, start_date_req, end_date_req, is_reset=is_reset, logger=app.logger
    )
    # ... rest of show_plot rendering logic using the returned values ...
    # Make sure err_msg handling uses err_func correctly
    err_msg = err_func # Assign the error message from the tuple
    # ... determine status code based on err_msg ...
    status = 500 if err_msg and "CRITICAL ERROR" in err_msg else \
             502 if err_msg and ("API Error" in err_msg or "Network error" in err_msg) else \
             500 if err_msg and ("JSON Decode Error" in err_msg or "Threshold data" in err_msg or "Unexpected" in err_msg) else \
             404 if err_msg and ("Could not find or validate" in err_msg or f"SiteID {site_id} not found" in err_msg or "Error: Threshold data missing" in err_msg) else \
             200 if err_msg and ("No data" in err_msg or "No date range" in err_msg) else \
             400 if err_msg else 200 # Default to 200 if no error message

    # ... handle plot_div conversion if fig exists ...
    plot_div = None
    plot_output_id = None
    if fig:
        try:
            plot_output_id = f"plotly-plot-{uuid.uuid4()}"
            plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': True, 'scrollZoom': True, 'responsive': True}, div_id=plot_output_id)
            app.logger.info(f"Plot converted to HTML div (id='{plot_output_id}')")
        except Exception as plot_e:
             app.logger.error(f"Error converting plot to HTML for {site_id}: {plot_e}", exc_info=True)
             err_msg = (err_msg + "; Error preparing plot.") if err_msg else "Error preparing plot."
             status = 500; plot_div = None; fig = None; plot_output_id = None


    # ... rendering logic ...
    start_final = final_start if final_start is not None else "" # Use final dates from generate_plot
    end_final = final_end if final_end is not None else ""
    return render_template_string(HTML_TEMPLATE,
                                  site_id=site_id, station_name=name_func or "N/A", start_date=start_final, end_date=end_final,
                                  error=err_msg, plot_div=plot_div, units=units_val or 'Unknown Units',
                                  current_thresholds=current_thresholds_for_template,
                                  stats_dict=stats_dict_for_template,
                                  plot_output_id=plot_output_id
                                  ), status


# --- Flask Route: Index / Root ---
@app.route('/')
def index():
    # (Keep your route function code here)
    today=datetime.now(); today_str=today.strftime('%Y-%m-%d'); month_ago=(today-timedelta(days=30)).strftime('%Y-%m-%d')
    app.logger.info("Rendering index page.")
    return render_template_string(HTML_TEMPLATE,
                                  site_id=None, station_name="Data Quality Analysis", start_date=month_ago, end_date=today_str,
                                  error=None, plot_div=None, units='Unknown Units', current_thresholds=None, stats_dict=None,
                                  plot_output_id=None)


# --- Main Execution Block ---
# This block is primarily for local development testing ('python main.py')
# It should NOT contain essential setup logic needed for Gunicorn deployment.
if __name__ == '__main__':
    # The threshold loading is now done *outside* this block.
    # You might still want a check here for direct execution convenience:
    from data_handler import thresholds_df_global # Import the global to check it
    if thresholds_df_global is None:
         print("\nFATAL ERROR: Thresholds failed to load during module import. Cannot start dev server.", file=sys.stderr)
         sys.exit(1)

    app.logger.info(f"Starting Flask development server on http://127.0.0.1:5000")
    # Use Flask's built-in server for local debugging
    app.run(host='127.0.0.1', port=5000, debug=True) # debug=True is useful locally
