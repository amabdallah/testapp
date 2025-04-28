# main_dash_app.py
# -*- coding: utf-8 -*-
# --- Imports ---
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, no_update, ctx
import dash_bootstrap_components as dbc

import logging
import os
import sys
from datetime import datetime, timedelta, date # Added date
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from typing import Dict, Any, Tuple, Optional, List
from urllib.parse import parse_qs, urlparse

# --- Import functions --- (Keep existing imports)
# ... (rest of your imports for threshold_manager_dash and plot_table_generator) ...
try:
    # Make sure you have renamed the file to threshold_manager_dash.py
    from threshold_manager_dash import (
        load_thresholds,
        update_threshold_in_csv,
        get_site_thresholds,
        THRESHOLDS_CSV_PATH,
        HAS_FCNTL,
        DEFAULT_REPEATED_THRESHOLD
    )
    print("INFO: Successfully imported from threshold_manager_dash.py", file=sys.stderr)
except ImportError as e:
    print(f"FATAL ERROR: Could not import required items from threshold_manager_dash.py. Error: {e}", file=sys.stderr)
    # ... (fallback code) ...
    def load_thresholds(path, logger): logger.error(f"Threshold load failed: {path} not found."); return None
    def update_threshold_in_csv(site_id, new_vals, logger): logger.error("Threshold update unavailable."); return False, "Threshold update unavailable."
    def get_site_thresholds(site_id, logger): logger.warning(f"Thresholds unavailable for site {site_id}."); return {}
    THRESHOLDS_CSV_PATH = "thresholds_placeholder.csv"
    HAS_FCNTL = False
    DEFAULT_REPEATED_THRESHOLD = 3


# Import plot generation and flagging functions
try:
    # Make sure you have renamed the file to plot_table_generator.py
    from plot_table_generator import (
        generate_plot_for_site,
        apply_flagging
    )
    print("INFO: Successfully imported from plot_table_generator.py", file=sys.stderr)
except ImportError as e:
    print(f"FATAL ERROR: Could not import required items from plot_table_generator.py. Error: {e}", file=sys.stderr)
    # ... (fallback code) ...
    def generate_plot_for_site(site_id, start, end, reset, logger, thresholds_override=None):
        logger.error("Plot generation unavailable.")
        fig = go.Figure().update_layout(title=f"Error: Plot generator unavailable for {site_id}")
        return fig, pd.DataFrame(), "Plot generator unavailable", site_id, start, end, "?", thresholds_override or {}, None
    def apply_flagging(df, thresholds, logger):
        logger.warning("Flagging unavailable.")
        if 'FLAGGED' not in df.columns: df['FLAGGED'] = False
        return df

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger('dash_app')

# --- Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# --- Log fcntl status ---
if not HAS_FCNTL: logger.warning("fcntl module not available. File locking DISABLED.")
else: logger.info("fcntl module available. File locking ENABLED.")

# --- Load Thresholds AT STARTUP ---
# ... (keep existing threshold loading code) ...
logger.info(f"Attempting initial threshold load from: {THRESHOLDS_CSV_PATH}")
if 'threshold_manager_dash' not in sys.modules:
    logger.warning(f"Skipping initial threshold load as threshold_manager_dash module was not found.")
elif load_thresholds(THRESHOLDS_CSV_PATH, logger) is None:
    logger.critical(f"CRITICAL STARTUP WARNING: Initial threshold load failed from '{THRESHOLDS_CSV_PATH}'.")
else:
    logger.info("Initial threshold load attempt complete.")


# --- Helper Function: create_dash_data_table ---
# ... (keep existing create_dash_data_table function) ...
def create_dash_data_table(df: pd.DataFrame, units: str, table_id: str, logger: logging.Logger) -> Any:
    """Creates a Dash DataTable component or an Alert if df is invalid."""
    logger.info(f"Creating Dash DataTable with id='{table_id}'...")
    if df is None or df.empty:
        logger.warning("Input DataFrame is empty or None. Skipping DataTable creation.")
        return dbc.Alert("No data available to display in the table.", color="info", className="mt-3")

    # Ensure Date column is datetime for formatting
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        logger.debug("Converting 'Date' column to datetime objects for table creation.")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            logger.warning("Some 'Date' values could not be converted to datetime.")
    elif 'Date' not in df.columns:
        logger.error("Missing 'Date' column in DataFrame for table creation.")
        return dbc.Alert("Data error: Missing 'Date' column.", color="danger", className="mt-3")


    # Identify which flag columns actually exist in the DataFrame
    possible_flag_cols = ['FLAG_LESS_THAN_Min._Value', 'FLAG_ZERO', 'FLAG_REPEATED', 'FLAG_GREATER_THAN_MaxValue', 'UNUSUAL_SPIKE', 'FLAG_BELOW_CAPACITY', 'FLAGGED']
    existing_flag_cols = [col for col in possible_flag_cols if col in df.columns]
    logger.debug(f"Flag columns found for table: {existing_flag_cols}")

    # Prepare data for the table display
    table_data = pd.DataFrame(index=df.index) # Preserve original index if needed

    # Use existing Date column (already validated)
    table_data['display_date'] = df['Date'].dt.strftime('%Y-%m-%d')

    if 'DISCHARGE' in df.columns:
        table_data['Discharge'] = df['DISCHARGE']
    else:
        logger.warning("Missing 'DISCHARGE' column for table.")
        table_data['Discharge'] = [np.nan] * len(df)

    # Use .get with default Series to handle potentially missing columns gracefully
    table_data['Review Status'] = df.get('ReviewStatus', pd.Series(['Unknown'] * len(df), index=df.index))
    # Ensure Qualifiers column exists before checking for notna
    if 'Qualifiers' in df.columns:
        table_data['Qualified'] = df['Qualifiers'].notna().map({True: 'Yes', False: 'No'})
    else:
        table_data['Qualified'] = pd.Series(['No'] * len(df), index=df.index) # Default if no Qualifiers column

    table_data['FLAGGED'] = df.get('FLAGGED', pd.Series([False] * len(df), index=df.index))

    # Generate 'Active Flags' string
    active_flags_list = []
    flag_map = {
        'FLAG_LESS_THAN_Min._Value': 'Below Min',
        'FLAG_ZERO': 'Zero',
        'FLAG_REPEATED': 'Repeated',
        'FLAG_GREATER_THAN_MaxValue': 'Above Max',
        'UNUSUAL_SPIKE': 'Spike',
        'FLAG_BELOW_CAPACITY': 'Below Capacity'
    }
    # Only check flags that exist in the DataFrame
    flag_cols_to_check = [col for col in flag_map if col in existing_flag_cols]

    if flag_cols_to_check:
        for index, row in df.iterrows(): # Iterate safely
            try:
                # Check boolean flags correctly
                active = [flag_map[col] for col in flag_cols_to_check if pd.notna(row.get(col)) and row.get(col) == True]
                active_flags_list.append(', '.join(active) if active else 'None')
            except Exception as e:
                logger.error(f"Error processing flags for table display row index {index}: {e}")
                active_flags_list.append('Error')
    else: # No flag columns found in data
        active_flags_list = ['N/A'] * len(df)

    # Handle potential length mismatch if errors occurred
    if len(active_flags_list) != len(df):
        logger.error("Length mismatch creating Active Flags column. Padding with 'Error'.")
        active_flags_list.extend(['Error'] * (len(df) - len(active_flags_list)))

    table_data['Active Flags'] = active_flags_list

    # Define table columns structure
    columns = [
        {"name": "Date", "id": "display_date", "editable": False},
        {"name": f"Discharge ({units})", "id": "Discharge", "type": "numeric", "format": {'specifier': '.2f'}, "editable": True},
        {"name": "Review Status", "id": "Review Status", "editable": False},
        {"name": "Qualified", "id": "Qualified", "editable": False},
        {"name": "Active Flags", "id": "Active Flags", "editable": False}
    ]
    # Convert prepared data to dictionary format for DataTable
    data_records = table_data.to_dict('records')

    # Create the DataTable component
    dash_table_component = dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=data_records,
        editable=True, # Only Discharge column is effectively editable via 'columns' config
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_selectable=False,
        row_deletable=False,
        page_action="native",
        page_current=0,
        page_size=25, # Keep page_size to set initial size
        style_table={'overflowX': 'auto', 'minWidth': '100%'},
        style_cell={
            'textAlign': 'left',
            'padding': '5px',
            'minWidth': '100px',
            'width': '150px',
            'maxWidth': '200px',
            'whiteSpace': 'normal',
            'height': 'auto',
            'fontFamily': 'sans-serif',
            'fontSize': '0.9rem'
        },
        style_header={
            'backgroundColor': 'paleturquoise',
            'fontWeight': 'bold',
            'border': '1px solid grey'
        },
        style_data={'border': '1px solid lightgrey'},
        style_data_conditional=[
            {'if': {'filter_query': '{FLAGGED} = True'}, 'backgroundColor': '#ffcccb'}, # Check FLAGGED column from data
            {'if': {'column_editable': True}, 'backgroundColor': '#f0f8ff', 'border': '1px solid blue'} # Highlight editable column
        ],
        persistence=True,
        persistence_type='session', # or 'local' or 'memory'
        persisted_props=['page_current', 'sort_by', 'filter_query'],
    )
    logger.info(f"Dash DataTable '{table_id}' created successfully with {len(data_records)} rows.")
    return dash_table_component

# --- Helper Function/Constants for Multi-Add Modal ---
NUM_INPUT_ROWS = 15 # Provide 15 blank rows for pasting
INPUT_TABLE_COLUMNS = [
    {"name": "Date (YYYY-MM-DD)", "id": "Date", "editable": True, "type": "text"}, # Use text for pasting flexibility, validate later
    {"name": "Discharge", "id": "Discharge", "editable": True, "type": "numeric"},
    {"name": "Qualifier (Optional)", "id": "Qualifier", "editable": True, "type": "text", "presentation": "input"} # 'input' allows empty strings easily
]

def get_initial_input_table_data(num_rows: int = NUM_INPUT_ROWS) -> List[Dict[str, Any]]:
    """Generates the initial empty data structure for the input DataTable."""
    return [{'Date': None, 'Discharge': None, 'Qualifier': None} for _ in range(num_rows)]


# --- Dash App Layout ---
app.layout = dbc.Container([
    # Stores and Location
    # ... (keep existing stores and location) ...
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='data-store'),
    dcc.Store(id='site-info-store'),
    dcc.Store(id='thresholds-store'),
    dcc.Store(id='clicked-point-store'),


    # Header and Disclaimer
    # ... (keep existing header) ...
    html.H3("Disclaimer: App for internal use, testing, and demonstration purposes", style={'color': 'red', 'textAlign': 'center'}),
    html.H1(id='main-title', children="Data Quality Analysis", style={'textAlign': 'center'}),
    html.Hr(),


    # Notification Area
    # ... (keep existing notification area) ...
    dbc.Row(dbc.Col(html.Div(id='notification-area'), width=12)),


    # --- Control Card ---
    # ... (keep existing control card, including the 'Add Data' button) ...
    dbc.Card(dbc.CardBody([
        dbc.Row([
            # Site ID Input
            dbc.Col([
                dbc.Label("Site ID:", html_for="site-id-input", className="fw-bold"),
                dbc.Input(id="site-id-input", type="text", placeholder="Enter Site ID", required=True, persistence=True, persistence_type='session')
            ], md=2),
            # Date Pickers
            dbc.Col([
                dbc.Label("Start Date:", html_for="start-date-picker", className="fw-bold"),
                dcc.DatePickerSingle(id='start-date-picker', display_format='YYYY-MM-DD', persistence=True, persistence_type='session')
            ], md=2),
            dbc.Col([
                dbc.Label("End Date:", html_for="end-date-picker", className="fw-bold"),
                dcc.DatePickerSingle(id='end-date-picker', display_format='YYYY-MM-DD', persistence=True, persistence_type='session')
            ], md=2),
            # Action Buttons
            dbc.Col([
                dbc.Button("Update Plot", id="update-button", color="primary", className="me-1 mt-4"),
                dbc.Button("Reset Range", id="reset-button", color="secondary", outline=True, className="me-1 mt-4"),
                # --- UPDATED BUTTONS ---
                dbc.Button("Record a measurement  Data", id="open-enter-data-modal-button", color="info", outline=True, className="me-1 mt-4", n_clicks=0),
                dbc.Button("Add Data", id="open-add-multiple-modal-button", color="success", outline=True, className="mt-4", n_clicks=0) # Keep Button
                # --- END UPDATED BUTTONS ---
            ], md=4, className="d-flex align-items-end flex-wrap"), # Added flex-wrap for responsiveness
            # Quick Date Selection
            dbc.Col([
                dbc.Label("Quick Dates:", className="fw-bold d-block"),
                dbc.ButtonGroup([
                    dbc.Button("Last Year", id="quick-year-button", outline=True, color="info", size="sm"),
                    dbc.Button("Last Month", id="quick-month-button", outline=True, color="info", size="sm")
                ], className="mt-2")
            ], md=2, className="text-center"),
        ], align="start", className="mb-3"),
    ]), className="mb-3 shadow-sm"),

    # --- Thresholds & Stats Row ---
    # ... (keep existing thresholds & stats row) ...
     dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Adjust QC Thresholds", className="card-title"),
            html.Div(id='threshold-form-content', children=html.P("Load data to view/edit thresholds."))
        ]), className="mb-3 shadow-sm"), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Statistics", className="card-title"),
            html.Div(id="statistics-display", children="Load data to view statistics.")
        ]), className="mb-3 shadow-sm"), md=6),
    ]),

    # --- Main Plot ---
    # ... (keep existing plot row) ...
    dbc.Row(dbc.Col(dcc.Graph(id='main-plot', config={'scrollZoom': True}), width=12)),
    html.Hr(),


    # --- Data Table Section (Collapsible) ---
    # ... (keep existing main data table section) ...
     dbc.Row([
        dbc.Col([
            # Heading and Toggle Button
            dbc.Row([
                dbc.Col(html.H4("Data Table"), width="auto"),
                dbc.Col(dbc.Button("Show/Hide Table", id="toggle-table-button", color="secondary", outline=True, size="sm", n_clicks=0), width="auto")
            ], align="center", className="mt-3 mb-2"),
            # Collapsible Content
            dbc.Collapse(
                id="table-collapse",
                is_open=True, # Start visible
                children=[
                    html.P("(Discharge column is editable)", className="small text-muted"),
                    dbc.Alert(id='table-edit-status', children="Click 'Save Changes' below after editing.", color="info", is_open=False, dismissable=True),
                    html.Div(id='table-container', children=dbc.Alert("Load data to view table.", color="secondary")),
                    dbc.Button("Save Changes", id="save-button", color="success", className="mt-2", n_clicks=0, disabled=True),
                    html.Div(id="save-status", className="mt-1")
                ]
            )
        ], width=12)
    ]),


    # --- Modals ---
    # QC Action Modal
    # ... (keep existing QC modal) ...
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Quality Control Action")),
        dbc.ModalBody("Point details...", id="qc-modal-body"),
        dbc.ModalFooter([
            dbc.Button("Approve", id="qc-approve-button", color="success", className="ms-1", n_clicks=0),
            dbc.Button("Interpolate", id="qc-interpolate-button", color="warning", className="ms-1", n_clicks=0),
            dbc.Button("Delete (Set to NaN)", id="qc-delete-button", color="danger", className="ms-1", n_clicks=0),
            dbc.Button("Close", id="qc-close-button", color="secondary", className="ms-auto", n_clicks=0)
        ])
    ], id="qc-action-modal", is_open=False, centered=True),

    # Enter Data Modal (Single Entry)
    # ... (keep existing single entry modal) ...
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Enter New Measurement")),
        dbc.ModalBody([
            dbc.Alert(id="enter-data-modal-alert", color="danger", is_open=False, duration=4000),
            dbc.Row([
                dbc.Label("Date:", width=2),
                dbc.Col(dcc.DatePickerSingle(id='enter-date-picker', display_format='YYYY-MM-DD', date=date.today().isoformat()), width=10)
            ], className="mb-3"),
            dbc.Row([
                dbc.Label("Discharge:", width=2),
                dbc.Col(dbc.Input(id='enter-discharge-input', type='number', placeholder="Enter value", step="any"), width=10)
            ], className="mb-3"),
                dbc.Row([
                dbc.Label("Qualifier:", width=2),
                dbc.Col(dbc.Input(id='enter-qualifier-input', type='text', placeholder="Optional: e.g., Manual Entry, Ice", value="Manual Entry"), width=10)
            ], className="mb-3"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Submit Measurement", id="submit-enter-data-button", color="primary", n_clicks=0),
            dbc.Button("Cancel", id="cancel-enter-data-button", color="secondary", className="ms-auto", n_clicks=0)
        ])
    ], id="enter-data-modal", is_open=False, centered=True),

    # --- MODIFIED MODAL: Add Multiple Data Points via Table ---
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Add Multiple Measurements (Paste from Spreadsheet)")),
        dbc.ModalBody([
            # Feedback area inside this modal
            dbc.Alert(id="add-multiple-data-modal-alert", color="danger", is_open=False, duration=5000),
            # Instructions
            html.P([
                "Paste data from a spreadsheet (up to ",
                f"{NUM_INPUT_ROWS} rows) into the table below. ",
                html.Strong("Required columns: Date (YYYY-MM-DD), Discharge."),
                " Qualifier is optional."]),
            # Input DataTable
            dash_table.DataTable(
                id='add-data-input-table',
                columns=INPUT_TABLE_COLUMNS,
                data=get_initial_input_table_data(), # Initialize with blank rows
                editable=True,
                row_deletable=False, # Keep simple for now, user can clear rows
                style_table={'overflowX': 'auto', 'maxHeight': '50vh', 'overflowY': 'auto'}, # Make table scrollable
                 style_cell={
                    'textAlign': 'left',
                    'padding': '5px',
                    'minWidth': '100px',
                    'width': '150px',
                    'maxWidth': '200px',
                 },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                 },
                 style_data_conditional=[ # Highlight editable
                    {'if': {'column_editable': True}, 'backgroundColor': 'rgb(240, 248, 255)'}
                 ]
            )
        ]),
        dbc.ModalFooter([
            dbc.Button("Submit Added Data", id="submit-multiple-data-button", color="primary", n_clicks=0),
            dbc.Button("Cancel", id="cancel-multiple-data-button", color="secondary", className="ms-auto", n_clicks=0)
        ])
    ], id="add-multiple-data-modal", is_open=False, centered=True, size="lg"), # Use large size

], fluid=True)

# --- Callbacks ---

# --- Existing Callbacks (set_initial_values_from_url, update_data_and_plots, toggle_table_collapse, build_threshold_form, update_site_thresholds, handle_table_edit, save_data, display_click_data, handle_qc_action, update_url_on_data_change, open_enter_data_modal, handle_submit_new_data, cancel_enter_data_modal) ---
# ... (Keep ALL previously existing callbacks AS THEY ARE) ...
# Example placeholder for existing callbacks:
@callback(
    [Output('site-id-input', 'value'), Output('start-date-picker', 'date'), Output('end-date-picker', 'date')],
    [Input('url', 'href')],
    prevent_initial_call=False # Run on initial load
)
def set_initial_values_from_url(href):
    # ... (implementation as before) ...
    try:
        today_local = datetime.now().date()
    except Exception as e:
        logger.error(f"Could not get current date: {e}. Using fixed fallback.")
        today_local = date(2023, 1, 1)
    default_end_date = today_local
    default_start_date = today_local - timedelta(days=30)
    initial_site_id = None
    start_date = default_start_date
    end_date = default_end_date
    if href:
        try:
            parsed_url = urlparse(href)
            query_params = parse_qs(parsed_url.query)
            initial_site_id = query_params.get('id', [None])[0]
            initial_start_str = query_params.get('start_date', [None])[0]
            initial_end_str = query_params.get('end_date', [None])[0]
            logger.info(f"Parsed URL parameters: id='{initial_site_id}', start='{initial_start_str}', end='{initial_end_str}'")
            if initial_start_str:
                try:
                    parsed_start = datetime.strptime(initial_start_str, '%Y-%m-%d').date()
                    start_date = parsed_start
                except ValueError:
                    logger.warning(f"Invalid start_date format '{initial_start_str}' in URL, using default: {default_start_date.isoformat()}.")
            if initial_end_str:
                try:
                    parsed_end = datetime.strptime(initial_end_str, '%Y-%m-%d').date()
                    if parsed_end >= start_date:
                        end_date = parsed_end
                    else:
                        logger.warning(f"URL end_date '{initial_end_str}' is before start_date '{start_date.isoformat()}', using default end date: {default_end_date.isoformat()}.")
                        end_date = default_end_date
                except ValueError:
                    logger.warning(f"Invalid end_date format '{initial_end_str}' in URL, using default: {default_end_date.isoformat()}.")
            if start_date > end_date:
                logger.warning(f"Resulting start date {start_date.isoformat()} is after end date {end_date.isoformat()}. Resetting to defaults.")
                start_date = default_start_date
                end_date = default_end_date
        except Exception as e:
            logger.error(f"Error parsing URL '{href}': {e}", exc_info=True)
            initial_site_id = None
            start_date = default_start_date
            end_date = default_end_date
    else:
        logger.debug("Initial load: No URL href provided, using default values.")
    logger.info(f"Setting initial input values: Site='{initial_site_id}', Start='{start_date.isoformat()}', End='{end_date.isoformat()}'")
    return initial_site_id, start_date.isoformat(), end_date.isoformat()

@callback(
    [Output('main-plot', 'figure'),
     Output('table-container', 'children'),
     Output('data-store', 'data'),
     Output('site-info-store', 'data'),
     Output('statistics-display', 'children'),
     Output('threshold-form-content', 'children'),
     Output('main-title', 'children'),
     Output('notification-area', 'children', allow_duplicate=True),
     Output('save-button', 'disabled'),
     Output('table-edit-status', 'is_open', allow_duplicate=True),
     Output('start-date-picker', 'date', allow_duplicate=True),
     Output('end-date-picker', 'date', allow_duplicate=True)],
    [Input('update-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('quick-year-button', 'n_clicks'),
     Input('quick-month-button', 'n_clicks'),
     Input('thresholds-store', 'data')],
    [State('site-id-input', 'value'),
     State('start-date-picker', 'date'),
     State('end-date-picker', 'date'),
     State('thresholds-store', 'data')],
    prevent_initial_call=True
)
def update_data_and_plots(update_clicks, reset_clicks, year_clicks, month_clicks, threshold_data_input,
                          site_id, current_start_date_str, current_end_date_str, current_thresholds_state):
    # ... (implementation as before) ...
    triggered_input_id = ctx.triggered_id
    logger.debug(f"update_data_and_plots triggered by: {triggered_input_id}")
    threshold_update_trigger = triggered_input_id == 'thresholds-store'
    empty_fig = go.Figure(layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'annotations': [{'text': 'No Data Loaded', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 20}}]})
    initial_table_msg = dbc.Alert("Click 'Update Plot' or select a site.", color="info")
    error_alert = None
    site_info_output = None
    stats_display = html.P("Load data to view statistics.")
    threshold_form = html.P("Load site data to view/edit thresholds.")
    main_title = "Data Quality Analysis"
    save_disabled = True
    edit_status_open = False
    output_start_date = no_update
    output_end_date = no_update
    button_id = triggered_input_id if not threshold_update_trigger else None
    is_reset = button_id == 'reset-button'
    is_quick_year = button_id == 'quick-year-button'
    is_quick_month = button_id == 'quick-month-button'
    is_update = button_id == 'update-button'

    if not triggered_input_id:
        logger.debug("Callback triggered without specific input ID. No update.")
        return empty_fig, initial_table_msg, None, None, stats_display, threshold_form, main_title, None, save_disabled, edit_status_open, no_update, no_update
    if not site_id:
        if threshold_update_trigger:
            logger.warning("Threshold update trigger ignored: No Site ID entered.")
            return no_update
        error_alert = dbc.Alert("Please enter a Site ID.", color="danger", dismissable=True)
        return empty_fig, initial_table_msg, None, None, stats_display, build_threshold_form(site_id, current_thresholds_state, "?"), main_title, error_alert, save_disabled, edit_status_open, no_update, no_update

    start_proc, end_proc = None, None
    try:
        today = datetime.now().date()
        if is_reset:
            logger.info(f"Reset button triggered for site {site_id}.")
            start_proc, end_proc = None, None
        elif is_quick_year or is_quick_month:
            logger.info(f"Quick date '{button_id}' triggered for site {site_id}.")
            end_dt_obj = today
            start_dt_obj = today - timedelta(days=365 if is_quick_year else 30)
            start_proc = start_dt_obj.strftime('%Y-%m-%d')
            end_proc = end_dt_obj.strftime('%Y-%m-%d')
            output_start_date = start_dt_obj.isoformat()
            output_end_date = end_dt_obj.isoformat()
        elif is_update or threshold_update_trigger:
            if threshold_update_trigger: logger.info(f"Threshold update triggered plot refresh for site {site_id}.")
            else: logger.info(f"'Update Plot' button triggered for site {site_id}.")
            if not current_start_date_str or not current_end_date_str:
                error_alert = dbc.Alert("Please select both Start and End Dates.", color="danger", dismissable=True)
                return empty_fig, initial_table_msg, None, {'site_id': site_id}, stats_display, build_threshold_form(site_id, current_thresholds_state, "?"), f"Error - {site_id}", error_alert, save_disabled, edit_status_open, no_update, no_update
            try:
                start_dt_obj = datetime.strptime(current_start_date_str[:10], '%Y-%m-%d').date()
                end_dt_obj = datetime.strptime(current_end_date_str[:10], '%Y-%m-%d').date()
                if start_dt_obj > end_dt_obj:
                    error_alert = dbc.Alert("Start Date cannot be after End Date.", color="danger", dismissable=True)
                    return empty_fig, initial_table_msg, None, {'site_id': site_id}, stats_display, build_threshold_form(site_id, current_thresholds_state, "?"), f"Date Error - {site_id}", error_alert, save_disabled, edit_status_open, no_update, no_update
                start_proc = start_dt_obj.strftime('%Y-%m-%d')
                end_proc = end_dt_obj.strftime('%Y-%m-%d')
            except ValueError:
                error_alert = dbc.Alert("Invalid date format selected.", color="danger", dismissable=True)
                return empty_fig, initial_table_msg, None, {'site_id': site_id}, stats_display, build_threshold_form(site_id, current_thresholds_state, "?"), f"Date Error - {site_id}", error_alert, save_disabled, edit_status_open, no_update, no_update
        else:
            logger.warning(f"Unhandled trigger in update_data_and_plots: {triggered_input_id}")
            return no_update
    except Exception as date_e:
        logger.error(f"Error processing dates for site {site_id}: {date_e}", exc_info=True)
        error_alert = dbc.Alert(f"Error processing dates: {date_e}", color="danger", dismissable=True)
        return empty_fig, initial_table_msg, None, {'site_id': site_id}, stats_display, build_threshold_form(site_id, current_thresholds_state, "?"), f"Date Error - {site_id}", error_alert, save_disabled, edit_status_open, no_update, no_update

    logger.info(f"Processing request: Site='{site_id}', Start='{start_proc}', End='{end_proc}', Reset={is_reset}, ThresholdTrigger={threshold_update_trigger}")
    try:
        thresholds_to_use = current_thresholds_state if threshold_update_trigger and current_thresholds_state else None
        fig, df_processed, err_func, name_func, final_start, final_end, units_val, found_thresholds, stats_dict = generate_plot_for_site(site_id, start_proc, end_proc, is_reset, logger, thresholds_override=thresholds_to_use)
        site_info_output = {'site_id': site_id, 'name': name_func, 'units': units_val, 'start': final_start, 'end': final_end}
        thresholds_for_form = found_thresholds if found_thresholds else (current_thresholds_state if current_thresholds_state else {})
        if is_reset and final_start and final_end:
            output_start_date = final_start
            output_end_date = final_end
        if err_func:
            error_alert = dbc.Alert(f"Error loading data for {site_id}: {err_func}", color="danger", dismissable=True)
            main_title = f"{name_func or '?'} ({site_id}) - Data Error"
            threshold_form = build_threshold_form(site_id, thresholds_for_form, units_val or "?")
            return empty_fig, initial_table_msg, None, site_info_output, stats_display, threshold_form, main_title, error_alert, save_disabled, edit_status_open, output_start_date, output_end_date
        if fig is None or df_processed is None or df_processed.empty:
            error_alert = dbc.Alert(f"No data found for site {site_id} in the selected period ({final_start} to {final_end}).", color="warning", dismissable=True)
            main_title = f"{name_func or '?'} ({site_id})"
            threshold_form = build_threshold_form(site_id, thresholds_for_form, units_val or "?")
            return empty_fig, initial_table_msg, None, site_info_output, stats_display, threshold_form, main_title, error_alert, save_disabled, edit_status_open, output_start_date, output_end_date

        logger.info(f"Data loaded successfully for {site_id}. Range: {final_start} to {final_end}. Units: {units_val}")
        table_component = create_dash_data_table(df_processed.copy(), units_val, 'editable-data-table', logger)
        if not isinstance(table_component, dash_table.DataTable):
            error_alert = table_component
            table_component = html.Div()
        df_store = df_processed.copy()
        if 'Date' in df_store.columns:
            df_store['Date'] = df_store['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        stored_data = df_store.to_json(orient='split', date_format='iso')
        threshold_form = build_threshold_form(site_id, thresholds_for_form, units_val)
        if stats_dict:
            stats_display = html.Div([
                html.P([html.Strong("Count: "), html.Span(stats_dict.get('count'))]),
                html.P([html.Strong(f"Mean ({stats_dict.get('units', '?')}): "), html.Span(f"{stats_dict.get('mean'):.2f}" if isinstance(stats_dict.get('mean'), (int, float)) else stats_dict.get('mean'))]),
                html.P([html.Strong(f"Min ({stats_dict.get('units', '?')}): "), html.Span(f"{stats_dict.get('min'):.2f}" if isinstance(stats_dict.get('min'), (int, float)) else stats_dict.get('min'))]),
                html.P([html.Strong(f"Max ({stats_dict.get('units', '?')}): "), html.Span(f"{stats_dict.get('max'):.2f}" if isinstance(stats_dict.get('max'), (int, float)) else stats_dict.get('max'))])
            ], className="small")
        else: stats_display = html.P("Statistics not available.")
        main_title = f"{name_func or '?'} ({site_id}) | {final_start} to {final_end}"
        save_disabled = False
        edit_status_open = True
        return fig, table_component, stored_data, site_info_output, stats_display, threshold_form, main_title, error_alert, save_disabled, edit_status_open, output_start_date, output_end_date
    except Exception as e:
        logger.error(f"Unhandled exception during plot/table generation for site {site_id}: {e}", exc_info=True)
        error_alert = dbc.Alert(f"An unexpected server error occurred: {e}", color="danger", dismissable=True)
        main_title = f"Error Processing {site_id}"
        try:
            thresholds_state_to_use = get_site_thresholds(site_id, logger) if site_id else {}
        except Exception as te:
            logger.error(f"Failed to get thresholds after main error: {te}")
            thresholds_state_to_use = {}
        threshold_form = build_threshold_form(site_id, thresholds_state_to_use, "?")
        return empty_fig, initial_table_msg, None, {'site_id': site_id}, stats_display, threshold_form, main_title, error_alert, save_disabled, edit_status_open, no_update, no_update

@callback(
    Output("table-collapse", "is_open"),
    Input("toggle-table-button", "n_clicks"),
    State("table-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_table_collapse(n_clicks, is_open):
    # ... (implementation as before) ...
    if n_clicks:
        logger.debug(f"Toggling table collapse. Current state: {is_open}, New state: {not is_open}")
        return not is_open
    logger.debug("Toggle table callback triggered but n_clicks is 0 or None.")
    return is_open

def build_threshold_form(site_id, thresholds, units):
     # ... (implementation as before) ...
    if not site_id:
        return html.P("Enter a Site ID to load thresholds.")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    unit_str = f" ({units})" if units and units != '?' else ""
    roc_unit_str = f" ({units}/day)" if units and units != '?' else " (units/day)"
    max_val = thresholds.get('max_val')
    spike_unusual = thresholds.get('spike_unusual')
    repeated_days = thresholds.get('repeated_values_threshold', DEFAULT_REPEATED_THRESHOLD)
    form_content = dbc.Form([
        dcc.Input(id='threshold-site-id-hidden', type='hidden', value=site_id),
        dbc.Row([ dbc.Label(f"Max Capacity{unit_str}:", width=5, className="text-end"), dbc.Col(dbc.Input(id='threshold-max-val', type='number', value=max_val, required=True, step='any', placeholder="e.g., 10000"), width=5), dbc.Col(width=2)], className="mb-2 align-items-center"),
        dbc.Row([ dbc.Label(f"Unusual Spike RoC{roc_unit_str}:", width=5, className="text-end"), dbc.Col(dbc.Input(id='threshold-spike-unusual', type='number', value=spike_unusual, required=True, step='any', placeholder="e.g., 5000"), width=5), dbc.Col(width=2)], className="mb-2 align-items-center"),
        dbc.Row([ dbc.Label("Repeated Value Days:", width=5, className="text-end"), dbc.Col(dbc.Input(id='threshold-repeated-days', type='number', value=repeated_days, required=True, step=1, min=2, placeholder="e.g., 3"), width=5), dbc.Col("(min 2)", width=2, className="text-muted small")], className="mb-2 align-items-center"),
        dbc.Button("Update Thresholds", id="update-thresholds-button", color="warning", className="mt-3")
    ])
    if not thresholds:
        return html.Div([ html.P(f"Thresholds not currently loaded for site {site_id}.", className="text-danger"), form_content])
    else:
        return form_content

@callback(
    [Output('notification-area', 'children', allow_duplicate=True),
     Output('thresholds-store', 'data', allow_duplicate=True)],
    Input('update-thresholds-button', 'n_clicks'),
    [State('threshold-site-id-hidden', 'value'),
     State('threshold-max-val', 'value'),
     State('threshold-spike-unusual', 'value'),
     State('threshold-repeated-days', 'value')],
    prevent_initial_call=True
)
def update_site_thresholds(n_clicks, site_id, max_val_in, spike_roc_in, repeated_days_in):
    # ... (implementation as before) ...
    if not n_clicks or not site_id:
        logger.debug("Update thresholds callback triggered without click or site_id.")
        return no_update, no_update
    logger.info(f"Attempting threshold update for site {site_id} from form.")
    error_messages = []
    try:
        if max_val_in is None: error_messages.append("Max Capacity is required.")
        else: max_val = float(max_val_in)
        if spike_roc_in is None: error_messages.append("Unusual Spike RoC is required.")
        else: spike_roc = float(spike_roc_in)
        if repeated_days_in is None: error_messages.append("Repeated Value Days is required.")
        else:
            repeated_days = int(repeated_days_in)
            if repeated_days < 2: error_messages.append("'Repeated Days' must be 2 or greater.")
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid number format entered in threshold form for site {site_id}: {e}")
        error_messages.append(f"Invalid number format entered: {e}")
    if error_messages:
        alert_message = html.Div([html.P("Threshold update failed:")] + [html.Li(msg) for msg in error_messages])
        return dbc.Alert(alert_message, color="danger", dismissable=True), no_update
    new_threshold_values = {'max_val': max_val, 'spike_unusual': spike_roc, 'repeated_values_threshold': repeated_days}
    try:
        logger.info(f"Calling update_threshold_in_csv for {site_id} with values: {new_threshold_values}")
        success, message = update_threshold_in_csv(site_id, new_threshold_values, logger)
        if success:
            logger.info(f"Thresholds for site {site_id} updated successfully in CSV.")
            notification = dbc.Alert(message, color="success", dismissable=True, duration=6000)
            updated_thresholds_for_site = get_site_thresholds(site_id, logger)
            if updated_thresholds_for_site is None:
                logger.error("Failed to reload thresholds after successful update!")
                return dbc.Alert("Update successful, but failed to reload new thresholds.", color="warning"), no_update
            else:
                return notification, updated_thresholds_for_site
        else:
            logger.error(f"Failed to update thresholds for site {site_id}: {message}")
            notification = dbc.Alert(f"Threshold update failed: {message}", color="danger", dismissable=True)
            return notification, no_update
    except Exception as e:
        logger.error(f"Unexpected error during threshold update for site {site_id}: {e}", exc_info=True)
        notification = dbc.Alert(f"An unexpected server error occurred during threshold update: {e}", color="danger", dismissable=True)
        return notification, no_update

@callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('table-container', 'children', allow_duplicate=True),
     Output('notification-area', 'children', allow_duplicate=True),
     Output('table-edit-status', 'is_open', allow_duplicate=True)],
    Input('editable-data-table', 'data'),
    [State('data-store', 'data'),
     State('site-info-store', 'data'),
     State('thresholds-store', 'data'),
     State('editable-data-table', 'data_previous')],
    prevent_initial_call=True
)
def handle_table_edit(edited_table_data, stored_json, site_info, thresholds, previous_table_data):
    # ... (implementation as before) ...
    if stored_json is None or edited_table_data is None or site_info is None:
        logger.warning("Table edit callback skipped - missing essential state.")
        return no_update, no_update, no_update, True
    if thresholds is None:
        logger.warning("Thresholds are missing. Proceeding with edit but cannot re-flag.")
        thresholds = {}
    units = site_info.get('units', '?')
    site_id = site_info.get('site_id', 'N/A')
    notification = None
    edit_status_open = True
    if previous_table_data is None:
        logger.debug("No previous table data found.")
        return no_update, no_update, no_update, edit_status_open
    try:
        df_edited = pd.DataFrame(edited_table_data)
        df_prev = pd.DataFrame(previous_table_data)
        df_edited['Discharge'] = pd.to_numeric(df_edited['Discharge'], errors='coerce')
        df_prev['Discharge'] = pd.to_numeric(df_prev['Discharge'], errors='coerce')
        diff_mask = df_prev['Discharge'].ne(df_edited['Discharge']) & ~(df_prev['Discharge'].isna() & df_edited['Discharge'].isna())
        changed_indices = diff_mask[diff_mask].index
        if not changed_indices.any():
            logger.debug("No change detected in 'Discharge' column.")
            return no_update, no_update, no_update, edit_status_open
        if len(changed_indices) > 1:
            logger.warning(f"Multiple rows changed simultaneously in table edit. Processing first change at index {changed_indices[0]}.")
        changed_view_idx = changed_indices[0]
        df_store_orig = pd.read_json(stored_json, orient='split')
        df_store_orig['Date'] = pd.to_datetime(df_store_orig['Date'], errors='coerce')
        if changed_view_idx >= len(df_store_orig):
            logger.error(f"Edit index {changed_view_idx} is out of bounds for stored DataFrame.")
            raise IndexError("Edit index mismatch between table view and stored data.")
        original_df_index = df_store_orig.index[changed_view_idx]
        new_value_edited = df_edited.loc[changed_view_idx, 'Discharge']
        old_value_original = df_store_orig.loc[original_df_index, 'DISCHARGE']
        logger.info(f"Table edit detected: Original DF Index={original_df_index}, Col='Discharge', Old='{old_value_original}', New='{new_value_edited}'")
        new_numeric_value = np.nan
        if pd.isna(new_value_edited):
            new_numeric_value = np.nan
        else:
            try: new_numeric_value = float(new_value_edited)
            except (ValueError, TypeError):
                logger.error(f"Invalid numeric input '{new_value_edited}' at index {original_df_index}.")
                reverted_table = create_dash_data_table(df_store_orig.copy(), units, 'editable-data-table', logger)
                return no_update, reverted_table, dbc.Alert(f"Invalid input: '{new_value_edited}' is not a valid number.", color="danger"), edit_status_open
        df_updated = df_store_orig.copy()
        df_updated.loc[original_df_index, 'DISCHARGE'] = new_numeric_value
        df_updated.loc[original_df_index, 'ReviewStatus'] = 'Edited'
        if 'Qualifiers' in df_updated.columns:
            current_qual = df_updated.loc[original_df_index, 'Qualifiers']
            df_updated.loc[original_df_index, 'Qualifiers'] = f"{current_qual};Edited" if pd.notna(current_qual) and current_qual else 'Edited'
        else:
            df_updated['Qualifiers'] = pd.Series(dtype='object')
            df_updated.loc[original_df_index, 'Qualifiers'] = 'Edited'
        logger.info(f"Re-flagging data for site {site_id} after edit at index {original_df_index}...")
        if not thresholds:
            logger.warning("Cannot re-flag: Thresholds are missing.")
            notification = dbc.Alert("Change applied, but cannot re-flag (thresholds missing). Click 'Save Changes'.", color="warning", duration=5000, dismissable=True)
        else:
            try:
                df_updated = apply_flagging(df_updated, thresholds, logger)
                logger.info("Re-flagging complete.")
                notification = dbc.Alert("Change processed and data re-flagged. Click 'Save Changes' to export.", color="warning", duration=5000, dismissable=True)
            except Exception as flag_e:
                logger.error(f"Error during re-flagging after edit: {flag_e}", exc_info=True)
                notification = dbc.Alert(f"Change applied, but error during re-flagging: {flag_e}", color="danger", duration=5000, dismissable=True)
        df_updated_store = df_updated.copy()
        if 'Date' in df_updated_store.columns:
            df_updated_store['Date'] = df_updated_store['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        updated_json_output = df_updated_store.to_json(orient='split', date_format='iso')
        new_table_component = create_dash_data_table(df_updated.copy(), units, 'editable-data-table', logger)
        if not isinstance(new_table_component, dash_table.DataTable):
            logger.error("Failed to recreate DataTable component after edit.")
            notification = dbc.Alert("Edit applied, but failed to update table display.", color="danger")
            new_table_component = html.Div("Error updating table.")
            return no_update, new_table_component, notification, edit_status_open
        logger.info(f"Successfully processed table edit for index {original_df_index}.")
        return updated_json_output, new_table_component, notification, edit_status_open
    except Exception as e:
        logger.error(f"Error handling table edit for site {site_id}: {e}", exc_info=True)
        notification = dbc.Alert(f"Error processing table edit: {e}", color="danger", dismissable=True)
        try:
            df_original_state = pd.read_json(stored_json, orient='split')
            df_original_state['Date'] = pd.to_datetime(df_original_state['Date'], errors='coerce')
            original_table_component = create_dash_data_table(df_original_state.copy(), units, 'editable-data-table', logger)
            return no_update, original_table_component, notification, edit_status_open
        except Exception as revert_e:
            logger.error(f"Failed to revert table display after edit error: {revert_e}")
            return no_update, html.Div("Error displaying table after edit failure."), notification, edit_status_open

@callback(
    Output('save-status', 'children'),
    Input('save-button', 'n_clicks'),
    [State('data-store', 'data'),
     State('site-info-store', 'data')],
    prevent_initial_call=True
)
def save_data(n_clicks, stored_json, site_info):
     # ... (implementation as before) ...
    if not n_clicks: return ""
    if stored_json is None or site_info is None:
        logger.warning("Save button clicked, but no data or site info found.")
        return dbc.Alert("No data loaded to save.", color="warning", dismissable=True)
    site_id = site_info.get('site_id', 'unknown_site')
    logger.info(f"Save button clicked for site {site_id}.")
    try:
        df_save = pd.read_json(stored_json, orient='split')
        if 'Date' in df_save.columns:
            df_save['Date'] = pd.to_datetime(df_save['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
            if df_save['Date'].isnull().any(): logger.warning("Some dates were invalid for saving.")
        else:
            logger.error("Cannot save: 'Date' column missing.")
            return dbc.Alert("Save Error: 'Date' column missing.", color="danger", dismissable=True)
        if 'DISCHARGE' in df_save.columns: df_save['DISCHARGE'] = pd.to_numeric(df_save['DISCHARGE'], errors='coerce').round(2)
        if 'ReviewStatus' not in df_save.columns: df_save['ReviewStatus'] = 'Unknown'
        if 'Qualifiers' not in df_save.columns: df_save['Qualifiers'] = None
        if 'Qualified' not in df_save.columns: df_save['Qualified'] = df_save['Qualifiers'].notna().map({True: 'Yes', False: 'No'})
        if 'Active Flags' not in df_save.columns:
            logger.warning("Recreating 'Active Flags' column for saving.")
            active_flags_list = []; flag_map = {'FLAG_LESS_THAN_Min._Value': 'Below Min', 'FLAG_ZERO': 'Zero', 'FLAG_REPEATED': 'Repeated', 'FLAG_GREATER_THAN_MaxValue': 'Above Max', 'UNUSUAL_SPIKE': 'Spike', 'FLAG_BELOW_CAPACITY': 'Below Capacity'}
            flag_cols_to_check = [col for col in flag_map if col in df_save.columns]
            if flag_cols_to_check:
                for _, row in df_save.iterrows():
                    try: active = [flag_map[col] for col in flag_cols_to_check if row.get(col) == True]; active_flags_list.append(', '.join(active) if active else 'None')
                    except Exception as e: logger.error(f"Error processing flags for save file: {e}"); active_flags_list.append('Error')
            else: active_flags_list = ['N/A'] * len(df_save)
            df_save['Active Flags'] = active_flags_list
        base_columns = ['Date', 'DISCHARGE', 'ReviewStatus', 'Qualifiers', 'Qualified', 'Active Flags']
        flag_columns = ['FLAG_LESS_THAN_Min._Value','FLAG_ZERO','FLAG_REPEATED','FLAG_GREATER_THAN_MaxValue','UNUSUAL_SPIKE','FLAG_BELOW_CAPACITY', 'FLAGGED']
        existing_flag_columns = [col for col in flag_columns if col in df_save.columns]
        columns_to_save = base_columns + existing_flag_columns
        columns_that_exist = [col for col in columns_to_save if col in df_save.columns]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"edited_data_{site_id}_{timestamp}.csv"
        save_path = Path(".") / filename
        logger.info(f"Saving columns: {columns_that_exist} to file: {save_path}")
        df_save[columns_that_exist].to_csv(save_path, index=False, encoding='utf-8')
        logger.info(f"Data for site {site_id} saved successfully to {filename}.")
        return dbc.Alert(f"Data saved successfully as: {filename}", color="success", duration=5000, dismissable=True)
    except Exception as e:
        logger.error(f"Error saving data for site {site_id}: {e}", exc_info=True)
        return dbc.Alert(f"Save Error: {e}", color="danger", dismissable=True)

@callback(
    [Output('qc-action-modal', 'is_open'),
     Output('qc-modal-body', 'children'),
     Output('clicked-point-store', 'data')],
    Input('main-plot', 'clickData'),
    State('main-plot', 'figure'),
    prevent_initial_call=True
)
def display_click_data(clickData, figure):
     # ... (implementation as before) ...
    if not clickData or not figure or 'points' not in clickData or not clickData['points']:
        logger.debug("Plot click detected, but no valid point data found.")
        return False, no_update, no_update
    point_data = clickData['points'][0]; curve_index = point_data.get('curveNumber', -1)
    logger.debug(f"Plot click details: {point_data}")
    open_modal = False; modal_body_content = "No details available."; clicked_point_info = None
    try:
        if 0 <= curve_index < len(figure.get('data', [])):
            trace = figure['data'][curve_index]
            original_index = None
            if 'markers' in trace.get('mode', '') and 'customdata' in point_data and point_data['customdata'] is not None:
                custom_data = point_data['customdata']
                if isinstance(custom_data, (list, tuple)) and len(custom_data) > 0: original_index = custom_data[0]
                elif isinstance(custom_data, (int, str, float)): original_index = custom_data
                else: logger.warning(f"Unexpected customdata format: {type(custom_data)}")
            if original_index is not None:
                flag_type = "Flagged Point"; point_number = point_data.get('pointNumber', -1)
                if 'meta' in trace and isinstance(trace.get('meta'), list) and 0 <= point_number < len(trace['meta']): flag_type = trace['meta'][point_number]
                elif 'name' in trace: flag_type = trace.get('name', flag_type)
                date_str = point_data.get('x'); value = point_data.get('y')
                value_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                modal_body_content = html.Div([ html.P([html.Strong("Date: "), html.Span(date_str)]), html.P([html.Strong("Value: "), html.Span(value_str)]), html.P([html.Strong("Flag Type: "), html.Span(flag_type)]), html.P(f"(Original Index: {original_index})", className="small text-muted")])
                clicked_point_info = {'original_index': original_index, 'date': date_str, 'value': value, 'flag_type': flag_type}
                open_modal = True
                logger.info(f"Flagged point clicked: Index={original_index}, Date={date_str}, Value={value_str}, Flag={flag_type}")
            else: logger.debug(f"Click not on flagged marker or customdata missing/invalid.")
        else: logger.warning(f"Invalid curve_index {curve_index} from click data.")
    except Exception as e:
        logger.error(f"Error processing plot click data: {e}", exc_info=True)
        modal_body_content = f"Error processing click: {e}"; open_modal = False; clicked_point_info = None
    if open_modal: return True, modal_body_content, clicked_point_info
    else: return False, no_update, None

@callback(
    [Output('qc-action-modal', 'is_open', allow_duplicate=True),
     Output('notification-area', 'children', allow_duplicate=True),
     Output('data-store', 'data', allow_duplicate=True),
     Output('table-container', 'children', allow_duplicate=True),
     Output('main-plot', 'figure', allow_duplicate=True)],
    [Input('qc-approve-button', 'n_clicks'),
     Input('qc-interpolate-button', 'n_clicks'),
     Input('qc-delete-button', 'n_clicks'),
     Input('qc-close-button', 'n_clicks')],
    [State('clicked-point-store', 'data'),
     State('data-store', 'data'),
     State('site-info-store', 'data'),
     State('thresholds-store', 'data')],
    prevent_initial_call=True
)
def handle_qc_action(approve_clicks, interpolate_clicks, delete_clicks, close_clicks,
                     clicked_point_data, stored_json, site_info, thresholds):
    # ... (implementation as before) ...
    triggered_button_id = ctx.triggered_id
    modal_is_open = False; notification = no_update; updated_data_store = no_update; updated_table = no_update; updated_plot = no_update
    if triggered_button_id == 'qc-close-button':
        logger.debug("QC modal closed via Close button.")
        return modal_is_open, notification, updated_data_store, updated_table, updated_plot
    if not triggered_button_id:
        logger.warning("QC action callback triggered without a button ID.")
        return modal_is_open, notification, updated_data_store, updated_table, updated_plot
    if not clicked_point_data or 'original_index' not in clicked_point_data:
        logger.warning(f"QC action '{triggered_button_id}' triggered, but no valid point selected.")
        notification = dbc.Alert("No point selected or index missing.", color="warning", dismissable=True)
        return modal_is_open, notification, updated_data_store, updated_table, updated_plot
    if not stored_json:
        logger.error(f"QC action '{triggered_button_id}' triggered, but data-store is empty.")
        notification = dbc.Alert("Cannot perform action: Data not found.", color="danger", dismissable=True)
        return modal_is_open, notification, updated_data_store, updated_table, updated_plot
    if not site_info:
        logger.error(f"QC action '{triggered_button_id}' triggered, but site-info-store is empty.")
        notification = dbc.Alert("Cannot perform action: Site info not found.", color="danger", dismissable=True)
        return modal_is_open, notification, updated_data_store, updated_table, updated_plot
    original_index = clicked_point_data.get('original_index')
    date_clicked = clicked_point_data.get('date')
    value_clicked = clicked_point_data.get('value')
    logger.info(f"Processing QC Action '{triggered_button_id}' for index: {original_index}, Date: {date_clicked}, Value: {value_clicked}")
    action_performed = None; df = None
    try:
        df = pd.read_json(stored_json, orient='split')
        try:
            if df.index.dtype != type(original_index):
                if isinstance(original_index, str) and original_index.isdigit(): original_index = int(original_index)
                elif isinstance(original_index, (int,float)) and not isinstance(df.index, pd.RangeIndex) : original_index = str(original_index)
        except Exception as e: logger.warning(f"Could not convert original_index type {type(original_index)} to match index type {df.index.dtype}: {e}")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if original_index not in df.index:
            logger.error(f"Index {original_index} (type {type(original_index)}) not found in DataFrame index (type {df.index.dtype})!")
            raise IndexError(f"Data mismatch: Point index {original_index} not found.")
        if 'Qualifiers' not in df.columns: df['Qualifiers'] = pd.Series(dtype='object')
        if 'ReviewStatus' not in df.columns: df['ReviewStatus'] = 'Unknown'
        if triggered_button_id == 'qc-approve-button':
            flags_to_clear = [col for col in ['FLAG_LESS_THAN_Min._Value','FLAG_ZERO','FLAG_REPEATED','FLAG_GREATER_THAN_MaxValue','UNUSUAL_SPIKE','FLAG_BELOW_CAPACITY', 'FLAGGED'] if col in df.columns]
            if flags_to_clear: df.loc[original_index, flags_to_clear] = False
            df.loc[original_index, 'ReviewStatus'] = 'Approved'
            current_qual = df.loc[original_index, 'Qualifiers'] if pd.notna(df.loc[original_index, 'Qualifiers']) else ''
            df.loc[original_index, 'Qualifiers'] = f"{current_qual};Approved".strip(';')
            action_performed = "Approved"
        elif triggered_button_id == 'qc-interpolate-button':
            df = df.sort_index()
            prev_idx = df.loc[:original_index-1, 'DISCHARGE'].last_valid_index()
            next_idx = df.loc[original_index+1:, 'DISCHARGE'].first_valid_index()
            if pd.notna(prev_idx) and pd.notna(next_idx):
                val_prev = df.loc[prev_idx, 'DISCHARGE']
                val_next = df.loc[next_idx, 'DISCHARGE']
                interpolated_val = np.interp(original_index, [prev_idx, next_idx], [val_prev, val_next])
                df.loc[original_index, 'DISCHARGE'] = interpolated_val
                df.loc[original_index, 'ReviewStatus'] = 'Interpolated'
                current_qual = df.loc[original_index, 'Qualifiers'] if pd.notna(df.loc[original_index, 'Qualifiers']) else ''
                df.loc[original_index, 'Qualifiers'] = f"{current_qual};Interpolated".strip(';')
                logger.info(f"Interpolated value at index {original_index} to {interpolated_val:.2f}")
                action_performed = "Interpolated"
            else:
                logger.warning(f"Cannot interpolate index {original_index}: Missing valid neighbors (prev={prev_idx}, next={next_idx}).")
                notification = dbc.Alert("Cannot interpolate: Missing valid neighbors.", color="warning", duration=4000, dismissable=True)
                action_performed = None
        elif triggered_button_id == 'qc-delete-button':
            df.loc[original_index, 'DISCHARGE'] = np.nan
            df.loc[original_index, 'ReviewStatus'] = 'Deleted'
            current_qual = df.loc[original_index, 'Qualifiers'] if pd.notna(df.loc[original_index, 'Qualifiers']) else ''
            df.loc[original_index, 'Qualifiers'] = f"{current_qual};Deleted".strip(';')
            action_performed = "Deleted (set to NaN)"
        if action_performed and df is not None:
            logger.info(f"Action '{action_performed}' applied for index {original_index}. Re-flagging...")
            units = site_info.get('units', '?')
            if thresholds:
                df = apply_flagging(df, thresholds, logger)
                logger.info("Re-flagging complete after QC action.")
            else:
                logger.warning("No thresholds available for re-flagging after QC action!")
                notification = notification or dbc.Alert("Action applied, but cannot re-flag (thresholds missing).", color="warning", dismissable=True)
            df_store_update = df.copy()
            if 'Date' in df_store_update.columns: df_store_update['Date'] = df_store_update['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            updated_data_store = df_store_update.to_json(orient='split', date_format='iso')
            updated_table = create_dash_data_table(df.copy(), units, 'editable-data-table', logger)
            if not isinstance(updated_table, dash_table.DataTable):
                logger.error("Failed to recreate table after QC action.")
                updated_table = html.Div("Error updating table.")
                notification = notification or dbc.Alert("Action applied, but failed to update table.", color="danger")
            updated_plot = no_update
            notification = notification or dbc.Alert(f"Action '{action_performed}' applied successfully. Save changes to export.", color="success", duration=4000, dismissable=True)
            logger.info(f"Store and table updated after '{action_performed}'.")
        elif not action_performed and not notification:
             notification = dbc.Alert(f"Action '{triggered_button_id}' did not result in changes.", color="secondary", dismissable=True)
        return modal_is_open, notification, updated_data_store, updated_table, updated_plot
    except Exception as e:
        logger.error(f"Error during QC action '{triggered_button_id}' for index {original_index}: {e}", exc_info=True)
        notification = dbc.Alert(f"Error processing action '{triggered_button_id}': {e}", color="danger", dismissable=True)
        return False, notification, no_update, no_update, no_update

@callback(
    Output('url', 'search'),
    Input('site-info-store', 'data'),
    prevent_initial_call=True
)
def update_url_on_data_change(site_info):
    # ... (implementation as before) ...
    if not site_info or not isinstance(site_info, dict):
        logger.debug("URL update skipped: site-info-store empty/invalid.")
        return no_update
    site_id = site_info.get('site_id'); start_date_str = site_info.get('start'); end_date_str = site_info.get('end')
    if not site_id or not start_date_str or not end_date_str:
        logger.warning(f"URL update skipped: Missing info in site-info: {site_info}")
        return no_update
    try:
        datetime.strptime(start_date_str, '%Y-%m-%d'); datetime.strptime(end_date_str, '%Y-%m-%d')
    except (ValueError, TypeError):
        logger.warning(f"URL update skipped: Invalid date format in site-info: start='{start_date_str}', end='{end_date_str}'")
        return no_update
    query_string = f"?id={site_id}&start_date={start_date_str}&end_date={end_date_str}"
    logger.info(f"Updating URL search string to: {query_string}")
    return query_string

@callback(
    Output('enter-data-modal', 'is_open', allow_duplicate=True),
    Output('notification-area', 'children', allow_duplicate=True),
    Input('open-enter-data-modal-button', 'n_clicks'),
    State('site-info-store', 'data'),
    prevent_initial_call=True
)
def open_enter_data_modal(n_clicks, site_info):
    # ... (implementation as before) ...
    if not n_clicks: return no_update, no_update
    if site_info and site_info.get('site_id'):
        logger.info("Opening enter data modal.")
        return True, no_update
    else:
        logger.warning("Cannot open Enter Data modal: No site data loaded.")
        return False, dbc.Alert("Load site data first before entering new measurements.", color="warning", duration=4000, dismissable=True)

@callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('table-container', 'children', allow_duplicate=True),
     Output('enter-data-modal', 'is_open', allow_duplicate=True),
     Output('enter-data-modal-alert', 'children'),
     Output('enter-data-modal-alert', 'is_open'),
     Output('notification-area', 'children', allow_duplicate=True)],
    Input('submit-enter-data-button', 'n_clicks'),
    [State('enter-date-picker', 'date'),
     State('enter-discharge-input', 'value'),
     State('enter-qualifier-input', 'value'),
     State('data-store', 'data'),
     State('site-info-store', 'data'),
     State('thresholds-store', 'data')],
    prevent_initial_call=True
)
def handle_submit_new_data(n_clicks, new_date_str, new_discharge, new_qualifier,
                           stored_json, site_info, thresholds):
    # ... (implementation as before) ...
    if not n_clicks: return no_update, no_update, no_update, no_update, False, no_update
    modal_errors = []
    if not new_date_str: modal_errors.append("Please select a date.")
    if new_discharge is None or new_discharge == '': modal_errors.append("Please enter a discharge value.")
    if stored_json is None or site_info is None:
        logger.error("Cannot add data: Missing session context.")
        main_notification = dbc.Alert("Error: Cannot add data, session context missing.", color="danger")
        error_msg = "Session data missing." + (" " + " ".join(modal_errors) if modal_errors else "")
        return no_update, no_update, True, error_msg, True, main_notification
    site_id = site_info.get('site_id'); units = site_info.get('units', '?')
    new_discharge_float = None
    if not modal_errors and new_discharge is not None:
        try: new_discharge_float = float(new_discharge)
        except (ValueError, TypeError):
            logger.error(f"Invalid discharge value: {new_discharge}")
            modal_errors.append(f"Invalid discharge value: '{new_discharge}'.")
    if modal_errors:
        error_message = html.Div([html.P("Please correct the following:")] + [html.Li(msg) for msg in modal_errors])
        return no_update, no_update, True, error_message, True, no_update
    logger.info(f"Attempting to add new measurement for site {site_id}: Date={new_date_str}, Discharge={new_discharge_float}, Qualifier={new_qualifier}")
    try:
        new_date = pd.to_datetime(new_date_str).normalize()
        df = pd.read_json(stored_json, orient='split')
        df['Date'] = pd.to_datetime(df['Date']).normalize()
        if new_date in df['Date'].values:
            logger.warning(f"Date {new_date_str} already exists. Appending.")
        base_cols = ['Date', 'DISCHARGE', 'ReviewStatus', 'Qualifiers', 'FLAGGED']
        flag_cols = ['FLAG_LESS_THAN_Min._Value','FLAG_ZERO','FLAG_REPEATED','FLAG_GREATER_THAN_MaxValue','UNUSUAL_SPIKE','FLAG_BELOW_CAPACITY']
        all_expected_cols = base_cols + flag_cols
        for col in all_expected_cols:
            if col not in df.columns:
                logger.debug(f"Adding missing column '{col}'.")
                if col in ['Date', 'DISCHARGE']: continue
                if col == 'FLAGGED' or col.startswith('FLAG_'): df[col] = False
                elif col == 'ReviewStatus': df[col] = 'Unknown'
                elif col == 'Qualifiers': df[col] = pd.Series(dtype='object')
                else: df[col] = np.nan
        new_row = { 'Date': new_date, 'DISCHARGE': new_discharge_float, 'ReviewStatus': 'Entered', 'Qualifiers': new_qualifier if new_qualifier else 'Manual Entry', 'FLAGGED': False, **{flag_col: False for flag_col in flag_cols if flag_col in df.columns}}
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df = df.sort_values(by='Date').reset_index(drop=True)
        logger.info(f"Appended new row for {new_date_str}. DataFrame size: {len(df)}")
        main_feedback = None
        if thresholds:
            logger.info("Re-flagging dataset after adding new row...")
            df = apply_flagging(df, thresholds, logger)
            logger.info("Re-flagging complete.")
            main_feedback = dbc.Alert(f"Measurement for {new_date_str} added and re-flagged.", color="success", duration=4000, dismissable=True)
        else:
            logger.warning("Thresholds missing, skipping re-flagging.")
            main_feedback = dbc.Alert(f"Measurement for {new_date_str} added, thresholds missing for re-flagging.", color="warning", duration=5000, dismissable=True)
        df_store_updated = df.copy()
        df_store_updated['Date'] = df_store_updated['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        updated_json_store = df_store_updated.to_json(orient='split', date_format='iso')
        new_table_component = create_dash_data_table(df.copy(), units, 'editable-data-table', logger)
        if not isinstance(new_table_component, dash_table.DataTable):
            logger.error("Failed to create table after adding data.")
            new_table_component = html.Div("Error updating table.")
            main_feedback = dbc.Alert("Measurement added, failed to update table.", color="danger")
            return updated_json_store, new_table_component, False, "Error updating table.", True, main_feedback
        logger.info(f"Successfully processed data for {new_date_str}.")
        return updated_json_store, new_table_component, False, no_update, False, main_feedback
    except Exception as e:
        logger.error(f"Error adding new data for site {site_id}: {e}", exc_info=True)
        return no_update, no_update, True, f"An unexpected error occurred: {e}", True, no_update

@callback(
    [Output('enter-data-modal', 'is_open', allow_duplicate=True),
     Output('enter-data-modal-alert', 'children', allow_duplicate=True),
     Output('enter-data-modal-alert', 'is_open', allow_duplicate=True)],
    [Input('cancel-enter-data-button', 'n_clicks')],
    prevent_initial_call=True
)
def cancel_enter_data_modal(cancel_clicks):
    # ... (implementation as before) ...
    if cancel_clicks:
        logger.debug("Closing enter data modal via Cancel.")
        return False, None, False
    return no_update, no_update, no_update

# --- REVISED & NEW CALLBACKS for Add Multiple Data Modal ---

# Callback to Open the Add Multiple Data modal & RESET Table
@callback(
    [Output('add-multiple-data-modal', 'is_open'),
     Output('notification-area', 'children', allow_duplicate=True),
     Output('add-data-input-table', 'data')], # ADDED Output to reset table
    Input('open-add-multiple-modal-button', 'n_clicks'),
    [State('site-info-store', 'data')], # Check if site data is loaded
    prevent_initial_call=True
)
def open_add_multiple_data_modal(n_clicks, site_info):
    """Opens the 'Add Multiple Data' modal if site data is loaded and resets the input table."""
    if not n_clicks:
        return no_update, no_update, no_update

    # Reset the table data whenever the modal is triggered to open
    initial_data = get_initial_input_table_data()

    if site_info and site_info.get('site_id'):
        logger.info("Opening add multiple data modal and resetting input table.")
        return True, no_update, initial_data # Open modal, clear notification, reset table
    else:
        logger.warning("Cannot open Add Multiple Data modal: No site data loaded.")
        # Don't open modal, show warning, return initial data (won't be visible)
        return False, dbc.Alert("Load site data first before adding multiple measurements.", color="warning", duration=4000, dismissable=True), initial_data

# REVISED Callback to Handle Submission of Multiple New Data Points from Input Table
@callback(
    [Output('data-store', 'data', allow_duplicate=True),             # Update main data
     Output('table-container', 'children', allow_duplicate=True),     # Update table display
     Output('add-multiple-data-modal', 'is_open', allow_duplicate=True),# Control modal visibility
     Output('add-multiple-data-modal-alert', 'children'),             # Feedback inside modal
     Output('add-multiple-data-modal-alert', 'is_open'),              # Show/hide modal alert
     Output('notification-area', 'children', allow_duplicate=True)],   # Feedback in main area
    Input('submit-multiple-data-button', 'n_clicks'),
    [State('add-data-input-table', 'data'), # *** CHANGED: Get data from the input table ***
     State('data-store', 'data'),
     State('site-info-store', 'data'),
     State('thresholds-store', 'data')],
    prevent_initial_call=True
)
def handle_submit_multiple_data(n_clicks, input_table_data, stored_json, site_info, thresholds):
    """Handles submission from the 'Add Multiple Data' modal using the DataTable."""
    if not n_clicks:
        return no_update, no_update, no_update, no_update, False, no_update

    # --- Basic Context Validation ---
    if stored_json is None or site_info is None:
        logger.error("Cannot add multiple data points: Missing current data-store or site-info-store.")
        main_notification = dbc.Alert("Error: Cannot add data, session context is missing. Please reload.", color="danger")
        return no_update, no_update, True, "Session data missing. Cannot proceed.", True, main_notification

    site_id = site_info.get('site_id'); units = site_info.get('units', '?')
    logger.info(f"Attempting to add measurements for site {site_id} from input table.")

    # --- Input Validation ---
    modal_errors = []
    valid_new_data_rows = [] # List to hold validated row data dictionaries

    if not input_table_data: # Check if the table data is empty
         modal_errors.append("No data entered or pasted into the table.")

    for i, row in enumerate(input_table_data):
        row_num = i + 1
        date_str = row.get('Date')
        discharge_val = row.get('Discharge')
        qualifier_val = row.get('Qualifier')

        # --- Filter out empty rows ---
        # Consider a row empty if both Date and Discharge are missing/None/empty strings
        if (date_str is None or str(date_str).strip() == "") and \
           (discharge_val is None or str(discharge_val).strip() == ""):
            continue # Skip this empty row entirely

        row_has_error = False
        validated_date = None
        discharge_float = None

        # Validate Date
        if date_str is None or str(date_str).strip() == "":
            modal_errors.append(f"Row {row_num}: Date is missing.")
            row_has_error = True
        else:
            try:
                # Attempt to parse strict YYYY-MM-DD format
                validated_date = datetime.strptime(str(date_str).strip(), '%Y-%m-%d').date()
            except (ValueError, TypeError):
                 try:
                     # Fallback: try pandas general parser (might be too lenient)
                     parsed_dt = pd.to_datetime(str(date_str).strip(), errors='coerce')
                     if pd.isna(parsed_dt):
                          modal_errors.append(f"Row {row_num}: Invalid date format ('{date_str}'). Use YYYY-MM-DD.")
                          row_has_error = True
                     else:
                          validated_date = parsed_dt.date()
                          logger.warning(f"Parsed potentially ambiguous date format '{date_str}' for row {row_num} as {validated_date}.")
                          # Optionally add a warning to modal_errors here
                 except Exception as e:
                    modal_errors.append(f"Row {row_num}: Invalid date ('{date_str}'). Use YYYY-MM-DD. Error: {e}")
                    row_has_error = True


        # Validate Discharge
        if discharge_val is None or str(discharge_val).strip() == "":
            modal_errors.append(f"Row {row_num}: Discharge value is missing.")
            row_has_error = True
        else:
            try:
                discharge_float = float(discharge_val)
            except (ValueError, TypeError):
                modal_errors.append(f"Row {row_num}: Invalid discharge value '{discharge_val}'. Must be a number.")
                row_has_error = True

        # If no errors for this row, add to our list of valid rows
        if not row_has_error and validated_date is not None and discharge_float is not None:
            valid_new_data_rows.append({
                'date': validated_date,
                'discharge': discharge_float,
                'qualifier': qualifier_val if qualifier_val else 'Manual Paste Entry' # Use input or default
            })
        elif not row_has_error:
             # This case should ideally not happen if validation is correct
             logger.error(f"Row {row_num} had no errors but data was not captured correctly. Date: {validated_date}, Discharge: {discharge_float}")
             modal_errors.append(f"Row {row_num}: Internal error processing valid data.")


    # Check for duplicate dates within the *valid* new entries
    submitted_dates = [row['date'] for row in valid_new_data_rows]
    if len(submitted_dates) != len(set(submitted_dates)):
        modal_errors.append("Duplicate dates found within the valid submitted entries. Please ensure all dates are unique.")

    # If no valid rows were found after filtering
    if not modal_errors and not valid_new_data_rows:
         modal_errors.append("No valid data rows found in the table. Please provide Date and Discharge.")


    # If any validation errors occurred, show them in the modal and stop
    if modal_errors:
        error_message = html.Div([html.P("Please correct the following errors:")] + [html.Li(msg) for msg in modal_errors], style={'maxHeight': '40vh', 'overflowY': 'auto'})
        return no_update, no_update, True, error_message, True, no_update

    # --- Process Valid Data ---
    try:
        # Load current data
        df = pd.read_json(stored_json, orient='split')
        df['Date'] = pd.to_datetime(df['Date']).normalize() # Ensure existing dates are comparable

        # Check for duplicate dates against existing data
        existing_dates = set(df['Date'].dt.date) # Compare date parts only
        duplicates_with_existing = [d.strftime('%Y-%m-%d') for d in submitted_dates if d in existing_dates]
        if duplicates_with_existing:
            logger.warning(f"Dates {duplicates_with_existing} already exist in the dataset. Appending new entries.")
            # Optionally add to modal_errors and return if duplicates should be prevented:
            # modal_errors.append(f"Dates already exist: {', '.join(duplicates_with_existing)}. Edit existing data or remove duplicates.")
            # error_message = ...
            # return no_update, no_update, True, error_message, True, no_update


        # --- Prepare and Add New Rows ---
        base_cols = ['Date', 'DISCHARGE', 'ReviewStatus', 'Qualifiers', 'FLAGGED']
        flag_cols = ['FLAG_LESS_THAN_Min._Value','FLAG_ZERO','FLAG_REPEATED','FLAG_GREATER_THAN_MaxValue','UNUSUAL_SPIKE','FLAG_BELOW_CAPACITY']
        all_expected_cols = base_cols + flag_cols
        for col in all_expected_cols:
            if col not in df.columns:
                logger.debug(f"Adding missing column '{col}' to DataFrame.")
                if col in ['Date', 'DISCHARGE']: continue
                if col == 'FLAGGED' or col.startswith('FLAG_'): df[col] = False
                elif col == 'ReviewStatus': df[col] = 'Unknown'
                elif col == 'Qualifiers': df[col] = pd.Series(dtype='object')
                else: df[col] = np.nan

        new_rows_prepared = []
        for row_data in valid_new_data_rows:
            new_rows_prepared.append({
                'Date': pd.Timestamp(row_data['date']), # Convert date back to Timestamp for DataFrame
                'DISCHARGE': row_data['discharge'],
                'ReviewStatus': 'Entered',
                'Qualifiers': row_data['qualifier'],
                'FLAGGED': False,
                **{flag_col: False for flag_col in flag_cols if flag_col in df.columns}
            })

        new_rows_df = pd.DataFrame(new_rows_prepared)

        # Append new rows
        df = pd.concat([df, new_rows_df], ignore_index=True)

        # Sort by date and reset index
        df = df.sort_values(by='Date').reset_index(drop=True)
        logger.info(f"Appended {len(new_rows_df)} new rows from input table. DataFrame size now: {len(df)}")

        # --- Re-flag Data ---
        main_feedback = None
        if thresholds:
            logger.info("Re-flagging entire dataset after adding multiple rows...")
            df = apply_flagging(df, thresholds, logger)
            logger.info("Re-flagging complete.")
            main_feedback = dbc.Alert(f"{len(new_rows_df)} measurements added and data re-flagged. Save changes to export.", color="success", duration=4000, dismissable=True)
        else:
            logger.warning("Thresholds not available, skipping re-flagging after adding data.")
            main_feedback = dbc.Alert(f"{len(new_rows_df)} measurements added, but thresholds missing for re-flagging. Save changes to export.", color="warning", duration=5000, dismissable=True)

        # --- Update Store and Table ---
        df_store_updated = df.copy()
        df_store_updated['Date'] = df_store_updated['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        updated_json_store = df_store_updated.to_json(orient='split', date_format='iso')

        new_table_component = create_dash_data_table(df.copy(), units, 'editable-data-table', logger)
        if not isinstance(new_table_component, dash_table.DataTable):
            logger.error("Failed to create table component after adding multiple data points.")
            new_table_component = html.Div("Error updating table display.")
            main_feedback = dbc.Alert("Measurements added, but failed to update table display.", color="danger")
            return updated_json_store, new_table_component, False, "Error updating table.", True, main_feedback

        logger.info(f"Successfully added and processed {len(new_rows_df)} data points from table.")
        # Close modal, clear modal alert, show success in main area
        return updated_json_store, new_table_component, False, None, False, main_feedback

    except Exception as e:
        logger.error(f"Error adding multiple new data points for site {site_id}: {e}", exc_info=True)
        # Keep modal open, show error inside modal
        return no_update, no_update, True, f"An unexpected error occurred: {e}", True, no_update


# Callback to Close the Add Multiple Data modal via Cancel button
@callback(
    [Output('add-multiple-data-modal', 'is_open', allow_duplicate=True),
     Output('add-multiple-data-modal-alert', 'children', allow_duplicate=True), # Clear alert
     Output('add-multiple-data-modal-alert', 'is_open', allow_duplicate=True)], # Hide alert
    Input('cancel-multiple-data-button', 'n_clicks'),
    prevent_initial_call=True
)
def cancel_add_multiple_data_modal(cancel_clicks):
    """Closes the 'Add Multiple Data' modal and clears its alert."""
    if cancel_clicks:
        logger.debug("Closing add multiple data modal via Cancel button.")
        # Close modal, clear and hide alert
        return False, None, False
    return no_update, no_update, no_update # No change if not triggered by cancel

# --- END REVISED/NEW CALLBACKS ---


# --- Main Execution Block ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    debug_env = os.environ.get("DASH_DEBUG", "True").lower()
    debug_mode = debug_env == "true"
    logger.info(f"Starting Dash server on http://127.0.0.1:{port}")
    logger.info(f" -> Debug mode: {'ON' if debug_mode else 'OFF'}")
    app.run(host='127.0.0.1', port=port, debug=debug_mode)

# --- END main_dash_app.py ---