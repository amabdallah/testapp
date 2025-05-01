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
from collections import Counter # Added for duplicate date checking

# --- Import functions ---
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
    # Fallback code
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
    # Fallback code
    def generate_plot_for_site(site_id, start, end, reset, logger, thresholds_override=None):
        logger.error("Plot generation unavailable.")
        fig = go.Figure().update_layout(title=f"Error: Plot generator unavailable for {site_id}")
        # Ensure fallback returns all expected values
        today_fb = date.today() # Use date directly
        start_fb = (today_fb - timedelta(days=30)).strftime('%Y-%m-%d')
        end_fb = today_fb.strftime('%Y-%m-%d')
        return fig, pd.DataFrame(), "Plot generator unavailable", site_id, start_fb, end_fb, "?", thresholds_override or {}, None
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
logger.info(f"Attempting initial threshold load from: {THRESHOLDS_CSV_PATH}")
if 'threshold_manager_dash' not in sys.modules:
    logger.warning(f"Skipping initial threshold load as threshold_manager_dash module was not found.")
elif load_thresholds(THRESHOLDS_CSV_PATH, logger) is None:
    logger.critical(f"CRITICAL STARTUP WARNING: Initial threshold load failed from '{THRESHOLDS_CSV_PATH}'.")
else:
    logger.info("Initial threshold load attempt complete.")


# --- Helper Function: create_dash_data_table ---
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
    # Note: If Date column contains time, this will truncate it for display
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


# --- Helper function for building threshold form ---
# <<< MODIFIED: Includes Subscribe Buttons >>>
def build_threshold_form(site_id, thresholds, units):
    if not site_id:
        return html.P("Enter a Site ID and load data to view/edit thresholds.")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    unit_str = f" ({units})" if units and units != '?' else ""
    roc_unit_str = f" ({units}/day)" if units and units != '?' else " (units/day)"
    # Use .get() with default None for safety
    max_val = thresholds.get('max_val')
    spike_unusual = thresholds.get('spike_unusual')
    repeated_days = thresholds.get('repeated_values_threshold', DEFAULT_REPEATED_THRESHOLD)

    # Adjusted column widths and added subscribe buttons
    form_content = dbc.Form([
        dcc.Input(id='threshold-site-id-hidden', type='hidden', value=site_id),
        dbc.Row([
            dbc.Label(f"Max Capacity{unit_str}:", width=4, className="text-end"),
            dbc.Col(dbc.Input(id='threshold-max-val', type='number', value=max_val, required=True, step='any', placeholder="e.g., 10000"), width=4),
            dbc.Col(dbc.Button("Subscribe", id='subscribe-max-val-button', color="success", size="sm"), width=3) # Added Button
        ], className="mb-2 align-items-center"),
        dbc.Row([
            dbc.Label(f"Unusual Spike RoC{roc_unit_str}:", width=4, className="text-end"),
            dbc.Col(dbc.Input(id='threshold-spike-unusual', type='number', value=spike_unusual, required=True, step='any', placeholder="e.g., 5000"), width=4),
            dbc.Col(dbc.Button("Subscribe", id='subscribe-spike-button', color="success", size="sm"), width=3) # Added Button
        ], className="mb-2 align-items-center"),
        dbc.Row([
            dbc.Label("Repeated Value Days:", width=4, className="text-end"),
            dbc.Col(dbc.Input(id='threshold-repeated-days', type='number', value=repeated_days, required=True, step=1, min=2, placeholder="e.g., 3"), width=4),
            dbc.Col(dbc.Button("Subscribe", id='subscribe-repeated-button', color="success", size="sm"), width=3), # Added Button
            # Removed the hint column for space, consider adding it back differently if needed
        ], className="mb-2 align-items-center"),
        dbc.Button("Update Thresholds", id="update-thresholds-button", color="warning", className="mt-3")
    ])

    if not thresholds:
        return html.Div([ html.P(f"Thresholds not currently loaded or set for site {site_id}.", className="text-warning"), form_content])
    else:
        return form_content
# <<< END: Modified build_threshold_form >>>


# --- Dash App Layout ---
app.layout = dbc.Container([
    # Stores and Location
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='data-store'),
    dcc.Store(id='site-info-store'),
    dcc.Store(id='thresholds-store'),
    dcc.Store(id='clicked-point-store'),

    # Header and Disclaimer
    html.H3("Disclaimer: App for internal use, testing, and demonstration purposes", style={'color': 'red', 'textAlign': 'center'}),
    html.H1(id='main-title', children="Data Quality Analysis", style={'textAlign': 'center'}),
    html.Hr(),

    # Notification Area
    dbc.Row(dbc.Col(html.Div(id='notification-area'), width=12)),

    # --- Control Card ---
    dbc.Card(dbc.CardBody([
        dbc.Row([
            # Site ID Input
            dbc.Col([
                dbc.Label("Site ID:", html_for="site-id-input", className="fw-bold"),
                dbc.Input(id="site-id-input", type="text", placeholder="Enter Site ID", required=False, persistence=True, persistence_type='session')
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

            # --- Action Buttons Column (Restructured with Labels - Option 1) ---
            dbc.Col([
                # --- Plot Group ---
                dbc.Label("Plot:", className="fw-bold d-block"), # Label for first group
                html.Div([ # Container for first button group
                    dbc.Button("Update Plot", id="update-button", color="primary", className="me-2"), # Added me-2 for space within group
                    dbc.Button("Reset Range", id="reset-button", color="secondary", outline=True, className=""), # No specific margin needed at end of group
                ], className="mt-1 d-flex flex-wrap"), # mt-1 adds space below label, d-flex allows wrapping

                # --- Data Group ---
                # <<< MODIFIED: Added Edit Data Button >>>
                dbc.Label("Data:", className="fw-bold d-block mt-3"), # Label for second group, mt-3 adds space above this label
                html.Div([ # Container for second button group
                     dbc.Button("Enter Measurement", id="open-enter-data-modal-button", color="info", outline=True, className="me-2", n_clicks=0),
                     dbc.Button("Upload Data", id="open-add-multiple-modal-button", color="success", outline=True, className="me-2", n_clicks=0), # Added me-2
                     dbc.Button("Edit Data", id="edit-data-button", color="warning", outline=True, className="", n_clicks=0) # Added Edit Button
                ], className="mt-1 d-flex flex-wrap") # mt-1 adds space below label, d-flex allows wrapping
                # <<< END: Modified Data Group >>>

            # Note: Removed d-flex align-items-end from this dbc.Col to allow natural top alignment for labels
            ], md=4),
            # --- End Action Buttons Column ---

            # Quick Date Selection
            dbc.Col([
                dbc.Label("Quick Dates:", className="fw-bold d-block"),
                dbc.ButtonGroup([
                    dbc.Button("Last Year", id="quick-year-button", outline=True, color="info", size="sm"),
                    dbc.Button("Last Month", id="quick-month-button", outline=True, color="info", size="sm")
                ], className="mt-2")
            ], md=2, className="text-center"),
        ], align="start", className="mb-3"), # align="start" might help vertically align the top of columns
    ]), className="mb-3 shadow-sm"), # <<< END OF CONTROL CARD

    # <<< --- ADDED TEST COMPONENTS --- >>>
    html.Div([
        html.Button("Test CS Callback", id="test-cs-button", n_clicks=0),
        html.Div(id="test-cs-output", style={'marginTop': '10px', 'border': '1px solid blue', 'padding': '5px'})
    ], style={'padding': '20px', 'border': '2px dashed red', 'margin': '10px 0'}),
    # <<< --- END OF TEST COMPONENTS --- >>>

    # --- Thresholds & Stats Row ---
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

    # --- Main Plot (Wrapped with dcc.Loading using custom_spinner) ---
    dbc.Row(dbc.Col(
        dcc.Loading(
            id="loading-plot-main",
            children=[ # The component(s) being loaded
                 html.Div(dcc.Graph(id='main-plot', config={'scrollZoom': True}), id='plot-output-container')
            ],
            custom_spinner=html.Div( # Define the custom spinner + message
                [
                    dbc.Spinner(size="lg", color="primary", spinnerClassName="me-2"), # Use spinnerClassName
                    html.Span("Please wait....Data is Loading") # Your message
                ],
                # Optional: Add some styling to center it
                style={'textAlign': 'center', 'padding': '50px'}
            )
        ),
        width=12
    )),
    # --- END Main Plot Section ---

    html.Hr(),

    # --- Data Table Section (Collapsible) ---
      dbc.Row([
        dbc.Col([
            # Heading and Toggle Button
            # <<< MODIFIED: Added ID for scrolling target >>>
            dbc.Row([
                dbc.Col(html.H4("Data Table"), width="auto"),
                dbc.Col(dbc.Button("Show/Hide Table", id="toggle-table-button", color="secondary", outline=True, size="sm", n_clicks=0), width="auto")
            ], align="center", className="mt-3 mb-2", id="data-table-section-header"), # Added id here
            # <<< END: Modified Row >>>
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

    # <<< MODIFIED: Enter Data Modal includes Time input >>>
    # Enter Data Modal (Single Entry)
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Enter New Measurement")),
        dbc.ModalBody([
            dbc.Alert(id="enter-data-modal-alert", color="danger", is_open=False, duration=4000),
            dbc.Row([
                dbc.Label("Date:", width=2),
                dbc.Col(dcc.DatePickerSingle(id='enter-date-picker', display_format='YYYY-MM-DD', date=date.today().isoformat()), width=10)
            ], className="mb-3"),
            # Added Time Input Row
            dbc.Row([
                dbc.Label("Time (HH:MM):", width=2),
                dbc.Col(dbc.Input(id='enter-time-input', type='text', placeholder="HH:MM", value="00:00"), width=10) # Added Time input
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
    # <<< END: Modified Enter Data Modal >>>

    # Add Multiple Data Points via Table Modal
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

# --- END app.layout ---


# --- Callbacks ---

# *** Main callback to handle initial load and updates ***
@callback(
    [Output('main-plot', 'figure'), # Target the Graph inside the Loading component
     Output('table-container', 'children'),
     Output('data-store', 'data'),
     Output('site-info-store', 'data'),
     Output('statistics-display', 'children'),
     Output('threshold-form-content', 'children'),
     Output('main-title', 'children'),
     Output('notification-area', 'children', allow_duplicate=True),
     Output('save-button', 'disabled'),
     Output('table-edit-status', 'is_open', allow_duplicate=True),
     Output('site-id-input', 'value'),
     Output('start-date-picker', 'date'),
     Output('end-date-picker', 'date'),
    ],
    [Input('url', 'href'),
     Input('update-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('quick-year-button', 'n_clicks'),
     Input('quick-month-button', 'n_clicks'),
     Input('thresholds-store', 'data')],
    [State('site-id-input', 'value'),
     State('start-date-picker', 'date'),
     State('end-date-picker', 'date'),
     State('thresholds-store', 'data')],
    prevent_initial_call='initial_duplicate' # Use 'initial_duplicate'
)
def update_data_and_plots(
    href,
    update_clicks, reset_clicks, year_clicks, month_clicks, threshold_data_input,
    state_site_id, state_start_date_str, state_end_date_str, current_thresholds_state):

    # ... (rest of this callback remains unchanged) ...
    triggered_input_id = ctx.triggered_id
    logger.info(f"update_data_and_plots triggered by: {triggered_input_id}")
    logger.debug(f"Current href: {href}")
    logger.debug(f"Current states: site='{state_site_id}', start='{state_start_date_str}', end='{state_end_date_str}'")


    # --- Default values and initial setup ---
    empty_fig = go.Figure(layout={
        'xaxis': {'visible': False},
        'yaxis': {'visible': False},
        'annotations': [{
            'text': 'No Data Loaded or Found',
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 20}
        }]
    })
    initial_table_msg = dbc.Alert("Enter Site ID and Date Range, then click 'Update Plot'.", color="info")
    error_alert = None
    site_info_output = None
    stats_display = html.P("Load data to view statistics.")
    threshold_form = html.P("Load site data to view/edit thresholds.")
    main_title = "Data Quality Analysis"
    save_disabled = True
    edit_status_open = False

    # --- Variables to hold the parameters for this run ---
    site_id_to_load = None
    start_date_to_load = None # Use string format<y_bin_925>-MM-DD for generate_plot function
    end_date_to_load = None   # Use string format