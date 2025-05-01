# -*- coding: utf-8 -*-
# --- Imports ---
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, no_update, ctx
import dash_bootstrap_components as dbc

import logsging
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

# --- Client-side callback for scrolling ---
# <<< ADDED: Client-side callback for scroll functionality >>>
app.clientside_callback(
    """
    function(n_clicks) {
        // Check if the button was actually clicked (n_clicks > 0)
        // n_clicks starts at 0 or None, increments on click
        if (n_clicks && n_clicks > 0) {
            // Find the target element by its ID
            const element = document.getElementById('data-table-section-header');
            if (element) {
                // Scroll the element into view
                element.scrollIntoView({
                    behavior: 'smooth', // Use smooth scrolling
                    block: 'start'      // Align the top of the element to the top of the viewport
                });
            } else {
                console.warn("Scroll target 'data-table-section-header' not found.");
            }
        }
        // No Dash output needs to be updated by this callback
        return window.dash_clientside.no_update;
    }
    """,
    # No Output needed, the callback performs a browser action directly
    Input('edit-data-button', 'n_clicks'),
    prevent_initial_call=True # Important to prevent firing on page load
)
# <<< END: Client-side callback section >>>


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
    ]), className="mb-3 shadow-sm"),

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
    start_date_to_load = None # Use string format YYYY-MM-DD for generate_plot function
    end_date_to_load = None   # Use string format YYYY-MM-DD for generate_plot function
    is_reset_action = False

    # --- Variables to hold the values to update the UI controls ---
    output_site_id = no_update
    output_start_date = no_update # Use ISO format string for DatePicker
    output_end_date = no_update   # Use ISO format string for DatePicker

    # --- Determine parameters based on trigger ---
    is_initial_load = triggered_input_id == 'url'
    is_update_button = triggered_input_id == 'update-button'
    is_reset_button = triggered_input_id == 'reset-button'
    is_quick_year = triggered_input_id == 'quick-year-button'
    is_quick_month = triggered_input_id == 'quick-month-button'
    is_threshold_update = triggered_input_id == 'thresholds-store'

    try:
        today = date.today() # Use date directly
        default_end = today
        default_start = today - timedelta(days=30)

        if is_initial_load:
            logger.info("Triggered by URL (initial load or refresh).")
            if href:
                try:
                    parsed_url = urlparse(href)
                    query_params = parse_qs(parsed_url.query)
                    url_site_id = query_params.get('id', [None])[0]
                    url_start_str = query_params.get('start_date', [None])[0]
                    url_end_str = query_params.get('end_date', [None])[0]
                    logger.info(f"Parsed URL: id='{url_site_id}', start='{url_start_str}', end='{url_end_str}'")

                    if url_site_id:
                        site_id_to_load = url_site_id
                        output_site_id = url_site_id # Update the input field

                        temp_start_date = default_start
                        temp_end_date = default_end

                        if url_start_str:
                            try:
                                parsed_start = datetime.strptime(url_start_str, '%Y-%m-%d').date()
                                temp_start_date = parsed_start
                            except ValueError:
                                logger.warning(f"Invalid start_date format '{url_start_str}' in URL. Using default.")
                                error_alert = dbc.Alert(f"Warning: Invalid start date '{url_start_str}' in URL. Using default.", color="warning", dismissable=True, duration=5000)

                        if url_end_str:
                            try:
                                parsed_end = datetime.strptime(url_end_str, '%Y-%m-%d').date()
                                temp_end_date = parsed_end
                            except ValueError:
                                logger.warning(f"Invalid end_date format '{url_end_str}' in URL. Using default.")
                                error_alert = dbc.Alert(f"Warning: Invalid end date '{url_end_str}' in URL. Using default.", color="warning", dismissable=True, duration=5000)


                        if temp_start_date > temp_end_date:
                            logger.warning(f"URL dates invalid (start > end). Using defaults.")
                            start_date_to_load = default_start.strftime('%Y-%m-%d')
                            end_date_to_load = default_end.strftime('%Y-%m-%d')
                            output_start_date = default_start.isoformat()
                            output_end_date = default_end.isoformat()
                            error_alert = dbc.Alert(f"Warning: Start date {temp_start_date.isoformat()} is after end date {temp_end_date.isoformat()}. Using defaults.", color="warning", dismissable=True, duration=5000)

                        else:
                            start_date_to_load = temp_start_date.strftime('%Y-%m-%d')
                            end_date_to_load = temp_end_date.strftime('%Y-%m-%d')
                            output_start_date = temp_start_date.isoformat()
                            output_end_date = temp_end_date.isoformat()
                    else:
                        # No site_id in URL, use defaults for UI, don't load data
                        logger.info("No site_id found in URL. Using default UI values.")
                        output_site_id = None # Or ""
                        output_start_date = default_start.isoformat()
                        output_end_date = default_end.isoformat()
                        # Don't set site_id_to_load, so data fetch is skipped below

                except Exception as e:
                    logger.error(f"Error parsing URL '{href}': {e}", exc_info=True)
                    error_alert = dbc.Alert(f"Error parsing URL parameters: {e}", color="danger", dismissable=True)
                    # Use defaults for UI
                    output_site_id = None
                    output_start_date = default_start.isoformat()
                    output_end_date = default_end.isoformat()
            else:
                 # No href (shouldn't normally happen if Location is present)
                 logger.warning("href is empty on initial load trigger.")
                 output_site_id = None
                 output_start_date = default_start.isoformat()
                 output_end_date = default_end.isoformat()

        elif is_reset_button:
            logger.info(f"Reset button triggered for site {state_site_id}.")
            if not state_site_id:
                 error_alert = dbc.Alert("Please enter a Site ID before resetting.", color="warning", dismissable=True)
                 # Keep current UI state, don't load
                 return no_update, no_update, no_update, no_update, no_update, no_update, no_update, error_alert, True, False, state_site_id, state_start_date_str, state_end_date_str
            site_id_to_load = state_site_id
            start_date_to_load = None # Signal to generate_plot to use defaults
            end_date_to_load = None
            is_reset_action = True
            output_site_id = state_site_id # Keep site ID input value

        elif is_quick_year or is_quick_month:
             logger.info(f"Quick date '{triggered_input_id}' triggered for site {state_site_id}.")
             if not state_site_id:
                  error_alert = dbc.Alert("Please enter a Site ID before using Quick Dates.", color="warning", dismissable=True)
                  return no_update, no_update, no_update, no_update, no_update, no_update, no_update, error_alert, True, False, state_site_id, state_start_date_str, state_end_date_str
             site_id_to_load = state_site_id
             end_dt_obj = today
             start_dt_obj = today - timedelta(days=365 if is_quick_year else 30)
             start_date_to_load = start_dt_obj.strftime('%Y-%m-%d')
             end_date_to_load = end_dt_obj.strftime('%Y-%m-%d')
             output_site_id = state_site_id # Keep site ID
             output_start_date = start_dt_obj.isoformat() # Update date pickers
             output_end_date = end_dt_obj.isoformat()

        elif is_update_button or is_threshold_update:
            if is_threshold_update:
                logger.info(f"Threshold update triggered plot refresh for site {state_site_id}.")
            else:
                logger.info(f"'Update Plot' button triggered for site {state_site_id}.")

            if not state_site_id:
                error_alert = dbc.Alert("Please enter a Site ID.", color="danger", dismissable=True)
                current_start = state_start_date_str or default_start.isoformat()
                current_end = state_end_date_str or default_end.isoformat()
                return empty_fig, initial_table_msg, None, None, stats_display, build_threshold_form(None, {}, "?"), main_title, error_alert, True, False, None, current_start, current_end

            site_id_to_load = state_site_id
            output_site_id = state_site_id # Keep current site ID in input

            if not state_start_date_str or not state_end_date_str:
                error_alert = dbc.Alert("Please select both Start and End Dates.", color="danger", dismissable=True)
                current_start = state_start_date_str or default_start.isoformat()
                current_end = state_end_date_str or default_end.isoformat()
                return empty_fig, initial_table_msg, None, {'site_id': site_id_to_load}, stats_display, build_threshold_form(site_id_to_load, current_thresholds_state, "?"), f"Error - {site_id_to_load}", error_alert, True, False, site_id_to_load, current_start, current_end

            try:
                start_dt_obj = datetime.strptime(state_start_date_str[:10], '%Y-%m-%d').date()
                end_dt_obj = datetime.strptime(state_end_date_str[:10], '%Y-%m-%d').date()
                if start_dt_obj > end_dt_obj:
                    error_alert = dbc.Alert("Start Date cannot be after End Date.", color="danger", dismissable=True)
                    return empty_fig, initial_table_msg, None, {'site_id': site_id_to_load}, stats_display, build_threshold_form(site_id_to_load, current_thresholds_state, "?"), f"Date Error - {site_id_to_load}", error_alert, True, False, site_id_to_load, state_start_date_str, state_end_date_str
                start_date_to_load = start_dt_obj.strftime('%Y-%m-%d')
                end_date_to_load = end_dt_obj.strftime('%Y-%m-%d')
                output_start_date = state_start_date_str
                output_end_date = state_end_date_str

            except (ValueError, TypeError):
                 error_alert = dbc.Alert("Invalid date format selected.", color="danger", dismissable=True)
                 return empty_fig, initial_table_msg, None, {'site_id': site_id_to_load}, stats_display, build_threshold_form(site_id_to_load, current_thresholds_state, "?"), f"Date Error - {site_id_to_load}", error_alert, True, False, site_id_to_load, state_start_date_str, state_end_date_str
        else:
            logger.debug(f"Callback triggered by '{triggered_input_id}' - no action needed or handled previously.")
            if is_initial_load and not site_id_to_load:
                 logger.info("Initial load without site ID in URL - showing empty state.")
                 return empty_fig, initial_table_msg, None, None, stats_display, build_threshold_form(None, {}, "?"), main_title, error_alert, True, False, output_site_id, output_start_date, output_end_date
            # If trigger was something else (like thresholds-store but not handled above), return no_update
            # This case might be hit if thresholds-store updates but site_id is not set yet.
            if triggered_input_id == 'thresholds-store' and not state_site_id:
                 logger.debug("Threshold update triggered, but no site ID. No plot update.")
                 # Return current state or empty state? Let's return empty state.
                 current_start = state_start_date_str or default_start.isoformat()
                 current_end = state_end_date_str or default_end.isoformat()
                 return empty_fig, initial_table_msg, None, None, stats_display, build_threshold_form(None, {}, "?"), main_title, error_alert, True, False, None, current_start, current_end

            return (no_update,) * 13 # Return no_update for all outputs


    except Exception as date_e:
        logger.error(f"Error processing inputs/dates: {date_e}", exc_info=True)
        error_alert = dbc.Alert(f"Error processing inputs: {date_e}", color="danger", dismissable=True)
        out_site = state_site_id if triggered_input_id != 'url' else None
        out_start = state_start_date_str if state_start_date_str else default_start.isoformat()
        out_end = state_end_date_str if state_end_date_str else default_end.isoformat()
        thresh = build_threshold_form(out_site, current_thresholds_state, "?")
        return empty_fig, initial_table_msg, None, {'site_id': out_site}, stats_display, thresh, f"Input Error - {out_site}", error_alert, True, False, out_site, out_start, out_end

    # --- Data Fetching (only if site_id_to_load is set) ---
    if not site_id_to_load:
         logger.info("No site ID determined for loading. Skipping data fetch.")
         # Ensure UI defaults are set correctly even if no fetch occurs
         final_output_start = output_start_date if output_start_date is not no_update else default_start.isoformat()
         final_output_end = output_end_date if output_end_date is not no_update else default_end.isoformat()
         final_output_site_id = output_site_id if output_site_id is not no_update else None
         return empty_fig, initial_table_msg, None, None, stats_display, build_threshold_form(None, {}, "?"), main_title, error_alert, True, False, final_output_site_id, final_output_start, final_output_end


    logger.info(f"Proceeding to fetch data: Site='{site_id_to_load}', Start='{start_date_to_load}', End='{end_date_to_load}', Reset={is_reset_action}, ThresholdTrigger={is_threshold_update}")

    try:
        # Use thresholds from state if trigger was threshold update, otherwise let function load defaults/from file
        thresholds_to_use = current_thresholds_state if is_threshold_update and current_thresholds_state else None
        fig, df_processed, err_func, name_func, final_start, final_end, units_val, found_thresholds, stats_dict = generate_plot_for_site(
            site_id_to_load,
            start_date_to_load, # Can be None for reset
            end_date_to_load,   # Can be None for reset
            is_reset_action,
            logger,
            thresholds_override=thresholds_to_use
            )

        # Store comprehensive site info
        site_info_output = {'site_id': site_id_to_load, 'name': name_func, 'units': units_val, 'start': final_start, 'end': final_end}
        # Determine thresholds to display in form (use newly found/updated ones preferentially)
        thresholds_for_form = found_thresholds if found_thresholds else (current_thresholds_state if current_thresholds_state else {})

        # Validate dates returned from generate_plot_for_site before using them
        valid_final_start = None
        valid_final_end = None
        if final_start:
            try:
                datetime.strptime(final_start, '%Y-%m-%d')
                valid_final_start = final_start
            except (ValueError, TypeError):
                 logger.warning(f"generate_plot_for_site returned invalid final_start: {final_start}")
        if final_end:
             try:
                 datetime.strptime(final_end, '%Y-%m-%d')
                 valid_final_end = final_end
             except (ValueError, TypeError):
                  logger.warning(f"generate_plot_for_site returned invalid final_end: {final_end}")

        # Update UI date pickers based on action and valid returned dates
        if is_reset_action and valid_final_start and valid_final_end:
            output_start_date = valid_final_start
            output_end_date = valid_final_end
        elif is_initial_load or is_quick_year or is_quick_month:
            # Keep output dates determined earlier in the callback
            pass
        else: # Regular update or threshold change, keep state dates
             output_start_date = state_start_date_str
             output_end_date = state_end_date_str

        # Handle errors from generate_plot_for_site
        if err_func:
            error_alert = dbc.Alert(f"Error loading data for {site_id_to_load}: {err_func}", color="danger", dismissable=True)
            main_title = f"{name_func or '?'} ({site_id_to_load}) - Data Error"
            threshold_form = build_threshold_form(site_id_to_load, thresholds_for_form, units_val or "?")
            # Ensure date pickers retain their current values on error
            final_output_start = output_start_date if output_start_date is not no_update else state_start_date_str
            final_output_end = output_end_date if output_end_date is not no_update else state_end_date_str
            return empty_fig, initial_table_msg, None, site_info_output, stats_display, threshold_form, main_title, error_alert, True, False, site_id_to_load, final_output_start, final_output_end

        # Handle case where no data is found for the period
        if fig is None or df_processed is None or df_processed.empty:
            display_start = valid_final_start or start_date_to_load or "N/A"
            display_end = valid_final_end or end_date_to_load or "N/A"
            error_alert = dbc.Alert(f"No data found for site {site_id_to_load} in the selected period ({display_start} to {display_end}).", color="warning", dismissable=True)
            main_title = f"{name_func or '?'} ({site_id_to_load})"
            threshold_form = build_threshold_form(site_id_to_load, thresholds_for_form, units_val or "?")
            # Ensure date pickers retain their current values
            final_output_start = output_start_date if output_start_date is not no_update else state_start_date_str
            final_output_end = output_end_date if output_end_date is not no_update else state_end_date_str
            return empty_fig, initial_table_msg, None, site_info_output, stats_display, threshold_form, main_title, error_alert, True, False, site_id_to_load, final_output_start, final_output_end

        # --- Success Case ---
        logger.info(f"Data loaded successfully for {site_id_to_load}. Range: {final_start} to {final_end}. Units: {units_val}")
        table_component = create_dash_data_table(df_processed.copy(), units_val, 'editable-data-table', logger)
        if not isinstance(table_component, dash_table.DataTable):
             error_alert = table_component # Show alert if table creation failed
             table_component = html.Div() # Prevent error by providing empty div
             logger.error("Failed to create data table component.")
             save_disabled = True # Disable save if table fails
             edit_status_open = False
        else:
             save_disabled = False # Enable save if table created
             edit_status_open = True # Show edit status bar

        # Prepare data for storage (convert datetime back to string)
        df_store = df_processed.copy()
        if 'Date' in df_store.columns:
             df_store['Date'] = df_store['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        stored_data = df_store.to_json(orient='split', date_format='iso')

        # Build Threshold form and Stats display
        threshold_form = build_threshold_form(site_id_to_load, thresholds_for_form, units_val)
        if stats_dict:
             stats_display = html.Div([
                 html.P([html.Strong("Count: "), html.Span(stats_dict.get('count'))]),
                 html.P([html.Strong(f"Mean ({stats_dict.get('units', '?')}): "), html.Span(f"{stats_dict.get('mean'):.2f}" if isinstance(stats_dict.get('mean'), (int, float)) else stats_dict.get('mean'))]),
                 html.P([html.Strong(f"Min ({stats_dict.get('units', '?')}): "), html.Span(f"{stats_dict.get('min'):.2f}" if isinstance(stats_dict.get('min'), (int, float)) else stats_dict.get('min'))]),
                 html.P([html.Strong(f"Max ({stats_dict.get('units', '?')}): "), html.Span(f"{stats_dict.get('max'):.2f}" if isinstance(stats_dict.get('max'), (int, float)) else stats_dict.get('max'))])
             ], className="small")
        else: stats_display = html.P("Statistics not available.")

        main_title = f"{name_func or '?'} ({site_id_to_load}) | {final_start} to {final_end}"

        # Ensure final date outputs are correct ISO strings
        final_output_start = output_start_date if output_start_date is not no_update else valid_final_start
        final_output_end = output_end_date if output_end_date is not no_update else valid_final_end

        return fig, table_component, stored_data, site_info_output, stats_display, threshold_form, main_title, error_alert, save_disabled, edit_status_open, site_id_to_load, final_output_start, final_output_end

    except Exception as e:
        logger.error(f"Unhandled exception during plot/table generation for site {site_id_to_load}: {e}", exc_info=True)
        error_alert = dbc.Alert(f"An unexpected server error occurred: {e}", color="danger", dismissable=True)
        main_title = f"Error Processing {site_id_to_load}"
        # Attempt to build threshold form even on error
        try:
             # Use state thresholds if available, else attempt fetch, else empty
             thresholds_state_to_use = current_thresholds_state if current_thresholds_state else (get_site_thresholds(site_id_to_load, logger) if site_id_to_load else {})
             if thresholds_state_to_use is None: thresholds_state_to_use = {}
        except Exception as te:
             logger.error(f"Failed to get thresholds after main error: {te}")
             thresholds_state_to_use = {}
        threshold_form = build_threshold_form(site_id_to_load, thresholds_state_to_use, "?")

        out_site = site_id_to_load
        default_start_iso = (date.today() - timedelta(days=30)).isoformat()
        default_end_iso = date.today().isoformat()
        # Ensure date pickers retain state or default
        out_start = output_start_date if output_start_date is not no_update else (state_start_date_str or default_start_iso)
        out_end = output_end_date if output_end_date is not no_update else (state_end_date_str or default_end_iso)

        return empty_fig, initial_table_msg, None, {'site_id': site_id_to_load}, stats_display, threshold_form, main_title, error_alert, True, False, out_site, out_start, out_end


# --- Other Existing Callbacks ---

@callback(
    Output("table-collapse", "is_open"),
    Input("toggle-table-button", "n_clicks"),
    State("table-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_table_collapse(n_clicks, is_open):
    if n_clicks:
        logger.debug(f"Toggling table collapse. Current state: {is_open}, New state: {not is_open}")
        return not is_open
    logger.debug("Toggle table callback triggered but n_clicks is 0 or None.")
    return is_open

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
    if not n_clicks or not site_id:
        logger.debug("Update thresholds callback triggered without click or site_id.")
        return no_update, no_update
    logger.info(f"Attempting threshold update for site {site_id} from form.")
    error_messages = []
    max_val, spike_roc, repeated_days = None, None, None # Initialize
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
                # Return updated thresholds to trigger plot refresh via thresholds-store input
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
    if stored_json is None or edited_table_data is None or site_info is None:
        logger.warning("Table edit callback skipped - missing essential state.")
        return no_update, no_update, no_update, True
    if thresholds is None:
        logger.warning("Thresholds are missing. Proceeding with edit but cannot re-flag.")
        thresholds = {} # Use empty dict if missing
    units = site_info.get('units', '?')
    site_id = site_info.get('site_id', 'N/A')
    notification = None
    edit_status_open = True
    if previous_table_data is None or edited_table_data == previous_table_data:
        logger.debug("No actual change detected in table data or no previous data.")
        return no_update, no_update, no_update, edit_status_open # No changes or first load

    try:
        df_edited = pd.DataFrame(edited_table_data)
        df_prev = pd.DataFrame(previous_table_data)

        # Ensure columns exist before comparison/conversion
        if 'Discharge' not in df_edited.columns or 'Discharge' not in df_prev.columns:
             logger.warning("Discharge column missing in edited or previous data during handle_table_edit.")
             return no_update, no_update, dbc.Alert("Table structure error during edit.", color="danger"), edit_status_open

        # Find changes specifically in the 'Discharge' column
        discharge_edited_num = pd.to_numeric(df_edited['Discharge'], errors='coerce')
        discharge_prev_num = pd.to_numeric(df_prev['Discharge'], errors='coerce')

        # Use pandas comparison that handles NaNs correctly
        diff_mask = ~discharge_edited_num.equals(discharge_prev_num)
        if not diff_mask and len(discharge_edited_num) == len(discharge_prev_num):
             try:
                 comparison = discharge_edited_num.compare(discharge_prev_num, keep_equal=False, keep_shape=True)
                 if not comparison.notna().any().any():
                     logger.debug("No change detected in 'Discharge' column after checking NaNs.")
                     return no_update, no_update, no_update, edit_status_open
             except Exception as compare_err:
                  logger.warning(f"Error during Series comparison, proceeding as if changed: {compare_err}")

        # Find the index of the first changed row
        try:
            changed_indices = discharge_edited_num.compare(discharge_prev_num, keep_equal=False).index
            if not changed_indices.any():
                logger.debug("No numeric change detected in Discharge column.")
                return no_update, no_update, no_update, edit_status_open
        except Exception as compare_idx_err:
            logger.warning(f"Error getting changed indices from compare, trying simple diff: {compare_idx_err}")
            changed_indices = df_edited.index[discharge_edited_num != discharge_prev_num]
            if not changed_indices.any():
                 logger.debug("No change detected in Discharge column (fallback check).")
                 return no_update, no_update, no_update, edit_status_open

        changed_view_idx = changed_indices[0] # Focus on the first change

        df_store_orig = pd.read_json(stored_json, orient='split')
        if 'Date' in df_store_orig:
            # Convert stored dates (likely strings) to datetime for comparison/indexing
            df_store_orig['Date'] = pd.to_datetime(df_store_orig['Date'], errors='coerce')
        else:
             logger.error("Original data store is missing 'Date' column.")
             return no_update, no_update, dbc.Alert("Data store integrity error.", color="danger"), edit_status_open


        if changed_view_idx >= len(df_store_orig):
            logger.error(f"Edit index {changed_view_idx} is out of bounds for stored DataFrame (length {len(df_store_orig)}).")
            raise IndexError("Edit index mismatch between table view and stored data.")

        # Map the visible row index (from table `data`) to the original DataFrame index
        # This assumes the order hasn't drastically changed; more robust mapping might be needed if sorting/filtering is complex
        original_df_index = df_store_orig.index[changed_view_idx]

        new_value_edited = df_edited.loc[changed_view_idx, 'Discharge']
        if 'DISCHARGE' not in df_store_orig.columns:
            logger.error("Original data store is missing 'DISCHARGE' column.")
            return no_update, no_update, dbc.Alert("Data store integrity error (missing DISCHARGE).", color="danger"), edit_status_open
        old_value_original = df_store_orig.loc[original_df_index, 'DISCHARGE']

        logger.info(f"Table edit detected: Original DF Index={original_df_index}, Col='Discharge', Old='{old_value_original}', New='{new_value_edited}'")

        new_numeric_value = np.nan
        if pd.isna(new_value_edited) or str(new_value_edited).strip() == '':
            new_numeric_value = np.nan
        else:
            try:
                new_numeric_value = float(new_value_edited)
            except (ValueError, TypeError):
                logger.error(f"Invalid numeric input '{new_value_edited}' at view index {changed_view_idx} (Original Index {original_df_index}).")
                # Revert table display to the original state before the invalid edit
                reverted_table = create_dash_data_table(df_store_orig.copy(), units, 'editable-data-table', logger)
                return no_update, reverted_table, dbc.Alert(f"Invalid input: '{new_value_edited}' is not a valid number. Edit reverted.", color="danger"), edit_status_open

        df_updated = df_store_orig.copy()
        df_updated.loc[original_df_index, 'DISCHARGE'] = new_numeric_value
        df_updated.loc[original_df_index, 'ReviewStatus'] = 'Edited'

        if 'Qualifiers' in df_updated.columns:
            current_qual = df_updated.loc[original_df_index, 'Qualifiers']
            if pd.isna(current_qual) or str(current_qual).strip() == '':
                df_updated.loc[original_df_index, 'Qualifiers'] = 'Edited'
            elif 'Edited' not in str(current_qual).split(';'):
                 df_updated.loc[original_df_index, 'Qualifiers'] = f"{str(current_qual).strip()};Edited"
        else:
            # If Qualifiers column doesn't exist, create it
            df_updated['Qualifiers'] = pd.Series(dtype='object')
            df_updated.loc[original_df_index, 'Qualifiers'] = 'Edited'


        logger.info(f"Re-flagging data for site {site_id} after edit at index {original_df_index}...")
        if not thresholds:
            logger.warning("Cannot re-flag: Thresholds are missing.")
            notification = dbc.Alert("Change applied, but cannot re-flag (thresholds missing). Click 'Save Changes'.", color="warning", duration=5000, dismissable=True)
        else:
            try:
                # Ensure Date column is datetime before flagging
                if 'Date' in df_updated and not pd.api.types.is_datetime64_any_dtype(df_updated['Date']):
                     df_updated['Date'] = pd.to_datetime(df_updated['Date'], errors='coerce')
                df_updated = apply_flagging(df_updated, thresholds, logger)
                logger.info("Re-flagging complete.")
                notification = dbc.Alert("Change processed and data re-flagged. Click 'Save Changes' to export.", color="warning", duration=5000, dismissable=True)
            except Exception as flag_e:
                logger.error(f"Error during re-flagging after edit: {flag_e}", exc_info=True)
                notification = dbc.Alert(f"Change applied, but error during re-flagging: {flag_e}", color="danger", duration=5000, dismissable=True)

        # Prepare updated data for storage (convert datetime back to string)
        df_updated_store = df_updated.copy()
        if 'Date' in df_updated_store and pd.api.types.is_datetime64_any_dtype(df_updated_store['Date']):
            df_updated_store['Date'] = df_updated_store['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        updated_json_output = df_updated_store.to_json(orient='split', date_format='iso')

        # Prepare data for table display (ensure datetime objects for date column)
        df_table_update = df_updated.copy()
        if 'Date' in df_table_update and not pd.api.types.is_datetime64_any_dtype(df_table_update['Date']):
            df_table_update['Date'] = pd.to_datetime(df_table_update['Date'], errors='coerce')
        new_table_component = create_dash_data_table(df_table_update, units, 'editable-data-table', logger)

        # Check if table creation failed
        if not isinstance(new_table_component, dash_table.DataTable):
            logger.error("Failed to recreate DataTable component after edit.")
            # Provide error message instead of the table component itself
            original_table_component = create_dash_data_table(df_store_orig.copy(), units, 'editable-data-table', logger) # Show original table
            notification = dbc.Alert("Edit applied, but failed to update table display. Edit reverted in table.", color="danger")
            return no_update, original_table_component, notification, edit_status_open # Return no_update for store

        logger.info(f"Successfully processed table edit for index {original_df_index}.")
        return updated_json_output, new_table_component, notification, edit_status_open

    except Exception as e:
        logger.error(f"Error handling table edit for site {site_id}: {e}", exc_info=True)
        notification = dbc.Alert(f"Error processing table edit: {e}", color="danger", dismissable=True)
        try:
            # Attempt to restore the table to its state before the error
            df_original_state = pd.read_json(stored_json, orient='split')
            if 'Date' in df_original_state:
                 df_original_state['Date'] = pd.to_datetime(df_original_state['Date'], errors='coerce')
            original_table_component = create_dash_data_table(df_original_state.copy(), units, 'editable-data-table', logger)
            # Don't update the data store, revert the table display
            return no_update, original_table_component, notification, edit_status_open
        except Exception as revert_e:
            logger.error(f"Failed to revert table display after edit error: {revert_e}")
            # If reverting fails, show a generic error message
            return no_update, html.Div("Error displaying table after edit failure."), notification, edit_status_open


@callback(
    Output('save-status', 'children'),
    Input('save-button', 'n_clicks'),
    [State('data-store', 'data'),
     State('site-info-store', 'data')],
    prevent_initial_call=True
)
def save_data(n_clicks, stored_json, site_info):
    if not n_clicks: return ""
    if stored_json is None or site_info is None:
        logger.warning("Save button clicked, but no data or site info found.")
        return dbc.Alert("No data loaded to save.", color="warning", dismissable=True)
    site_id = site_info.get('site_id', 'unknown_site')
    logger.info(f"Save button clicked for site {site_id}.")
    try:
        df_save = pd.read_json(stored_json, orient='split')
        if 'Date' in df_save.columns:
            # NOTE: Adjust if storing full timestamps; strftime format might need change
            # Format Date as YYYY-MM-DD for CSV output
            df_save['Date'] = pd.to_datetime(df_save['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
            if df_save['Date'].isnull().any(): logger.warning("Some dates were invalid during save conversion.")
        else:
            logger.error("Cannot save: 'Date' column missing.")
            return dbc.Alert("Save Error: 'Date' column missing.", color="danger", dismissable=True)

        if 'DISCHARGE' in df_save.columns:
             df_save['DISCHARGE'] = pd.to_numeric(df_save['DISCHARGE'], errors='coerce').round(2)
        else:
             df_save['DISCHARGE'] = np.nan

        # Ensure essential columns exist before selecting
        if 'ReviewStatus' not in df_save.columns: df_save['ReviewStatus'] = 'Unknown'
        if 'Qualifiers' not in df_save.columns: df_save['Qualifiers'] = None

        # Recreate Qualified and Active Flags based on potentially modified data
        df_save['Qualified'] = df_save['Qualifiers'].notna().map({True: 'Yes', False: 'No'})

        active_flags_list = []; flag_map = {'FLAG_LESS_THAN_Min._Value': 'Below Min', 'FLAG_ZERO': 'Zero', 'FLAG_REPEATED': 'Repeated', 'FLAG_GREATER_THAN_MaxValue': 'Above Max', 'UNUSUAL_SPIKE': 'Spike', 'FLAG_BELOW_CAPACITY': 'Below Capacity'}
        flag_cols_to_check = [col for col in flag_map if col in df_save.columns]
        if flag_cols_to_check:
            for _, row in df_save.iterrows():
                try:
                    active = [flag_map[col] for col in flag_cols_to_check if pd.notna(row.get(col)) and row.get(col) == True]
                    active_flags_list.append(', '.join(active) if active else 'None')
                except Exception as e:
                     logger.error(f"Error processing flags for save file row: {e}"); active_flags_list.append('Error')
        else: active_flags_list = ['N/A'] * len(df_save)
        if len(active_flags_list) != len(df_save):
             logger.error("Length mismatch saving Active Flags column. Padding with 'Error'.")
             active_flags_list.extend(['Error'] * (len(df_save) - len(active_flags_list)))

        df_save['Active Flags'] = active_flags_list

        # Define the desired order and selection of columns for the CSV
        base_columns = ['Date', 'DISCHARGE', 'ReviewStatus', 'Qualifiers', 'Qualified', 'Active Flags']
        flag_columns = ['FLAG_LESS_THAN_Min._Value','FLAG_ZERO','FLAG_REPEATED','FLAG_GREATER_THAN_MaxValue','UNUSUAL_SPIKE','FLAG_BELOW_CAPACITY', 'FLAGGED']
        existing_flag_columns = [col for col in flag_columns if col in df_save.columns]
        columns_to_save_ordered = base_columns + existing_flag_columns

        # Ensure only columns that actually exist in the DataFrame are selected
        columns_that_exist = [col for col in columns_to_save_ordered if col in df_save.columns]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"edited_data_{site_id}_{timestamp}.csv"
        save_path = Path(".") / filename # Save in the current working directory
        logger.info(f"Saving columns: {columns_that_exist} to file: {save_path}")

        # Save the selected columns to CSV
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
    if not clickData or not figure or 'points' not in clickData or not clickData['points']:
        logger.debug("Plot click detected, but no valid point data found.")
        return False, no_update, no_update
    point_data = clickData['points'][0]; curve_index = point_data.get('curveNumber', -1)
    logger.debug(f"Plot click details: {point_data}")
    open_modal = False; modal_body_content = "No details available."; clicked_point_info = None
    try:
        if 'data' in figure and 0 <= curve_index < len(figure.get('data', [])):
            trace = figure['data'][curve_index]
            original_index = None
            # Check if the click was on a marker trace with customdata
            if 'customdata' in point_data and point_data['customdata'] is not None and 'markers' in trace.get('mode', ''):
                custom_data = point_data['customdata']
                # Extract original index (assuming it's the first element)
                if isinstance(custom_data, (list, tuple)) and len(custom_data) > 0:
                    original_index = custom_data[0]
                # Handle cases where customdata might be a single value
                elif isinstance(custom_data, (int, str, float)):
                    original_index = custom_data
                else:
                     logger.warning(f"Unexpected customdata format found: {type(custom_data)}")

            if original_index is not None:
                # Attempt to get flag type info
                flag_type = "Flagged Point"; point_number = point_data.get('pointNumber', -1)
                # Prefer 'meta' if available (more specific)
                if 'meta' in trace and isinstance(trace.get('meta'), list) and 0 <= point_number < len(trace['meta']):
                    flag_type = trace['meta'][point_number]
                # Fallback to trace name
                elif 'name' in trace:
                    flag_type = trace.get('name', flag_type)

                date_str = point_data.get('x'); value = point_data.get('y')
                # Note: date_str from plot might include time if axis is datetime
                value_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                modal_body_content = html.Div([
                    html.P([html.Strong("Date/Time: "), html.Span(date_str)]), # Changed label slightly
                    html.P([html.Strong("Value: "), html.Span(value_str)]),
                    html.P([html.Strong("Flag Type: "), html.Span(flag_type)]),
                    html.P(f"(Original Index: {original_index})", className="small text-muted")
                 ])
                # Store all relevant info for the QC action callback
                clicked_point_info = {'original_index': original_index, 'date': date_str, 'value': value, 'flag_type': flag_type}
                open_modal = True
                logger.info(f"Flagged point clicked: Index={original_index}, Date={date_str}, Value={value_str}, Flag={flag_type}")
            else:
                 logger.debug(f"Click not on flagged marker or customdata missing/invalid in trace {curve_index}.")
        else:
             logger.warning(f"Invalid curve_index {curve_index} from click data or figure data missing.")
    except Exception as e:
        logger.error(f"Error processing plot click data: {e}", exc_info=True)
        modal_body_content = f"Error processing click: {e}"; open_modal = False; clicked_point_info = None

    if open_modal:
        return True, modal_body_content, clicked_point_info
    else:
        # Don't open modal if click wasn't valid or on a non-flagged point
        return False, no_update, None

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
     State('thresholds-store', 'data'),
     State('main-plot', 'figure')],
    prevent_initial_call=True
)
def handle_qc_action(approve_clicks, interpolate_clicks, delete_clicks, close_clicks,
                     clicked_point_data, stored_json, site_info, thresholds, current_figure):
    triggered_button_id = ctx.triggered_id
    modal_is_open = False # Close modal after action by default
    notification = no_update
    updated_data_store = no_update
    updated_table = no_update
    updated_plot = no_update

    if triggered_button_id == 'qc-close-button':
        logger.debug("QC modal closed via Close button.")
        return modal_is_open, notification, updated_data_store, updated_table, updated_plot

    if not triggered_button_id:
        logger.warning("QC action callback triggered without a button ID.")
        return modal_is_open, notification, updated_data_store, updated_table, updated_plot

    if not clicked_point_data or 'original_index' not in clicked_point_data:
        logger.warning(f"QC action '{triggered_button_id}' triggered, but no valid point selected.")
        notification = dbc.Alert("No point selected or index missing.", color="warning", dismissable=True)
        return False, notification, updated_data_store, updated_table, updated_plot # Keep modal open? No, action failed.

    if not stored_json:
        logger.error(f"QC action '{triggered_button_id}' triggered, but data-store is empty.")
        notification = dbc.Alert("Cannot perform action: Data not found.", color="danger", dismissable=True)
        return modal_is_open, notification, updated_data_store, updated_table, updated_plot

    if not site_info:
        logger.error(f"QC action '{triggered_button_id}' triggered, but site-info-store is empty.")
        notification = dbc.Alert("Cannot perform action: Site info not found.", color="danger", dismissable=True)
        return modal_is_open, notification, updated_data_store, updated_table, updated_plot

    original_index_from_click = clicked_point_data.get('original_index')
    date_clicked = clicked_point_data.get('date') # This might be a string with time
    value_clicked = clicked_point_data.get('value')
    logger.info(f"Processing QC Action '{triggered_button_id}' for index: {original_index_from_click}, Date/Time: {date_clicked}, Value: {value_clicked}")

    action_performed = None; df = None

    try:
        df = pd.read_json(stored_json, orient='split')
        if 'Date' in df.columns:
            # Convert stored dates (likely strings) to datetime objects for processing
            # NOTE: Decide if you need to normalize here or keep full timestamp from store
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').normalize() # Keep normalizing for now for simplicity
        else:
             raise ValueError("Data store is missing 'Date' column.")

        target_index = None
        # Try matching the index directly first
        if original_index_from_click in df.index:
            target_index = original_index_from_click
        else:
            # Fallback: Try converting index types if they mismatch
            logger.warning(f"Direct index match failed for {original_index_from_click} (type {type(original_index_from_click)}) in index (type {df.index.dtype}). Attempting type conversion/date matching.")
            try:
                if df.index.dtype == 'int64' and isinstance(original_index_from_click, str) and original_index_from_click.isdigit():
                    converted_index = int(original_index_from_click)
                    if converted_index in df.index: target_index = converted_index
                elif df.index.dtype == 'object' and isinstance(original_index_from_click, int):
                    converted_index = str(original_index_from_click)
                    if converted_index in df.index: target_index = converted_index
            except Exception as e:
                logger.warning(f"Could not convert clicked index type: {e}")

            # Fallback 2: Match by Date/Time if index matching failed
            if target_index is None and date_clicked:
                 try:
                      # NOTE: Need careful parsing here if `date_clicked` has time
                      # Normalize clicked date/time to match how it's stored/compared in df
                      clicked_dt_normalized = pd.to_datetime(date_clicked).normalize() # Normalize for now
                      matches = df[df['Date'] == clicked_dt_normalized]
                      if len(matches) == 1:
                          target_index = matches.index[0]
                          logger.warning(f"Index mismatch resolved by matching normalized date {clicked_dt_normalized.date()} to index {target_index}.")
                      elif len(matches) > 1:
                          # If multiple points have the same normalized date, we can't be sure which one was clicked
                          # unless we add more customdata (like the value) to the plot click info.
                          logger.error(f"Index mismatch: Multiple rows found for normalized date {clicked_dt_normalized.date()}. Cannot proceed unambiguously.")
                          raise ValueError(f"Ambiguous date {clicked_dt_normalized.date()} found.")
                      else:
                          logger.error(f"Index mismatch: No row found for normalized date {clicked_dt_normalized.date()} after index lookup failed.")
                 except Exception as date_match_e:
                      logger.error(f"Error during date fallback matching: {date_match_e}")

        # Final check if we found a target index
        if target_index is None:
            logger.error(f"Could not definitively match clicked point (Original Index: {original_index_from_click}, Date: {date_clicked}) to DataFrame index (Type: {df.index.dtype}). Aborting action.")
            raise IndexError(f"Data mismatch: Point index {original_index_from_click} not found in DataFrame.")

        logger.info(f"Successfully matched clicked point to DataFrame index: {target_index}")

        # Ensure essential columns exist before modification
        if 'Qualifiers' not in df.columns: df['Qualifiers'] = pd.Series(dtype='object')
        if 'ReviewStatus' not in df.columns: df['ReviewStatus'] = 'Unknown'
        if 'DISCHARGE' not in df.columns: df['DISCHARGE'] = np.nan # Should exist, but safety check

        # --- Apply Action based on button clicked ---
        if triggered_button_id == 'qc-approve-button':
            # Clear all flags for this point
            flags_to_clear = [col for col in ['FLAG_LESS_THAN_Min._Value','FLAG_ZERO','FLAG_REPEATED','FLAG_GREATER_THAN_MaxValue','UNUSUAL_SPIKE','FLAG_BELOW_CAPACITY', 'FLAGGED'] if col in df.columns]
            if flags_to_clear:
                 df.loc[target_index, flags_to_clear] = False
            # Update status and qualifier
            df.loc[target_index, 'ReviewStatus'] = 'Approved'
            current_qual = df.loc[target_index, 'Qualifiers']
            if pd.isna(current_qual) or str(current_qual).strip() == '': new_qual = 'Approved'
            elif 'Approved' not in str(current_qual).split(';'): new_qual = f"{str(current_qual).strip()};Approved"
            else: new_qual = current_qual # Already contains 'Approved'
            df.loc[target_index, 'Qualifiers'] = new_qual
            action_performed = "Approved"

        elif triggered_button_id == 'qc-interpolate-button':
            # NOTE: Interpolation logic assumes sorted data and uses row position,
            # may need refinement if dealing with irregular time series.
            df = df.sort_values(by='Date') # Ensure sorted by date
            try:
                # Find the integer position (iloc) of the target index after sorting
                loc_index = df.index.get_loc(target_index)
            except KeyError:
                 logger.error(f"Target index {target_index} not found after potential sort. Aborting interpolation.")
                 raise ValueError("Interpolation target index lost.")

            prev_val, next_val = np.nan, np.nan
            prev_idx_pos, next_idx_pos = -1, -1
            # Find previous non-NaN value's integer position (iloc)
            for i in range(loc_index - 1, -1, -1):
                idx_prev = df.index[i] # Get the actual index label at position i
                if pd.notna(df.loc[idx_prev, 'DISCHARGE']):
                    prev_val = df.loc[idx_prev, 'DISCHARGE']
                    prev_idx_pos = i # Store the integer position
                    break
            # Find next non-NaN value's integer position (iloc)
            for i in range(loc_index + 1, len(df)):
                idx_next = df.index[i] # Get the actual index label at position i
                if pd.notna(df.loc[idx_next, 'DISCHARGE']):
                    next_val = df.loc[idx_next, 'DISCHARGE']
                    next_idx_pos = i # Store the integer position
                    break

            # Check if valid neighbors were found
            if pd.notna(prev_val) and pd.notna(next_val) and prev_idx_pos != -1 and next_idx_pos != -1:
                # Simple linear interpolation based on integer position in sorted frame
                interpolated_val = np.interp(loc_index, [prev_idx_pos, next_idx_pos], [prev_val, next_val])
                df.loc[target_index, 'DISCHARGE'] = interpolated_val
                df.loc[target_index, 'ReviewStatus'] = 'Interpolated'
                current_qual = df.loc[target_index, 'Qualifiers']
                if pd.isna(current_qual) or str(current_qual).strip() == '': new_qual = 'Interpolated'
                elif 'Interpolated' not in str(current_qual).split(';'): new_qual = f"{str(current_qual).strip()};Interpolated"
                else: new_qual = current_qual
                df.loc[target_index, 'Qualifiers'] = new_qual
                logger.info(f"Interpolated value at index {target_index} (position {loc_index}) to {interpolated_val:.2f}")
                action_performed = "Interpolated"
            else:
                # Failed to interpolate
                logger.warning(f"Cannot interpolate index {target_index}: Missing valid neighbors (prev={prev_val} at pos {prev_idx_pos}, next={next_val} at pos {next_idx_pos}).")
                notification = dbc.Alert("Cannot interpolate: Missing valid neighbors.", color="warning", duration=4000, dismissable=True)
                action_performed = None # Ensure action is not marked as performed

        elif triggered_button_id == 'qc-delete-button':
            # Set discharge to NaN
            df.loc[target_index, 'DISCHARGE'] = np.nan
            df.loc[target_index, 'ReviewStatus'] = 'Deleted'
            current_qual = df.loc[target_index, 'Qualifiers']
            if pd.isna(current_qual) or str(current_qual).strip() == '': new_qual = 'Deleted'
            elif 'Deleted' not in str(current_qual).split(';'): new_qual = f"{str(current_qual).strip()};Deleted"
            else: new_qual = current_qual
            df.loc[target_index, 'Qualifiers'] = new_qual
            action_performed = "Deleted (set to NaN)"

        # --- Post-Action Processing (if action was successful) ---
        if action_performed and df is not None:
            logger.info(f"Action '{action_performed}' applied for index {target_index}. Re-flagging...")
            units = site_info.get('units', '?')

            # Re-apply flagging logic
            if thresholds:
                try:
                    # Ensure Date column is datetime before flagging
                    if 'Date' in df and not pd.api.types.is_datetime64_any_dtype(df['Date']):
                         df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = apply_flagging(df, thresholds, logger)
                    logger.info("Re-flagging complete after QC action.")
                except Exception as flag_e:
                     logger.error(f"Error during re-flagging after QC action: {flag_e}", exc_info=True)
                     # Append to notification if one already exists from interpolation failure etc.
                     current_alert_children = notification.children if notification else ""
                     notification = dbc.Alert(f"{current_alert_children} Action applied, but error during re-flagging: {flag_e}. Save changes carefully.", color="danger", dismissable=True)
            else:
                logger.warning("No thresholds available for re-flagging after QC action!")
                current_alert_children = notification.children if notification else ""
                notification = dbc.Alert(f"{current_alert_children} Action applied, but cannot re-flag (thresholds missing). Save changes carefully.", color="warning", dismissable=True)

            # Prepare updated data for storage (convert datetime back to string)
            df_store_update = df.copy()
            if 'Date' in df_store_update and pd.api.types.is_datetime64_any_dtype(df_store_update['Date']):
                df_store_update['Date'] = df_store_update['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S') # ISO format
            updated_data_store = df_store_update.to_json(orient='split', date_format='iso')

            # Prepare data for table display (ensure datetime objects for date column)
            df_table_update = df.copy()
            if 'Date' in df_table_update and not pd.api.types.is_datetime64_any_dtype(df_table_update['Date']):
                 df_table_update['Date'] = pd.to_datetime(df_table_update['Date'], errors='coerce')
            updated_table = create_dash_data_table(df_table_update, units, 'editable-data-table', logger)

            # Check if table creation failed
            if not isinstance(updated_table, dash_table.DataTable):
                logger.error("Failed to recreate table after QC action.")
                updated_table = html.Div("Error updating table.") # Provide placeholder
                current_alert_children = notification.children if notification else ""
                notification = dbc.Alert(f"{current_alert_children} Action applied, but failed to update table.", color="danger")

            # Re-generating plot after every QC click can be slow.
            # Returning no_update means the user has to click "Update Plot" manually.
            # Consider adding a "Refresh Plot" button to the modal or updating the figure directly for a better UX.
            updated_plot = no_update # Signal that plot needs manual refresh

            # Provide success feedback if no prior error notification exists
            if not notification:
                notification = dbc.Alert(f"Action '{action_performed}' applied successfully. Click 'Update Plot' to refresh graph. Save changes to export.", color="success", duration=5000, dismissable=True)
            logger.info(f"Store and table updated after '{action_performed}'. Plot refresh needed.")

        elif not action_performed and not notification:
             # Case where action failed (e.g., interpolation) but didn't set a specific notification
             notification = dbc.Alert(f"Action '{triggered_button_id}' could not be completed.", color="warning", dismissable=True)

        # Close the modal and return updates
        return modal_is_open, notification, updated_data_store, updated_table, updated_plot

    except Exception as e:
        logger.error(f"Error during QC action '{triggered_button_id}' for index {original_index_from_click}: {e}", exc_info=True)
        notification = dbc.Alert(f"Error processing action '{triggered_button_id}': {e}", color="danger", dismissable=True)
        # Keep modal closed on error, return no updates to data/table/plot
        return False, notification, no_update, no_update, no_update

@callback(
    Output('url', 'search'),
    Input('site-info-store', 'data'),
    prevent_initial_call=True
)
def update_url_on_data_change(site_info):
    """Updates the URL query string when site info (like date range) changes."""
    if not site_info or not isinstance(site_info, dict):
        logger.debug("URL update skipped: site-info-store empty/invalid.")
        return no_update
    # Get data needed for URL, check if they exist
    site_id = site_info.get('site_id'); start_date_str = site_info.get('start'); end_date_str = site_info.get('end')
    if not site_id or not start_date_str or not end_date_str:
        logger.warning(f"URL update skipped: Missing info in site-info: {site_info}")
        return no_update
    # Validate date formats before creating URL
    try:
        datetime.strptime(start_date_str, '%Y-%m-%d'); datetime.strptime(end_date_str, '%Y-%m-%d')
    except (ValueError, TypeError):
        logger.warning(f"URL update skipped: Invalid date format in site-info: start='{start_date_str}', end='{end_date_str}'")
        return no_update
    # Construct the query string
    query_string = f"?id={site_id}&start_date={start_date_str}&end_date={end_date_str}"
    logger.info(f"Updating URL search string to: {query_string}")
    return query_string


# <<< MODIFIED: open_enter_data_modal includes time population >>>
@callback(
    [Output('enter-data-modal', 'is_open', allow_duplicate=True),
     Output('notification-area', 'children', allow_duplicate=True),
     Output('enter-time-input', 'value')], # Added Output for time input value
    Input('open-enter-data-modal-button', 'n_clicks'),
    State('site-info-store', 'data'),
    prevent_initial_call=True
)
def open_enter_data_modal(n_clicks, site_info):
    if not n_clicks:
        # Don't change anything if the button wasn't clicked
        return no_update, no_update, no_update

    # Get current time formatted as HH:MM
    # Using current system time where the server is running
    current_time_str = datetime.now().strftime('%H:%M')

    if site_info and site_info.get('site_id'):
        logger.info(f"Opening enter data modal. Setting time to {current_time_str}.")
        # Open modal, no notification update, set the current time in the input field
        return True, no_update, current_time_str
    else:
        logger.warning("Cannot open Enter Data modal: No site data loaded.")
        # Don't open modal, show notification, don't change time input
        alert = dbc.Alert("Load site data first before entering new measurements.", color="warning", duration=4000, dismissable=True)
        return False, alert, no_update
# <<< END: Modified open_enter_data_modal >>>


# <<< MODIFIED: handle_submit_new_data uses time input state >>>
@callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('table-container', 'children', allow_duplicate=True),
     Output('enter-data-modal', 'is_open', allow_duplicate=True),
     Output('enter-data-modal-alert', 'children'),
     Output('enter-data-modal-alert', 'is_open'),
     Output('notification-area', 'children', allow_duplicate=True)],
    Input('submit-enter-data-button', 'n_clicks'),
    [State('enter-date-picker', 'date'),
     State('enter-time-input', 'value'), # Added Time Input State
     State('enter-discharge-input', 'value'),
     State('enter-qualifier-input', 'value'),
     State('data-store', 'data'),
     State('site-info-store', 'data'),
     State('thresholds-store', 'data')],
    prevent_initial_call=True
)
def handle_submit_new_data(n_clicks, new_date_str,
                           new_time_str, # Added Time Input Argument
                           new_discharge, new_qualifier,
                           stored_json, site_info, thresholds):
    if not n_clicks: return no_update, no_update, no_update, no_update, False, no_update

    modal_errors = []
    if not new_date_str: modal_errors.append("Please select a date.")
    # Added basic Time validation
    if not new_time_str:
         modal_errors.append("Please enter a time (e.g., HH:MM).")
    else:
        # Optional: Add more robust time format validation here if needed
        try:
            datetime.strptime(new_time_str, '%H:%M')
        except ValueError:
            modal_errors.append("Invalid time format. Use HH:MM.")

    if new_discharge is None or str(new_discharge).strip() == '': modal_errors.append("Please enter a discharge value.")

    if stored_json is None or site_info is None:
        logger.error("Cannot add data: Missing session context.")
        main_notification = dbc.Alert("Error: Cannot add data, session context missing.", color="danger")
        error_msg = "Session data missing." + (" " + " ".join(modal_errors) if modal_errors else "")
        return no_update, no_update, True, error_msg, True, main_notification

    site_id = site_info.get('site_id'); units = site_info.get('units', '?')
    new_discharge_float = None

    if not modal_errors and new_discharge is not None:
        try:
            new_discharge_float = float(new_discharge)
        except (ValueError, TypeError):
            logger.error(f"Invalid discharge value: {new_discharge}")
            modal_errors.append(f"Invalid discharge value: '{new_discharge}'. Must be a number.")

    if modal_errors:
        error_message = html.Div([html.P("Please correct the following:")] + [html.Li(msg) for msg in modal_errors])
        return no_update, no_update, True, error_message, True, no_update # Keep modal open

    # Log the time value
    logger.info(f"Attempting to add new measurement for site {site_id}: Date={new_date_str}, Time={new_time_str}, Discharge={new_discharge_float}, Qualifier={new_qualifier}")
    try:
        try:
            # --- NOTE: Current logic normalizes date (removes time). ---
            # If you want to store the specific time, you need to combine
            # new_date_str and new_time_str into a datetime object here
            # and ensure the rest of the application handles datetimes correctly.
            # Example (requires careful testing and updates elsewhere):
            # combined_dt_str = f"{new_date_str} {new_time_str}"
            # new_datetime = pd.to_datetime(combined_dt_str, format='%Y-%m-%d %H:%M', errors='raise')
            # --- Using date-only logic for now: ---
            new_date = pd.to_datetime(new_date_str).normalize() # Processed date/time object
            # ---
        except (ValueError, TypeError):
             logger.error(f"Invalid date/time format from picker: Date='{new_date_str}', Time='{new_time_str}'")
             return no_update, no_update, True, f"Invalid date/time selected: D='{new_date_str}', T='{new_time_str}'", True, no_update

        df = pd.read_json(stored_json, orient='split')
        if 'Date' in df.columns:
            # --- NOTE: Comparison based on normalized date. Adjust if using full timestamps ---
            # Convert existing data to datetime (normalized) for comparison
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').normalize()
        else:
             raise ValueError("Data store missing 'Date' column.")

        # Check if the DATE already exists (adjust if checking full timestamp)
        if new_date in df['Date'].values:
            logger.warning(f"Date {new_date_str} already exists. Appending new entry (potential duplicate).")
            # Optionally, you could add a warning in the modal alert here

        # Ensure all necessary columns exist in the DataFrame before adding the new row
        base_cols = ['Date', 'DISCHARGE', 'ReviewStatus', 'Qualifiers', 'FLAGGED']
        flag_cols = ['FLAG_LESS_THAN_Min._Value','FLAG_ZERO','FLAG_REPEATED','FLAG_GREATER_THAN_MaxValue','UNUSUAL_SPIKE','FLAG_BELOW_CAPACITY']
        all_expected_cols = base_cols + flag_cols
        for col in all_expected_cols:
            if col not in df.columns:
                logger.debug(f"Adding missing column '{col}' before appending new data.")
                if col in ['Date', 'DISCHARGE']: continue # These will be in the new row
                # Add default values for missing columns
                if col == 'FLAGGED' or col.startswith('FLAG_'): df[col] = False
                elif col == 'ReviewStatus': df[col] = 'Unknown'
                elif col == 'Qualifiers': df[col] = pd.Series(dtype='object')
                else: df[col] = np.nan # Default for potentially numeric flags if they were missing

        # Create the new row dictionary
        new_row = {
            'Date': new_date, # Use the processed date/datetime variable here
            'DISCHARGE': new_discharge_float,
            'ReviewStatus': 'Entered',
            'Qualifiers': new_qualifier if new_qualifier else 'Manual Entry',
            'FLAGGED': False, # Initialize flags as False
            # Initialize specific flag columns that exist in the DataFrame
            **{flag_col: False for flag_col in flag_cols if flag_col in df.columns}
        }
        new_row_df = pd.DataFrame([new_row])

        # Append the new row and sort
        df = pd.concat([df, new_row_df], ignore_index=True)
        # --- NOTE: Sorting by 'Date' assumes it's comparable (date or datetime) ---
        df = df.sort_values(by='Date').reset_index(drop=True)
        logger.info(f"Appended new row for {new_date_str}. DataFrame size: {len(df)}")

        main_feedback = None
        # Re-apply flagging logic
        if thresholds:
            logger.info("Re-flagging dataset after adding new row...")
            try:
                # Ensure Date column is datetime before flagging
                if 'Date' in df and not pd.api.types.is_datetime64_any_dtype(df['Date']):
                     df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Convert to datetime if needed
                df = apply_flagging(df, thresholds, logger)
                logger.info("Re-flagging complete.")
                main_feedback = dbc.Alert(f"Measurement for {new_date_str} added and re-flagged.", color="success", duration=4000, dismissable=True)
            except Exception as flag_e:
                 logger.error(f"Error during re-flagging after add: {flag_e}", exc_info=True)
                 main_feedback = dbc.Alert(f"Measurement for {new_date_str} added, but error during re-flagging: {flag_e}", color="danger", duration=5000, dismissable=True)
        else:
            logger.warning("Thresholds missing, skipping re-flagging.")
            main_feedback = dbc.Alert(f"Measurement for {new_date_str} added, thresholds missing for re-flagging.", color="warning", duration=5000, dismissable=True)

        # Prepare updated data for storage (convert datetime back to string)
        df_store_updated = df.copy()
        if 'Date' in df_store_updated and pd.api.types.is_datetime64_any_dtype(df_store_updated['Date']):
            df_store_updated['Date'] = df_store_updated['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S') # ISO format
        updated_json_store = df_store_updated.to_json(orient='split', date_format='iso')

        # Prepare data for table display (ensure datetime objects for date column)
        df_table_update = df.copy()
        if 'Date' in df_table_update and not pd.api.types.is_datetime64_any_dtype(df_table_update['Date']):
            df_table_update['Date'] = pd.to_datetime(df_table_update['Date'], errors='coerce') # Ensure datetime for table helper
        new_table_component = create_dash_data_table(df_table_update, units, 'editable-data-table', logger)

        # Check if table creation failed
        if not isinstance(new_table_component, dash_table.DataTable):
            logger.error("Failed to create table after adding data.")
            new_table_component = html.Div("Error updating table.") # Placeholder
            main_feedback = dbc.Alert("Measurement added, failed to update table.", color="danger")
            # Still update the store even if table fails, but close modal and show error
            return updated_json_store, new_table_component, False, "Error updating table.", True, main_feedback

        logger.info(f"Successfully processed data for {new_date_str}.")
        # Close modal, clear modal alert, show success feedback
        return updated_json_store, new_table_component, False, None, False, main_feedback

    except Exception as e:
        logger.error(f"Error adding new data for site {site_id}: {e}", exc_info=True)
        # Include time in error log context
        logger.error(f"(Input values: Date='{new_date_str}', Time='{new_time_str}', Discharge='{new_discharge}', Qualifier='{new_qualifier}')", exc_info=False)
        # Keep modal open, show error in modal alert
        return no_update, no_update, True, f"An unexpected error occurred: {e}", True, no_update
# <<< END: Modified handle_submit_new_data >>>

@callback(
    [Output('enter-data-modal', 'is_open', allow_duplicate=True),
     Output('enter-data-modal-alert', 'children', allow_duplicate=True),
     Output('enter-data-modal-alert', 'is_open', allow_duplicate=True)],
    [Input('cancel-enter-data-button', 'n_clicks')],
    prevent_initial_call=True
)
def cancel_enter_data_modal(cancel_clicks):
    if cancel_clicks:
        logger.debug("Closing enter data modal via Cancel.")
        # Close modal, clear any alerts
        return False, None, False
    return no_update, no_update, no_update


# --- Callbacks for Add Multiple Data Modal ---

@callback(
    [Output('add-multiple-data-modal', 'is_open'),
     Output('notification-area', 'children', allow_duplicate=True),
     Output('add-data-input-table', 'data')],
    Input('open-add-multiple-modal-button', 'n_clicks'),
    [State('site-info-store', 'data')],
    prevent_initial_call=True
)
def open_add_multiple_data_modal(n_clicks, site_info):
    if not n_clicks:
        return no_update, no_update, no_update
    # Reset the input table to blank rows when opening
    initial_data = get_initial_input_table_data()
    if site_info and site_info.get('site_id'):
        logger.info("Opening add multiple data modal and resetting input table.")
        return True, no_update, initial_data
    else:
        logger.warning("Cannot open Add Multiple Data modal: No site data loaded.")
        # Don't open modal, show notification, provide initial (empty) table data structure
        alert = dbc.Alert("Load site data first before adding multiple measurements.", color="warning", duration=4000, dismissable=True)
        return False, alert, initial_data


@callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('table-container', 'children', allow_duplicate=True),
     Output('add-multiple-data-modal', 'is_open', allow_duplicate=True),
     Output('add-multiple-data-modal-alert', 'children'),
     Output('add-multiple-data-modal-alert', 'is_open'),
     Output('notification-area', 'children', allow_duplicate=True)],
    Input('submit-multiple-data-button', 'n_clicks'),
    [State('add-data-input-table', 'data'),
     State('data-store', 'data'),
     State('site-info-store', 'data'),
     State('thresholds-store', 'data')],
    prevent_initial_call=True
)
def handle_submit_multiple_data(n_clicks, input_table_data, stored_json, site_info, thresholds):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, False, no_update

    if stored_json is None or site_info is None:
        logger.error("Cannot add multiple data points: Missing current data-store or site-info-store.")
        main_notification = dbc.Alert("Error: Cannot add data, session context is missing. Please reload.", color="danger")
        # Keep modal open and show error inside modal
        return no_update, no_update, True, "Session data missing. Cannot proceed.", True, main_notification

    site_id = site_info.get('site_id'); units = site_info.get('units', '?')
    logger.info(f"Attempting to add measurements for site {site_id} from input table.")

    modal_errors = []
    valid_new_data_rows = []
    submitted_dates_for_dup_check = []
    has_actual_input = False # Flag to check if any non-empty rows were submitted

    if not input_table_data:
        modal_errors.append("No data entered or pasted into the table.")
    else:
        for i, row in enumerate(input_table_data):
            row_num = i + 1
            date_str = row.get('Date')
            discharge_val = row.get('Discharge')
            qualifier_val = row.get('Qualifier')

            # Skip completely empty rows silently
            if (date_str is None or str(date_str).strip() == "") and \
               (discharge_val is None or str(discharge_val).strip() == "") and \
               (qualifier_val is None or str(qualifier_val).strip() == ""):
                continue

            has_actual_input = True # Mark that we found at least one row with some data
            row_has_error = False
            validated_date = None
            discharge_float = None

            # Validate Date
            if date_str is None or str(date_str).strip() == "":
                # Date is required if discharge is present
                if discharge_val is not None and str(discharge_val).strip() != "":
                    modal_errors.append(f"Row {row_num}: Date is missing but Discharge is present.")
                    row_has_error = True
                # If discharge is also missing, we already skipped this row
            else:
                try:
                    # Try specific format first
                    validated_date = datetime.strptime(str(date_str).strip(), '%Y-%m-%d').date()
                    submitted_dates_for_dup_check.append(validated_date)
                except (ValueError, TypeError):
                     try:
                         # Fallback to general pandas parsing
                         parsed_dt = pd.to_datetime(str(date_str).strip(), errors='coerce')
                         if pd.isna(parsed_dt):
                             modal_errors.append(f"Row {row_num}: Invalid date format ('{date_str}'). Use YYYY-MM-DD.")
                             row_has_error = True
                         else:
                             validated_date = parsed_dt.date()
                             submitted_dates_for_dup_check.append(validated_date)
                             logger.warning(f"Parsed potentially ambiguous date format '{date_str}' for row {row_num} as {validated_date}. Please use YYYY-MM-DD.")
                     except Exception as e_parse:
                         modal_errors.append(f"Row {row_num}: Error parsing date ('{date_str}'): {e_parse}. Use YYYY-MM-DD.")
                         row_has_error = True

            # Validate Discharge
            if discharge_val is None or str(discharge_val).strip() == "":
                 # Discharge is required if date is present
                 if date_str is not None and str(date_str).strip() != "":
                     modal_errors.append(f"Row {row_num}: Discharge value is missing but Date is present.")
                     row_has_error = True
                 # If date is also missing, we already skipped this row
            else:
                try:
                    discharge_float = float(discharge_val)
                except (ValueError, TypeError):
                    modal_errors.append(f"Row {row_num}: Invalid discharge value '{discharge_val}'. Must be a number.")
                    row_has_error = True

            # If no errors for this row, add it to the list of valid rows
            if not row_has_error and validated_date is not None and discharge_float is not None:
                valid_new_data_rows.append({
                    'date': validated_date,
                    'discharge': discharge_float,
                    'qualifier': qualifier_val if qualifier_val else 'Manual Paste Entry'
                })
            elif not row_has_error and (validated_date or discharge_float):
                 # This case should ideally not be hit if validation is correct
                 logger.error(f"Row {row_num} had no errors but data was not fully captured. Date: {validated_date}, Discharge: {discharge_float}")
                 modal_errors.append(f"Row {row_num}: Internal error processing potentially valid data.")

    # After checking all rows:
    # Check for duplicate dates within the submitted valid entries
    if not modal_errors and valid_new_data_rows:
        date_counts = Counter(submitted_dates_for_dup_check)
        duplicates = sorted([d.strftime('%Y-%m-%d') for d, count in date_counts.items() if count > 1])
        if duplicates:
            modal_errors.append(f"Duplicate dates found within the submitted entries: {', '.join(duplicates)}. Please ensure all dates are unique before submitting.")

    # Check if input was provided but none of it was valid
    if not modal_errors and not valid_new_data_rows and has_actual_input:
         modal_errors.append("No valid data rows found. Ensure each row has both a Date (YYYY-MM-DD) and a numeric Discharge value.")

    # If any errors were found, display them in the modal and keep it open
    if modal_errors:
        error_message = html.Div([html.P("Please correct the following errors:")] + [html.Li(msg) for msg in modal_errors], style={'maxHeight': '40vh', 'overflowY': 'auto', 'color': 'red'})
        return no_update, no_update, True, error_message, True, no_update

    # --- If validation passed ---
    try:
        df = pd.read_json(stored_json, orient='split')
        if 'Date' in df.columns:
             # --- NOTE: Normalize existing dates for comparison. Adjust if using full timestamps ---
             df['Date'] = pd.to_datetime(df['Date'], errors='coerce').normalize()
        else:
             raise ValueError("Data store missing 'Date' column.")

        # Check for duplicates against existing data (optional warning)
        existing_dates = set(df['Date'].dt.date)
        submitted_dates_obj = [row['date'] for row in valid_new_data_rows]
        duplicates_with_existing = sorted([d.strftime('%Y-%m-%d') for d in submitted_dates_obj if d in existing_dates])
        if duplicates_with_existing:
            logger.warning(f"Dates {duplicates_with_existing} already exist in the dataset. Appending new entries (potential duplicates).")
            # Optionally add a non-blocking warning in the main notification area upon success

        # Ensure all necessary columns exist before adding new data
        base_cols = ['Date', 'DISCHARGE', 'ReviewStatus', 'Qualifiers', 'FLAGGED']
        flag_cols = ['FLAG_LESS_THAN_Min._Value','FLAG_ZERO','FLAG_REPEATED','FLAG_GREATER_THAN_MaxValue','UNUSUAL_SPIKE','FLAG_BELOW_CAPACITY']
        all_expected_cols = base_cols + flag_cols
        for col in all_expected_cols:
            if col not in df.columns:
                logger.debug(f"Adding missing column '{col}' to DataFrame before adding multiple rows.")
                if col in ['Date', 'DISCHARGE']: continue
                if col == 'FLAGGED' or col.startswith('FLAG_'): df[col] = False
                elif col == 'ReviewStatus': df[col] = 'Unknown'
                elif col == 'Qualifiers': df[col] = pd.Series(dtype='object')
                else: df[col] = np.nan

        # Prepare list of dictionaries for the new rows
        new_rows_prepared = []
        for row_data in valid_new_data_rows:
            new_rows_prepared.append({
                # --- NOTE: Store as Timestamp (normalized date). Adjust if using full timestamps ---
                'Date': pd.Timestamp(row_data['date']),
                'DISCHARGE': row_data['discharge'],
                'ReviewStatus': 'Entered',
                'Qualifiers': row_data['qualifier'],
                'FLAGGED': False,
                **{flag_col: False for flag_col in flag_cols if flag_col in df.columns} # Initialize existing flags
            })

        # Create DataFrame from new rows and concatenate
        new_rows_df = pd.DataFrame(new_rows_prepared)
        df = pd.concat([df, new_rows_df], ignore_index=True)
        # --- NOTE: Sorting by 'Date'. Adjust if using full timestamps ---
        df = df.sort_values(by='Date').reset_index(drop=True)
        logger.info(f"Appended {len(new_rows_df)} new rows from input table. DataFrame size now: {len(df)}")

        main_feedback = None
        # Re-apply flagging logic
        if thresholds:
            logger.info("Re-flagging entire dataset after adding multiple rows...")
            try:
                 # --- NOTE: Ensure correct Date type before flagging ---
                 if 'Date' in df and not pd.api.types.is_datetime64_any_dtype(df['Date']):
                     df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Ensure datetime
                 df = apply_flagging(df, thresholds, logger)
                 logger.info("Re-flagging complete.")
                 # Add warning about existing duplicates if any were found
                 dup_warning = f" Warning: Dates already existed for {', '.join(duplicates_with_existing)}." if duplicates_with_existing else ""
                 main_feedback = dbc.Alert(f"{len(new_rows_df)} measurements added and data re-flagged.{dup_warning} Save changes to export.", color="success", duration=6000, dismissable=True)
            except Exception as flag_e:
                 logger.error(f"Error during re-flagging after multi-add: {flag_e}", exc_info=True)
                 main_feedback = dbc.Alert(f"{len(new_rows_df)} measurements added, but error during re-flagging: {flag_e}. Save changes carefully.", color="danger", duration=5000, dismissable=True)
        else:
            logger.warning("Thresholds not available, skipping re-flagging after adding data.")
            main_feedback = dbc.Alert(f"{len(new_rows_df)} measurements added, but thresholds missing for re-flagging. Save changes to export.", color="warning", duration=5000, dismissable=True)

        # Prepare updated data for storage (convert datetime back to string)
        df_store_updated = df.copy()
        if 'Date' in df_store_updated and pd.api.types.is_datetime64_any_dtype(df_store_updated['Date']):
            df_store_updated['Date'] = df_store_updated['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S') # ISO format
        updated_json_store = df_store_updated.to_json(orient='split', date_format='iso')

        # Prepare data for table display (ensure datetime objects for date column)
        df_table_update = df.copy()
        if 'Date' in df_table_update and not pd.api.types.is_datetime64_any_dtype(df_table_update['Date']):
             df_table_update['Date'] = pd.to_datetime(df_table_update['Date'], errors='coerce') # Ensure datetime
        new_table_component = create_dash_data_table(df_table_update, units, 'editable-data-table', logger)

        # Check if table creation failed
        if not isinstance(new_table_component, dash_table.DataTable):
            logger.error("Failed to create table component after adding multiple data points.")
            new_table_component = html.Div("Error updating table display.") # Placeholder
            main_feedback = dbc.Alert("Measurements added, but failed to update table display.", color="danger")
            # Close modal even if table fails, show error in main area
            return updated_json_store, new_table_component, False, None, False, main_feedback

        logger.info(f"Successfully added and processed {len(new_rows_df)} data points from table.")
        # Close modal, clear modal alert, show success feedback in main area
        return updated_json_store, new_table_component, False, None, False, main_feedback

    except Exception as e:
        logger.error(f"Error adding multiple new data points for site {site_id}: {e}", exc_info=True)
        # Keep modal open, show error in modal alert
        return no_update, no_update, True, f"An unexpected error occurred: {e}", True, no_update


@callback(
    [Output('add-multiple-data-modal', 'is_open', allow_duplicate=True),
     Output('add-multiple-data-modal-alert', 'children', allow_duplicate=True),
     Output('add-multiple-data-modal-alert', 'is_open', allow_duplicate=True)],
    Input('cancel-multiple-data-button', 'n_clicks'),
    prevent_initial_call=True
)
def cancel_add_multiple_data_modal(cancel_clicks):
    if cancel_clicks:
        logger.debug("Closing add multiple data modal via Cancel button.")
        # Close modal, clear any alerts inside it
        return False, None, False
    return no_update, no_update, no_update


# --- Main Execution Block ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    # Default to False for production, True for local dev if env var not set
    debug_env = os.environ.get("DASH_DEBUG", "False").lower()
    debug_mode = debug_env == "true"
    # Determine host based on environment (e.g., for Google Cloud Run)
    # Cloud Run expects host '0.0.0.0' to be accessible externally
    host = '0.0.0.0' if 'K_SERVICE' in os.environ else '127.0.0.1'

    logger.info(f"Starting Dash server on http://{host}:{port}")
    logger.info(f" -> Debug mode: {'ON' if debug_mode else 'OFF'}")
    # Use app.run for local development with Gunicorn or similar in production
    app.run(host=host, port=port, debug=debug_mode)

# --- END main.py ---