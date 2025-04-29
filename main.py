import dash
from dash import Dash
import dash_html_components as html

# Setup app
app = Dash(__name__)
server = app.server

# Layout
app.layout = html.Div([
    html.H1("Hello World from Dash!")
])


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080)

