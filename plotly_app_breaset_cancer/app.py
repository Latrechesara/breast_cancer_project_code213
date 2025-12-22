import dash
from dash import html, page_container
import dash_bootstrap_components as dbc

# ----------------------------#
# Initialize app
# ----------------------------#
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Breast Cancer Dashboard"
)
server = app.server
# ----------------------------#
# Header
# ----------------------------#
header = html.Header([
    html.Div("Breast Cancer Project Dashboard", className="header"),
    html.Nav([
        dash.dcc.Link("Introduction", href="/", className="nav-link"),
        dash.dcc.Link("Data Analysis", href="/data-analysis", className="nav-link"),
        dash.dcc.Link("Prediction", href="/prediction", className="nav-link")
    ], className="navbar")
])

# ----------------------------#
# Footer
# ----------------------------#
footer = html.Footer([
    html.Div("© 2025 Latreche Sara|Breast Cancer Analysis", className="footer")
])

# ----------------------------#
# Layout
# ----------------------------#
app.layout = html.Div([
    header,
    html.Div(page_container, className="main-container"),
    footer
])

# ----------------------------#
# Run
# ----------------------------#
if __name__ == "__main__":
    app.run(debug=True)
