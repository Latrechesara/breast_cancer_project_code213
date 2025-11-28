import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from sklearn.datasets import load_breast_cancer

dash.register_page(__name__, path="/data-analysis", name="Data Analysis")

# --------------------------------------------
# Load dataset
# --------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["target_label"] = df["target"].map({0: "Malignant", 1: "Benign"})

# Important features
important_features = [
    "texture error", "area error", "smoothness error",
    "concavity error", "symmetry error",
    "fractal dimension error", "worst concavity"
]
top4_features = important_features[:4]  # For pairplots

# --------------------------------------------
# Layout
# --------------------------------------------
layout = html.Div(
    style={"padding": "30px"},
    children=[
        html.H2("Data Analysis", className="page-title"),

        # Filters row
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="diagnosis-filter",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Benign", "value": 1},
                            {"label": "Malignant", "value": 0},
                        ],
                        value="all",
                        clearable=False,
                        placeholder="Filter by Diagnosis",
                        style={
                            "background": "linear-gradient(135deg, #ff77b4, #e61227)",
                            "color": "#fff",
                            "border-radius": "8px",
                            "padding": "5px"
                        }
                    ),
                    width=3
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="feature-select",
                        options=[{"label": f, "value": f} for f in important_features],
                        value=important_features[0],
                        clearable=False,
                        placeholder="Select Feature",
                        style={
                            "background": "linear-gradient(135deg, #ff77b4, #e61227)",
                            "color": "#fff",
                            "border-radius": "8px",
                            "padding": "5px"
                        }
                    ),
                    width=3
                )
            ],
            className="mb-4",
            align="center",
            justify="start",
            style={"gap": "20px"}
        ),

        # Heatmap Card
        dbc.Card(
            dbc.CardBody(dcc.Graph(id="heatmap-plot")),
            className="custom-card shadow mb-4"
        ),

        # Pairplot Cards for top 4 features (2x2 grid)
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id=f"pairplot-{i}"), md=6)
                for i in range(6)  # 6 combinations for 4 features
            ],
            className="mb-4",
        ),

        # Box Plot Card
        dbc.Card(
            dbc.CardBody(dcc.Graph(id="box-plot")),
            className="custom-card shadow mb-4"
        ),

        # Violin Plot Card
        dbc.Card(
            dbc.CardBody(dcc.Graph(id="violin-plot")),
            className="custom-card shadow mb-4"
        ),

        # Histogram Card
        dbc.Card(
            dbc.CardBody(dcc.Graph(id="histogram-plot")),
            className="custom-card shadow mb-4"
        ),
    ]
)

# --------------------------------------------
# Callbacks
# --------------------------------------------
from dash import callback

@callback(
    Output("heatmap-plot", "figure"),
    [Output(f"pairplot-{i}", "figure") for i in range(6)],
    Output("box-plot", "figure"),
    Output("violin-plot", "figure"),
    Output("histogram-plot", "figure"),
    Input("diagnosis-filter", "value"),
    Input("feature-select", "value")
)
def update_plots(filter_value, selected_feature):
    # Filter data
    filtered = df.copy()
    if filter_value != "all":
        filtered = filtered[filtered["target"] == filter_value]

    # Heatmap
    corr = filtered[important_features].corr()
    fig_heatmap = px.imshow(corr, color_continuous_scale="RdPu")

    # Pairplots (all pairs of top 4 features)
    from itertools import combinations
    pair_figs = []
    for f1, f2 in combinations(top4_features, 2):
        fig = px.scatter(
            filtered,
            x=f1,
            y=f2,
            color="target_label",
            color_discrete_map={"Benign": "#ff77b4", "Malignant": "#9c005d"},
        )
        fig.update_layout(
            xaxis_tickangle=45,
            yaxis_tickangle=0,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        pair_figs.append(fig)

    # Box plot
    fig_box = px.box(
        filtered,
        y=selected_feature,
        color="target_label",
        color_discrete_map={"Benign": "#ff77b4", "Malignant": "#9c005d"},
        points="all"
    )

    # Violin plot
    fig_violin = px.violin(
        filtered,
        y=selected_feature,
        color="target_label",
        color_discrete_map={"Benign": "#ff77b4", "Malignant": "#9c005d"},
        box=True,
        points="all"
    )

    # Histogram plot
    fig_hist = px.histogram(
        filtered,
        x=selected_feature,
        color="target_label",
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"Benign": "#ff77b4", "Malignant": "#9c005d"}
    )

    return fig_heatmap, *pair_figs, fig_box, fig_violin, fig_hist

