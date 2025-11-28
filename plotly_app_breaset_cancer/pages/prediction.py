import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/prediction", name="Prediction")

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "logreg_breastcancer_reduced.pkl")
TRAIN_PATH = os.path.join(BASE_DIR, "saved_models", "X_train_y_train.csv")

# ----------------------------
# Load model and training data
# ----------------------------
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

if os.path.exists(TRAIN_PATH):
    train_data = pd.read_csv(TRAIN_PATH)
else:
    raise FileNotFoundError(f"Training data CSV not found at {TRAIN_PATH}")

# ----------------------------
# Features used in model
# ----------------------------
features = [
    'texture error', 'area error', 'smoothness error', 'concavity error',
    'symmetry error', 'fractal dimension error', 'worst concavity'
]

feature_labels = [
    "Texture Error", "Area Error", "Smoothness Error", "Concavity Error",
    "Symmetry Error", "Fractal Dimension Error", "Worst Concavity"
]

X_train = train_data[features]
y_train = train_data['target']

# ----------------------------
# Precompute performance plots from training data
# ----------------------------
y_proba_train = model.predict_proba(X_train)[:, 1]
y_pred_train = model.predict(X_train)

# ROC Curve
fpr, tpr, _ = roc_curve(y_train, y_proba_train)
auc_score = roc_auc_score(y_train, y_proba_train)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=fpr, y=tpr, mode='lines',
    line=dict(color="#e61227", width=3),
    name=f"AUC={auc_score:.3f}"
))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color="gray", dash="dash")))
fig_roc.update_layout(title="ROC Curve (Training Data)", margin=dict(l=20, r=20, t=20, b=20))

# Feature importance
clf = model.named_steps[list(model.named_steps.keys())[-1]]
if hasattr(clf, "coef_"):
    coefs = clf.coef_.ravel()
    fig_feat = px.bar(
        x=features, y=coefs, color=coefs,
        color_continuous_scale="RdPu",
        labels={"x": "Feature", "y": "Coefficient"},
        title="Feature Importance (Training Data)"
    )
    fig_feat.update_layout(margin=dict(l=20, r=20, t=20, b=20))
else:
    fig_feat = go.Figure()

# Confusion Matrix
cm = confusion_matrix(y_train, y_pred_train)
fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=["Pred Benign", "Pred Malignant"],
    y=["Actual Benign", "Actual Malignant"],
    colorscale="RdPu",
    showscale=False
))
fig_cm.update_layout(title="Confusion Matrix (Training Data)", margin=dict(l=20, r=20, t=20, b=20))

# ----------------------------
# Layout
# ----------------------------
layout = html.Div([
    html.H2("Breast Cancer Prediction", style={"margin-bottom": "30px"}),

    # Prediction Card
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label(label),
                    dbc.Input(id=f"input-{feat}", type="number", step=0.01, value=0)
                ], width=3)
                for feat, label in zip(features, feature_labels)
            ], className="mb-3", style={"gap": "15px"}),
            dbc.Button("Predict", id="predict-btn", color="primary", className="mb-3"),
            html.Div(id="prediction-output")
        ]),
        style={
            "background": "linear-gradient(135deg, #ff77b4, #e61227)",
            "color": "#fff",
            "box-shadow": "0 4px 15px rgba(0,0,0,0.2)",
            "padding": "20px",
            "border-radius": "10px",
            "margin-bottom": "40px"
        }
    ),

    # Performance Card (precomputed from training data)
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_roc), md=4),
                dbc.Col(dcc.Graph(figure=fig_feat), md=4),
                dbc.Col(dcc.Graph(figure=fig_cm), md=4),
            ])
        ]),
        style={
            "background": "#fff",
            "box-shadow": "0 4px 15px rgba(0,0,0,0.1)",
            "padding": "20px",
            "border-radius": "10px"
        }
    )
], style={"margin-left": "3%", "margin-right": "3%", "margin-top": "20px"})

# ----------------------------
# Callback for user prediction
# ----------------------------
@dash.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(f"input-{feat}", "value") for feat in features]
)
def predict_user(n_clicks, *vals):
    if n_clicks is None:
        return ""
    x_input = pd.DataFrame(np.array(vals).reshape(1, -1), columns=features)
    y_pred = model.predict(x_input)[0]
    y_proba = model.predict_proba(x_input)[0, 1]
    return html.Div([
        html.H4(f"Prediction: {'Benign' if y_pred==1 else 'Malignant'}"),
        html.P(f"Probability of being benign: {y_proba:.3f}")
    ])
