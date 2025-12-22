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
# Precompute performance plots
# ----------------------------
y_proba_train = model.predict_proba(X_train)[:, 1]
y_pred_train = model.predict(X_train)

# ROC Curve
fpr, tpr, _ = roc_curve(y_train, y_proba_train)
auc_score = roc_auc_score(y_train, y_proba_train)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color="#e61227", width=3), name=f"AUC={auc_score:.3f}"))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color="gray", dash="dash")))
fig_roc.update_layout(title="ROC Curve", margin=dict(l=20, r=20, t=40, b=20))

# Feature importance
clf = model.named_steps[list(model.named_steps.keys())[-1]]
if hasattr(clf, "coef_"):
    coefs = clf.coef_.ravel()
    fig_feat = px.bar(x=features, y=coefs, color=coefs, color_continuous_scale="RdPu", title="Feature Importance")
    fig_feat.update_layout(margin=dict(l=20, r=20, t=40, b=20))
else:
    fig_feat = go.Figure()

# Confusion Matrix
cm = confusion_matrix(y_train, y_pred_train)
fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Pred Malignant (0)", "Pred Benign (1)"], y=["Actual Malignant (0)", "Actual Benign (1)"], colorscale="RdPu", showscale=False))
fig_cm.update_layout(title="Confusion Matrix", margin=dict(l=20, r=20, t=40, b=20))

# ----------------------------
# Layout
# ----------------------------
layout = html.Div([
    html.H2("Breast Cancer Prediction", style={"margin-bottom": "30px"}),

    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label(label),
                    dbc.Input(id=f"input-{feat}", type="number", step=0.0001, value=0)
                ], width=3, className="mb-2")
                for feat, label in zip(features, feature_labels)
            ], className="mb-3"),
            dbc.Button("Run Diagnostic Prediction", id="predict-btn", color="light", className="mb-3"),
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

    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_roc), md=4),
                dbc.Col(dcc.Graph(figure=fig_feat), md=4),
                dbc.Col(dcc.Graph(figure=fig_cm), md=4),
            ])
        ])
    )
], style={"margin": "20px 3%"})

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

    # Sanitize inputs (The Bug Fix)
    cleaned_vals = [float(v) if v is not None else 0.0 for v in vals]

    try:
        x_input = pd.DataFrame([cleaned_vals], columns=features)
        
        # Get binary prediction (0 or 1)
        y_pred = model.predict(x_input)[0]
        
        # Get probabilities for both classes
        # probas[0] is for class 0 (Malignant), probas[1] is for class 1 (Benign)
        probas = model.predict_proba(x_input)[0]
        prob_malignant = probas[0]
        prob_benign = probas[1]

        # Determine label based on Jupyter logic (0=Malignant, 1=Benign)
        if y_pred == 0:
            result_text = "MALIGNANT"
            result_color = "#FFD700" # Warning Gold
            badge_color = "danger"
        else:
            result_text = "BENIGN"
            result_color = "#00FF7F" # Spring Green
            badge_color = "success"

        return html.Div([
            html.Hr(style={"borderTop": "1px solid white"}),
            html.H3([
                f"Model Classification: ",
                dbc.Badge(result_text, color=badge_color, className="ms-2")
            ], style={"fontWeight": "bold"}),
            
            dbc.Row([
                dbc.Col([
                    html.P(f"Probability of Malignant (Class 0): {prob_malignant:.2%}"),
                    dbc.Progress(value=prob_malignant*100, color="dark", style={"height": "10px"})
                ], md=6),
                dbc.Col([
                    html.P(f"Probability of Benign (Class 1): {prob_benign:.2%}"),
                    dbc.Progress(value=prob_benign*100, color="info", style={"height": "10px"})
                ], md=6),
            ], className="mt-3")
        ], style={"color": "white"})

    except Exception as e:
        return html.Div(f"Prediction Error: {e}", style={"color": "white", "background": "red", "padding": "10px"})