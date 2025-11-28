import dash
from dash import html, dcc
from dash import dash_table
import pandas as pd

dash.register_page(__name__, path="/", name="Introduction")

# -------------------------------------------------------------
#  Summary text of original paper
# -------------------------------------------------------------
paper_summary = """
The Breast Cancer Wisconsin Diagnostic dataset originates from the work of
W. Nick Street, William H. Wolberg, and O. L. Mangasarian (1992). Their study
introduced an automated system for classifying breast tumors using
fine-needle aspirated (FNA) cytology images. 

The system extracted quantitative characteristics of cell nuclei — such as radius,
texture, smoothness, concavity, and symmetry — using active contour models 
(“snakes”) to trace boundaries of nuclei precisely. With these features, the authors 
used linear-programming-based classifiers and achieved cross-validation accuracy 
reaching **97%**, marking one of the earliest strong uses of machine learning in 
medical diagnosis.
"""

# -------------------------------------------------------------
# Feature table
# -------------------------------------------------------------
feature_table = pd.DataFrame({
    "Feature Group": [
        "Radius", "Texture", "Perimeter", "Area", "Smoothness",
        "Compactness", "Concavity", "Concave Points",
        "Symmetry", "Fractal Dimension"
    ],
    "Description": [
        "Mean distance from center to perimeter",
        "Variation in pixel intensity",
        "Boundary length of the nucleus",
        "Size of the cell nucleus",
        "Local variation in radius",
        "Perimeter² / area",
        "Severity of concave regions",
        "Number of concave contour portions",
        "Nuclear symmetry",
        "Irregularity of contour"
    ]
})

# -------------------------------------------------------------
# Component style helpers (consistent with your app theme)
# -------------------------------------------------------------
CARD_STYLE = {
    "background": "white",
    "padding": "25px",
    "borderRadius": "12px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.1)",
    "marginBottom": "25px"
}

TITLE_STYLE = {
    "fontSize": "26px",
    "fontWeight": "600",
    "marginBottom": "15px",
    "color": "#333"
}

TEXT_STYLE = {
    "fontSize": "17px",
    "lineHeight": "1.6",
    "textAlign": "justify",
    "color": "#444"
}

# -------------------------------------------------------------
# Introduction Page Layout
# -------------------------------------------------------------
layout = html.Div(
    style={
        "maxWidth": "950px",
        "margin": "auto",
        "padding": "30px"
    },
    children=[

        # -----------------------
        # Title card
        # -----------------------
        html.Div(
            style=CARD_STYLE,
            children=[
                html.H2("Introduction", style=TITLE_STYLE),
                html.P(paper_summary, style=TEXT_STYLE)
            ]
        ),

        # -----------------------
        # Image Card
        # -----------------------
        html.Div(
            style=CARD_STYLE,
            children=[
                html.Img(
                    src="/assets/celltumor.PNG",
                    style={
                        "width": "100%",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 10px rgba(0,0,0,0.15)"
                    }
                ),
                html.P(
                    "Figure: Example of cell nuclei boundaries from the original study.",
                    style={"textAlign": "center", "fontStyle": "italic", "marginTop": "10px"}
                )
            ]
        ),

        # -----------------------
        # Dataset info card
        # -----------------------
        html.Div(
            style=CARD_STYLE,
            children=[
                html.H3("Dataset Overview", style=TITLE_STYLE),
                html.P(
                    """
                    The Breast Cancer Wisconsin Diagnostic dataset contains 569 samples 
                    describing malignant and benign tumors. Each sample includes 30 numerical 
                    features derived from nuclear characteristics. The values are computed as:
                    mean, standard error, and “worst case” (largest) measures.
                    """,
                    style=TEXT_STYLE
                ),

                dash_table.DataTable(
                    data=feature_table.to_dict('records'),
                    columns=[{"name": col, "id": col} for col in feature_table.columns],
                    style_cell={
                        "fontSize": "16px",
                        "padding": "8px",
                        "textAlign": "left"
                    },
                    style_header={
                        "backgroundColor": "#fafafa",
                        "fontWeight": "600",
                        "fontSize": "17px",
                        "borderBottom": "1px solid #ddd"
                    },
                    style_table={
                        "marginTop": "20px",
                        "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                        "borderRadius": "8px",
                        "overflow": "hidden"
                    }
                )
            ]
        )
    ]
)
