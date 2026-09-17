"""Restrained Plotly figures for the single-patient review screen."""
from __future__ import annotations

import plotly.graph_objects as go

from .predictor import DurabilityPrediction


INK = "#22282D"
MUTED = "#697279"
GRID = "#E4E7E9"
TEAL = "#286E6A"
REFERENCE = "#AAB1B5"
PAPER = "#FFFFFF"


def valve_outlook_curve(
    prediction: DurabilityPrediction,
    *,
    median_year: float | None = None,
) -> go.Figure:
    """Compare this profile with its reference and mark its 50% point."""

    t = prediction.forecast_times
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=t, y=prediction.conditional_event_free_high * 100, mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
    figure.add_trace(go.Scatter(x=t, y=prediction.conditional_event_free_low * 100, mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(40,110,106,0.10)", name="89% uncertainty interval", hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=t, y=prediction.reference_event_free_median * 100, mode="lines", line={"color": REFERENCE, "width": 2, "dash": "dash"}, name="Approach-matched reference", hovertemplate="Reference<br>%{x:.1f} y: %{y:.1f}%<extra></extra>"))
    figure.add_trace(go.Scatter(x=t, y=prediction.conditional_event_free_median * 100, mode="lines", line={"color": TEAL, "width": 3}, name="This patient profile", hovertemplate="Patient profile<br>%{x:.1f} y: %{y:.1f}%<extra></extra>"))
    if median_year is not None and median_year <= float(t.max()):
        figure.add_hline(y=50, line_width=1, line_dash="dot", line_color="#C3C9CB")
        figure.add_vline(
            x=median_year,
            line_width=1.5,
            line_dash="dot",
            line_color="#926C29",
        )
        figure.add_annotation(
            x=median_year,
            y=0.55,
            yref="paper",
            text="Median durability estimate",
            showarrow=False,
            textangle=-90,
            font={"size": 11, "color": "#76551F"},
            xanchor="right",
        )
    figure.update_layout(
        height=460,
        margin={"l": 98, "r": 18, "t": 36, "b": 64},
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font={"family": "Arial, Segoe UI, sans-serif", "color": INK, "size": 12},
        hoverlabel={"bgcolor": INK, "font_color": "white", "bordercolor": INK},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "font": {"size": 11, "color": MUTED}},
    )
    figure.update_xaxes(title={"text": "Years from today", "standoff": 14}, range=[0, float(t.max())], dtick=2, showgrid=False, zeroline=False, linecolor=GRID)
    figure.update_yaxes(title={"text": "Probability of remaining free from valve failure", "standoff": 14}, range=[0, 100], dtick=20, ticksuffix="%", showgrid=True, gridcolor=GRID, zeroline=False)
    return figure
