import plotly
import plotly.graph_objs as go

import numpy as np


def plot_histogram(data):

    plot_data = []

    for dat in data:
        plot_data.append(go.Histogram(
            x=dat,
            opacity=0.75,
            xbins=dict(
                start=0,
                end=350,
                size=5,
            ),
        ))

    layout = go.Layout(
        barmode='group',
        bargap=0.2,
        bargroupgap=-0.3,
    )

    fig = go.Figure(data=plot_data, layout=layout)
    plotly.offline.plot(fig)
