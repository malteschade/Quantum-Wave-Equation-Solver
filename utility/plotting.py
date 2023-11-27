import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _plot_vals(val1, val2, plot_val1: bool = True, plot_val2: bool = True,
                val1_range: tuple = (0, 1), val2_range: tuple = (0, 1), title: str = '',
                val1_name: str = '', val2_name: str = '', x_name: str = '') -> go.Figure:
    x = np.arange(np.max([len(val1), len(val2)]))
    
    fig = make_subplots(rows=1, cols=1,subplot_titles=[title], specs=[[{'secondary_y': True}]])
    
    if plot_val2:
        fig.add_trace(go.Scatter(x=x, y=val2, mode='lines', name=val2_name, line=dict(color='blue'),
                                showlegend=False), secondary_y=True)
    
    if plot_val1:
        fig.add_trace(go.Scatter(x=x, y=val1, mode='lines', name=val1_name, line=dict(color='red'),
                                showlegend=False), secondary_y=False)
    
    fig.update_yaxes(title_text=val1_name, range=val1_range, exponentformat='e',
                        showexponent='all', title_font=dict(color='red'))
    fig.update_yaxes(title_text=val2_name, secondary_y=True,
                        range=val2_range, title_font=dict(color='blue'))
    fig.update_xaxes(title_text=x_name)
    
    fig.update_layout(font=dict(size=15))
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=20)
        annotation['yshift'] = 10
    return fig

def plot_uv(u, v, time, range_u: tuple = (-1, 1), range_v: tuple = (-4000, 4000), plot_u=True, plot_v=False) -> go.Figure:
    return _plot_vals(u, v, plot_val1=plot_u, plot_val2=plot_v, val1_range=range_u, val2_range=range_v,
                            title=f'$$Wave \; Field \; t = {np.round(time, 8)}$$', val1_name=r'$u [\mu m]$', val2_name=r'$v [\mu m / s^2]$',
                            x_name=r'$x [m]$')

def plot_state(state, time,  range_u: tuple = (-1, 1), range_v: tuple = (-3000, 3000), plot_u=True, plot_v=False) -> go.Figure:
    state_u, state_v = state[:len(state)//2], state[len(state)//2:]
    return _plot_vals(state_u, state_v, plot_val1=plot_u, plot_val2=plot_v, val1_range=range_u, val2_range=range_v,
                            title=f'$$Statevector \; t={np.round(time, 8)}$$', val1_name=r'$TM^{1/2}u$', val2_name=r'$TM^{1/2}v$',
                            x_name=r'$x [m]$')
    
def plot_medium(mu, rho, range_mu: tuple = (0, 5e10), range_rho: tuple = (0, 5e3), plot_mu=True, plot_rho=True) -> go.Figure:
    return _plot_vals(mu, rho, plot_val1=plot_mu, plot_val2=plot_rho, val1_range=range_mu, val2_range=range_rho,
                            title=r'$Medium \; Parameters$', val1_name=r'$\mu [Pa]$', val2_name=r'$\rho [kg/m^3]$',
                            x_name=r'$x [m]$')
    
def plot_heatmap(array: np.ndarray, title='') -> go.Figure:
    assert array.ndim == 2, "Array must be 2D"
    array = np.imag(array) if ~np.all(np.isreal(array)) else array
    fig = go.Figure(data=go.Heatmap(z=array, colorscale='Viridis'))
    fig.update_layout(
        font=dict(size=15),
        title=f'$$Matrix \; Heatmap \; ({title})$$',
        xaxis=dict(scaleanchor='y', constrain='domain', title=f'$$Matrix \; Columns ={array.shape[1]}$$'),
        yaxis=dict(autorange='reversed', title=f'$$Matrix \; Rows = {array.shape[0]}$$')
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=20)
    return fig
