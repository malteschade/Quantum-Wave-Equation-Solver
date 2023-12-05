import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
plt.rcParams['font.family'] = 'Times New Roman'

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

def plot_anim(y_values_list: list, y_axis_list: list,
              fps: int = 2, filename: str = 'wave_field.mp4') -> plt.Figure:
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    lines = []
    for ax, y_values, y_axis in zip(axs, y_values_list, y_axis_list):
        line, = ax.plot(y_values[0, :], color='blue')
        ax.set_ylim(y_axis)
        lines.append(line)

    def animate(i):
        for line, y_values in zip(lines, y_values_list):
            line.set_ydata(y_values[i, :])
        return lines

    anim = FuncAnimation(fig, animate, frames=len(y_values_list[0]), interval=3000, blit=True)
    if filename:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(filename, writer=writer)
    else:
        return anim

def plot_multi_v1(idx, times, ode, sim, qpu, rho, mu, bcs):
    nx = len(rho)

    lbcs, rbcs = bcs['left'], bcs['right']
    pos_new = {k: np.zeros((len(times), nx+2)) for k in ['ode', 'sim', 'qpu']}

    for key in pos_new:
        pos_new[key][:, 1:-1] = locals()[key]
        if lbcs == 'NBC':
            pos_new[key][:, 0] = locals()[key][:, 0]
        if rbcs == 'NBC':
            pos_new[key][:, -1] = locals()[key][:, -1]

    ode = pos_new['ode']
    sim = pos_new['sim']
    qpu = pos_new['qpu']

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))

    ax = axes[0, 0]
    ax2 = ax.twinx()

    ax.plot(np.arange(nx), rho, color='blue', label='$\\rho$')
    ax.set_ylabel('$\\rho$ [kg/m$^3$]', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_ylim(1e3, 5e3)
    lines, labels = ax.get_legend_handles_labels()

    ax2.plot(np.arange(nx), mu[:-1], color='red', label='$\\mu$')
    ax2.set_ylabel('$\\mu$ [Pa]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0.5e10, 4.5e10)
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(lines + lines2, labels + labels2, loc='lower right')

    for i_plt, i in enumerate(idx):
        ax = axes[(i_plt+1) // 3, (i_plt+1) % 3]
        sns.scatterplot(x=np.arange(nx+2), y=ode[i], ax=ax,
                        label='Classical ODE Solver', color='black')
        sns.lineplot(x=np.arange(nx+2), y=sim[i], ax=ax,
                     label='Noise Free Simulator', color='red')
        sns.lineplot(x=np.arange(nx+2), y=qpu[i], ax=ax,
                     label='Quantum Computer', color='blue')
        ax.set_title(f"t = {times[i]:.4f} s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("u [$\mu$ m]")
        ax.set_ylim(-1, 1)
        if i_plt == 0:
            ax.legend(loc='lower right')
        else:
            ax.legend([],[], frameon=False)

    plt.tight_layout()
    return fig
