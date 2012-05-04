from common_imports import *
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid import make_axes_locatable
from nitime.utils import triu_indices

def drawmatrix_channels(df, x_tick_rot=90,
                        size=None, cmap=plt.cm.gray_r,
                        color_anchor=[0,1], title=None):
    
    m = df.corr().abs()
    N = m.shape[0]

    # clear diagonal (all 1s)
    np.fill_diagonal(m.as_matrix(),np.nan)

    # Compute normalized occurences
    prior = 1.*df.sum()/df.sum().sum()
    m.insert(N,'prior',prior)
    channel_names = m.columns

    fig = plt.figure()

    if size is not None:
        fig.set_figwidth(size[0])
        fig.set_figheight(size[1])

    ax_im = fig.add_subplot(1, 1, 1)

    divider = make_axes_locatable(ax_im)
    ax_cb = divider.new_vertical(size="10%", pad=0.1, pack_start=True)
    fig.add_axes(ax_cb)

    #Extract the minimum and maximum values for scaling of the
    #colormap/colorbar:
    max_val = np.nanmax(m)
    min_val = np.nanmin(m)

    if color_anchor is None:
        color_min = min_val
        color_max = max_val
    elif color_anchor == 0:
        bound = max(abs(max_val), abs(min_val))
        color_min = -bound
        color_max = bound
    else:
        color_min = color_anchor[0]
        color_max = color_anchor[1]

    #The call to imshow produces the matrix plot:
    im = ax_im.imshow(m, origin='upper', interpolation='nearest',
       vmin=color_min, vmax=color_max, cmap=cmap)

    #Formatting:
    ax = ax_im
    ax.grid(True)

    #ax.set_axis_off()
    ax.set_xticks(np.arange(N+1))
    ax.set_xticklabels(channel_names,rotation=x_tick_rot)
    fig.autofmt_xdate(rotation=x_tick_rot)
    for tick in ax.xaxis.iter_ticks():
      tick[0].label2On = True
      tick[0].label1On = False
      tick[0].label2.set_rotation(x_tick_rot)

    ax.set_yticks(np.arange(N))
    ax.set_yticklabels(channel_names)

    ax.set_ybound([-0.5, N - 0.5])
    ax.set_xbound([-0.5, N + 0.5])

    #Make the tick-marks invisible:
    for line in ax.xaxis.get_ticklines():
        line.set_markeredgewidth(0)

    for line in ax.yaxis.get_ticklines():
        line.set_markeredgewidth(0)

    if title is not None:
        ax.set_title(title)

    #The following produces the colorbar and sets the ticks
    #Set the ticks - if 0 is in the interval of values, set that, as well
    #as the maximal and minimal values:
    if min_val < 0:
        ticks = [color_min, min_val, 0, max_val, color_max]
    #Otherwise - only set the minimal and maximal value:
    else:
        ticks = [color_min, min_val, max_val, color_max]
    ticks = np.unique(ticks)

    # line separating prior
    l = mpl.lines.Line2D([N-0.48,N-0.48],[-.48,N-0.48],ls='--',c='gray')
    l = ax.add_line(l)

    #This makes the colorbar:
    cb = fig.colorbar(im, cax=ax_cb, orientation='horizontal',
                      cmap=cmap,
                      norm=im.norm,
                      boundaries=np.linspace(color_min, color_max, 256),
                      ticks=ticks,
                      format='%.2f')
    return fig

df = DataFrame.load('temp.df')
cmap = plt.cm.gray_r
f = drawmatrix_channels(df,cmap=cmap)

cmap = plt.cm.jet
df = DataFrame.load('temp2.df')
f2 = drawmatrix_channels(df,cmap=cmap)
f3 = drawmatrix_channels(df,size=(10,10),cmap=cmap)