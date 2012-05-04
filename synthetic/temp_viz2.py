from common_imports import *
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid import make_axes_locatable
from nitime.utils import triu_indices

def plot_first_order_conditional(df, cmap=plt.cm.gray_r, color_anchor=[0,1],
  x_tick_rot=90, size=None, title=None, plot_vals=True):
    """
    Take df, a DataFrame of class occurence ground truth, and plot 
    a heat map of conditional occurence, where cell (i,j) means
    P(C_i|C_j). The last column in the K x (K+1) heat map corresponds
    to the prior P(c_i).

    Return the figure.
    """
    # Construct the conditional co-occurence
    N = len(df.columns)
    data = np.zeros((N,N))
    for i in range(0,N):
      data[i,:] = df[df[df.columns[i]]].sum()
    data/=data.diagonal()
    m = DataFrame(data.T,index=df.columns,columns=df.columns)    
    np.fill_diagonal(m.as_matrix(),np.nan)

    # Compute P(-|X) from sum over y of P(y|X)
    nothing_given_X = 1-m.sum(1)
    m.insert(N,'nothing',nothing_given_X)

    # Compute priors
    prior = 1.*df.sum()/df.sum().sum()
    m.insert(N+1,'prior',prior)

    if size:
      fig = plt.figure(figsize=size)
    else:
      w=max(6,m.shape[1]/1)
      h=max(6,m.shape[0]/1)
      fig = plt.figure(figsize=(w,h))
    ax_im = fig.add_subplot(111)

    # make axes for colorbar
    divider = make_axes_locatable(ax_im)
    ax_cb = divider.new_vertical(size="5%", pad=0.1, pack_start=True)
    fig.add_axes(ax_cb)

    #The call to imshow produces the matrix plot:
    im = ax_im.imshow(m, origin='upper', interpolation='nearest',
       vmin=color_anchor[0], vmax=color_anchor[1], cmap=cmap)

    #Formatting:
    ax = ax_im
    ax.set_xticks(np.arange(N+2))
    ax.set_xticklabels(m.columns)
    for tick in ax.xaxis.iter_ticks():
      tick[0].label2On = True
      tick[0].label1On = False
      tick[0].label2.set_rotation(x_tick_rot)
      tick[0].label2.set_fontsize('x-large')

    ax.set_yticks(np.arange(N))
    ax.set_yticklabels(m.columns,size='x-large')

    ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(-.5,N+0.5)))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(-.5,N+1.5)))
    ax.grid(False,which='major')
    ax.grid(True,which='minor',ls='-',lw=7,c='w')

    #Make the tick-marks invisible:
    for line in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        line.set_markeredgewidth(0)
    for line in ax.xaxis.get_minorticklines()+ax.yaxis.get_minorticklines():
        line.set_markeredgewidth(0)
    if title is not None:
        ax.set_title(title)

    ax.set_ybound([-0.5, N - 0.5])
    ax.set_xbound([-0.5, N + 1.5])

    #The following produces the colorbar and sets the ticks
    #Set the ticks - if 0 is in the interval of values, set that, as well
    #as the maximal and minimal values:
    #Extract the minimum and maximum values for scaling
    max_val = np.nanmax(m)
    min_val = np.nanmin(m)
    if min_val < 0:
        ticks = [color_anchor[0], min_val, 0, max_val, color_anchor[1]]
    #Otherwise - only set the maximal value:
    else:
        ticks = [color_anchor[0], max_val, color_anchor[1]]

    # lines separating 'nothing' and 'prior'
    l = mpl.lines.Line2D([N-0.5,N-0.5],[-.5,N-0.5],
      ls='--',c='gray',lw=2)
    l = ax.add_line(l)
    l.set_zorder(3)

    for i in xrange(0, m.shape[0]):
      for j in xrange(0,m.shape[1]):
        val = m.as_matrix()[i,j]
        if not np.isnan(val):
          ax.text(j-0.2,i+0.1,'%.2f'%val)

    # Just doing ax.set_frame_on(False) results in weird thin lines
    # from imshow() on the edges. This covers them up.
    for spine in ax.spines.values():
        spine.set_edgecolor('w')

    #This makes the colorbar:
    cb = fig.colorbar(im, cax=ax_cb, orientation='horizontal',
            cmap=cmap, ticks=ticks, format='%.2f')
    cb.ax.artists.remove(cb.outline)
    return fig

df = DataFrame.load('temp.df')
f = plot_first_order_conditional(df,plt.cm.gray_r)
f.savefig('temp_synth.png')

df2 = DataFrame.load('temp2.df')
f3 = plot_first_order_conditional(df2,plt.cm.Reds,[0,1])
f3.savefig('temp_pascal.png')