from common_imports import *
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid import make_axes_locatable
from nitime.utils import triu_indices

def plot_coocurrence(df, cmap=plt.cm.gray_r, color_anchor=[0,1],
  x_tick_rot=90, size=None, title=None, plot_vals=True,
  second_order=False):
    """
    Take df, a DataFrame of class occurence ground truth, and plot 
    a heat map of conditional occurence, where cell (i,j) means
    P(C_i|C_j). The last column in the K x (K+2) heat map corresponds
    to the prior P(c_i).

    If second_order, plots (K choose 2) x (K+2) heat map corresponding
    to P(C_i|C_j,C_k): second-order correlations.

    Return the figure.
    """
    # Construct the conditional co-occurence
    
    combinations = [x for x in itertools.combinations(df.columns,2)]
    combination_strs = ['%s, %s'%(x[0],x[1]) for x in combinations]

    total = df.shape[0]
    N = len(df.columns)
    K = len(combinations) if second_order else len(df.columns)
    data = np.zeros((K,N+1))
    prior = np.zeros(K)
    for i in range(0,K):
      if not second_order:
        cls = df.columns[i]
        conditioned = df[df[cls]]
      else:
        cls1 = combinations[i][0]
        cls2 = combinations[i][1]
        conditioned = df[df[cls1]&df[cls2]]      

      # count all the classes
      data[i,:-1] = conditioned.sum()

      # count the number of times that cls was the only one present
      if not second_order:
        data[i,-1] = ((conditioned.sum(1)-1)==0).sum()
      else:
        data[i,-1] = ((conditioned.sum(1)-2)==0).sum()

      # normalize
      max_val = np.max(data[i,:])
      data[i,:] /= max_val
      data[i,:][data[i,:]==1]=np.nan

      # use the max count to compute the prior
      prior[i] = max_val / total

      index = combination_strs if second_order else df.columns
      columns = df.columns.tolist()+['nothing']
      m = DataFrame(data,index=index,columns=columns)

    # Insert P(X) as the last column
    m.insert(N+1,'prior',prior)

    # Sort by prior
    m = m.sort('prior',ascending=False)

    if size:
      fig = plt.figure(figsize=size)
    else:
      w=max(12,m.shape[1]/1)
      h=max(12,m.shape[0]/1)
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

    ax.set_yticks(np.arange(K))
    ax.set_yticklabels(m.index,size='x-large')

    ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(-.5,K+0.5)))
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

    ax.set_ybound([-0.5, K - 0.5])
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
    l = mpl.lines.Line2D([N-0.5,N-0.5],[-.5,K-0.5],
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
f = plot_coocurrence(df,plt.cm.Reds)
f.savefig('../data/results/dataset_stats/synth_cooccur.png')
f = plot_coocurrence(df,plt.cm.Reds,second_order=True)
f.savefig('../data/results/dataset_stats/synth_cooccur_2.png')

df2 = DataFrame.load('pascal_val.df')
#df2 = df2[['aeroplane','bird','car','person','tvmonitor']]
f3 = plot_coocurrence(df2,plt.cm.Reds,[0,1])
f3.savefig('../data/results/dataset_stats/pascal_val_cooccur.png')
f3 = plot_coocurrence(df2,plt.cm.Reds,[0,1],second_order=True)
f3.savefig('../data/results/dataset_stats/pascal_val_cooccur_2.png')

df2 = DataFrame.load('pascal_train.df')
#df2 = df2[['aeroplane','bird','car','chair','diningtable','person','tvmonitor']]
f3 = plot_coocurrence(df2,plt.cm.Reds,[0,1])
f3.savefig('../data/results/dataset_stats/pascal_train_cooccur.png')
f3 = plot_coocurrence(df2,plt.cm.Reds,[0,1],second_order=True)
#f3.savefig('../data/results/dataset_stats/pascal_train_cooccur_2.png')