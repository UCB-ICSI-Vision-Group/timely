from synthetic.common_imports import *

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid import make_axes_locatable

cmap = plt.cm.Blues
#cmap = plt.cm.gray_r

df = DataFrame.load('temp2.df')
N = len(df.columns)

def channel_formatter(x, pos=None):
  thisind = np.clip(int(x), 0, N - 1)
  return channel_names[thisind]

# TODO: set w and h based on N
fig = plt.figure()
#fig.set_figwidth(w)
#fig.set_figheight(h)
ax = fig.add_subplot(1,1,1)
ax.set_frame_on(False)

divider = make_axes_locatable(ax)
ax_cb = divider.new_vertical(size="10%", pad=0.2, pack_start=True)
fig.add_axes(ax_cb)

# Compute normalized occurences
prior = 1.*df.sum()/df.sum().sum()
data = df.corr().abs()
# set the diagonal to 0
np.fill_diagonal(data.as_matrix(),0)
data.insert(N,'prior',prior)

color_min = 0.
color_max = 1.
im = ax.imshow(data, origin='upper', interpolation='nearest',
  vmin=color_min, vmax=color_max, cmap=cmap)

for i in xrange(0, data.shape[0]):
            if i < (data.shape[0] - 1):
                ax.text(i - 0.3, i, df.columns[i], rotation=0)
            if i > 0:
                ax.text(-1, i + 0.3, df.columns[i],
                        horizontalalignment='right')

ax.set_xticks(np.arange(N+1))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(channel_formatter))

ax.set_yticks(np.arange(N))
ax.set_yticklabels(df.columns)

ax.set_xticklabels(data.columns)
for tick in ax.xaxis.iter_ticks():
  tick[0].label2On = True
  tick[0].label1On = False

ax.set_ybound([-0.5, N - 0.5])
ax.set_xbound([-0.5, N - 1.5])

ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(-.5,N+0.5)))
ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(-.5,N+1.5)))
#ax.grid(False,which='major')
#ax.grid(True,which='minor',ls='-',lw=7,c='w')

# TODO: *sigh* the thin lines at the edges are driving me crazy but oh well

#Make the tick-marks invisible:
for line in ax.xaxis.get_ticklines()+ax.yaxis.get_ticklines():
  line.set_markeredgewidth(0)
for line in ax.xaxis.get_minorticklines()+ax.yaxis.get_minorticklines():
  line.set_markeredgewidth(0)

max_val = np.nanmax(data)
min_val = np.nanmin(data)

ticks = [color_min,min_val,max_val,color_max] 
cb = fig.colorbar(im, cax=ax_cb, orientation='horizontal',
  norm=im.norm, boundaries=np.linspace(color_min, color_max, 256),
  cmap=cmap, ticks=ticks, format='%.2f')
cb.ax.set_frame_on(False)

im.set_extent((-0.5,3.5,2.5,-0.5))
l = mpl.lines.Line2D([N-0.5,N-0.5],[-.5,N-0.5],ls='--',c='gray')
l = ax.add_line(l)
l.set_zorder(3)

# TODO: don't want to figure out how to do this right now
#ax2 = ax.twinx()
#ax2.grid(True,lw=5)
#ax2.patch.set_alpha(0)

fig.savefig('temp.png')