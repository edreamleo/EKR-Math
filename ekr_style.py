#@+leo-ver=5-thin
#@+node:ekr.20241218044150.1: * @file ekr_style.py
#@@language python
#@@nopyflakes  # There are python errors in this file.

# style file for plt.style.use(style_file)
# See https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html

# Entries are RcParams instances:
#  https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams

axes.facecolor: white
axes.grid : True

text.hinting_factor : 8
xtick.direction: out
ytick.direction: out

grid.color: lightgrey
grid.linestyle: -    # solid line

figure.facecolor: (.90, .90, .90)
figure.edgecolor: 0.50
figure.figsize: 8, 5.5

axes.prop_cycle: cycler('color', ['black', 'lightgrey', 'indigo', 'red', 'blue', 'cyan', 'gray', 'magenta'])

mathtext.fontset : stix
#@-leo
