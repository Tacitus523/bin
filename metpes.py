#!/usr/bin/python3
"""
  Script creates metad pes from hills

  Change for your needs.

  Credits to DM

  USAGE: METPES e_max [folder]

"""

################
#import modules#
################

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

#################
##User Settings##
#################
FES_FILE = "fes.dat"

TITEL = ""

LOWER_LIMIT = 2           # Lower limit on x and y axis
UPPER_LIMIT = 3.3         # Upper limit on x and y axis
TICK_SPACE = 0.3          # Spacing between ticks on x and y axis

LABEL_FS = 18             # Font size of labels
TICKS_FS = 16             # Tick size of labels
LABELPAD = -.8            # Padding in front of the labels

CB_SIZE_FACTOR = 0.98     # Factor for colorbar scaling, adjust to scale the colorbar to fit the size of the graph


if not sys.argv[1]:
	print('Usage: METPES e_max [folder]')
	exit()  
elif sys.argv[1] == '--help':
	print('Usage: METPES e_max [folder]')
	exit()
else:
  e_max = float(sys.argv[1])

if sys.argv[2]:
  fes_folder = sys.argv[2]
  FES_FILE = os.path.join(fes_folder, FES_FILE)

x, y, srp_m1 = np.loadtxt(FES_FILE, usecols=(0,1,2), skiprows=0, unpack=True)

xx = np.linspace(np.min(x)*10, np.max(x)*10, 95)
yy = np.linspace(np.min(y)*10, np.max(y)*10, 95)
xx, yy = np.meshgrid(xx, yy)


#fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=[14,8], dpi=300)
fig = plt.figure(figsize=[7,6], dpi=400)

ax = plt.gca()
ax.set_xlim(LOWER_LIMIT, UPPER_LIMIT)
ax.set_ylim(LOWER_LIMIT, UPPER_LIMIT)
ax.set_xticks(ticks=np.arange(LOWER_LIMIT, UPPER_LIMIT, TICK_SPACE))
ax.set_yticks(ticks=np.arange(LOWER_LIMIT, UPPER_LIMIT, TICK_SPACE))
ax.tick_params(axis='x', labelsize=TICKS_FS)
ax.tick_params(axis='y', labelsize=TICKS_FS)
ax.set_aspect('equal')
	
# labels, latex math style
ax.set_xlabel(u"S$^\mathrm{1}$\N{MINUS SIGN}S$^\mathrm{2}$ [\u212B]", 
				fontsize=LABEL_FS, labelpad=LABELPAD)
ax.set_ylabel(u"S$^\mathrm{2}$\N{MINUS SIGN}S$^\mathrm{3}$ [\u212B]", 
				fontsize=LABEL_FS, labelpad=LABELPAD)

######################
### plot 1 -- SRP/M1 #
######################
lines=np.arange(0, e_max+.1, 1.0)
linesf=np.arange(0, e_max+.1, 0.1)
cbticks=np.arange(0, e_max+.1, 5)

energies = srp_m1
if TITEL:
  ax.set_title(TITEL, fontsize=LABEL_FS, pad=0, linespacing=0)

# kJ/mol 
#zz = np.reshape(energies, xx.shape)

# kcal/mol
zz = np.reshape(energies/4.184, xx.shape)
fig1 = ax.contourf(xx, yy, zz, cmap="viridis", levels=linesf)
fig2 = ax.contour(xx, yy, zz, levels=lines, colors="black", linewidths=0.3)

cb = plt.colorbar(fig1, ticks=cbticks, ax=ax, pad=0.02, shrink=CB_SIZE_FACTOR)
cb.set_label("$\Delta$G [kcal/mol]", fontsize=LABEL_FS, labelpad=5, rotation=90)
cb.ax.tick_params(labelsize=TICKS_FS)

# Labeling
loc_x = 0.2
loc_y = 11.3
ax.text(loc_x, loc_y, "A)", fontsize=LABEL_FS+2, fontweight='bold', va='top', ha='right')

#fig.tight_layout()
plt.savefig("PMF.png")
