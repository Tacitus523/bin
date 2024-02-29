import matplotlib.pyplot as plt
import numpy as np

labels = 'Classical MD', 'Random Displacement', 'Metadynamics', 'Potential Energy Scan'
sizes = np.array([9998, 9000, 13548-5112, 5112], dtype=int)

def absolute_value(val):
    a  = np.round(val/100.*sizes.sum(), 0)
    return f"{a: 5.0f}"

fig = plt.figure(figsize=(8,8))
labelsize=18
plt.pie(sizes, labels=labels, autopct=absolute_value, rotatelabels=False, startangle=90, textprops={'fontsize': labelsize}, counterclock=False)
#plt.legend(labels, loc=1, bbox_to_anchor=(1.15, 1.15), prop={'size': labelsize})
plt.tight_layout()
fig.savefig("structure_sources.png", transparent=True)