import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')

plt.rcParams['xtick.major.size']  = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size']  = 2
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size']  = 5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size']  = 2
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.direction']   = 'in'
plt.rcParams['ytick.direction']   = 'in'
plt.rcParams['xtick.major.pad']   = 8
plt.rcParams['xtick.top']         = True
plt.rcParams['ytick.right']       = True
plt.rcParams["figure.figsize"]    = (7, 6)
plt.rcParams["mathtext.fontset"]  = 'stix'
plt.rcParams["font.family"]       = 'STIXGeneral'
plt.rcParams["font.size"]         = 65