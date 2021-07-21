import os
import sys
from pathlib import Path

import pandas as pd

base_path = Path(os.getcwd().split('/flightmare/')[0]+'/flightmare/')
sys.path.insert(0, base_path.as_posix())

from analysis.utils import *

# Plot flight times of original trajectories
filepaths = sorted((base_path/'analysis'/'logs'/'dda_encfts-coll-medtraj'
                                                '').rglob(
    '*original.csv'))

times = []
for f in filepaths:
    df = pd.read_csv(f)
    times.append(df['time-since-start [s]'].iloc[-1] - df['time-since-start ['
                                                          's]'].iloc[0])
times = np.array(times)

plt.hist(times, bins=50)
plt.title('median={}, iqr={}'.format(np.median(times), iqr(times)))
plt.show()