import os
import sys
from pathlib import Path

import pandas as pd

base_path = Path(os.getcwd().split('/flightmare/')[0]+'/flightmare/')
sys.path.insert(0, base_path.as_posix())

from analysis.utils import *


f = Path('/home/cp3fr/Desktop')/'RWB_FLAT1_DRONE.csv'

df = pd.read_csv(f)

plt.figure()
plt.subplot(2,1,1)
for n in ['qx','qy','qz','qw']:
    plt.plot(df['t'].values, df[n].values, label=n)
plt.legend()
plt.subplot(2,1,2)
for n in ['wx','wy','wz']:
    plt.plot(df['t'].values, df[n].values, label=n)
plt.legend()
plt.show()
