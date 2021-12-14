import pandas as pd
from pyforest import *

base_path = Path(os.getcwd().split('/flightmare/')[0]+'/flightmare/')
sys.path.insert(0, base_path.as_posix())

from analysis.utils import *

path={}
path['logs']=base_path/'analysis'/'logs'
path['process']=base_path/'analysis'/'process'

filepaths=sorted(path['process'].rglob(
    'dda_img-coll-totmedtraj/*/mpc_nw_act_00/reference.csv'))
pprint(filepaths)

d=pd.DataFrame([])
for f in filepaths:
    r = pd.read_csv(f).loc[:,('t', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'ax',
                              'ay', 'az')]
    r['subject']=int(f.parts[-3].split('-')[2].replace('s',''))
    r['dataset']=f.parts[-3].split('-')[-1].split('_')[0]
    e=pd.read_csv(f.parent/'events.csv')
    obj_id=8
    ind=(e['object-id'].values==obj_id) & (e['is-pass'].values==1)
    t=e.loc[ind,'t'].values
    r['start_time']=t[0]
    r['end_time']=t[1]
    r['lap_time']=np.diff(t)[0]
    r['filepath']=f
    d=d.append(r)
print(d.shape)
print(d.columns)

plt.figure()
for tt in d['dataset'].unique():
    cdict={'train':'b', 'test':'r'}
    label=tt.upper()
    for s in d['subject'].unique():
        ind=((d['dataset'].values==tt) &
            (d['subject'].values==s))
        x=d.loc[ind,'px'].values
        y=d.loc[ind,'py'].values
        plt.plot(x,y,color=cdict[tt],label=label)
        label='_nolegend_'
plt.xlabel('Position X [m]')
plt.xlabel('Position Y [m]')
plt.legend()
plt.show()

#todo: plot drone state variables across time to show variance between
# trajectories