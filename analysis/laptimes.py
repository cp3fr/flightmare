import pandas as pd
from pyforest import *

base_path = Path(os.getcwd().split('/flightmare/')[0]+'/flightmare/')
sys.path.insert(0, base_path.as_posix())

from analysis.utils import *

path={}
path['logs']=base_path/'analysis'/'logs'
path['process']=base_path/'analysis'/'process'


filepaths=sorted(path['process'].rglob(
    'dda_img-coll-totmedtraj/*/mpc_nw_act_00/events.csv'))
pprint(filepaths)

ddict={}
for f in filepaths:
    ddict.setdefault('subject',[])
    ddict['subject'].append(int(f.parts[-3].split('-')[2].replace('s','')))
    ddict.setdefault('dataset', [])
    ddict['dataset'].append(f.parts[-3].split('-')[-1].split('_')[0])
    e=pd.read_csv(f)
    obj_id=8
    ind=(e['object-id'].values==obj_id) & (e['is-pass'].values==1)
    t=e.loc[ind,'t'].values
    ddict.setdefault('start_time', [])
    ddict['start_time'].append(t[0])
    ddict.setdefault('end_time', [])
    ddict['end_time'].append(t[-1])
    ddict.setdefault('lap_time', [])
    ddict['lap_time'].append(np.diff(t)[0])
    ddict.setdefault('start_object_id', [])
    ddict['start_object_id'].append(obj_id)
    ddict.setdefault('end_object_id', [])
    ddict['end_object_id'].append(obj_id)
    ddict.setdefault('filepath', [])
    ddict['filepath'].append(f)
pprint(ddict)

l=pd.DataFrame(ddict)
print(l)

for n in l['dataset'].unique():
    ind=l['dataset'].values==n
    x=l.loc[ind,'lap_time'].values
    print('{} dataset, {} subjects, lap time median={:.3f}s, mean={'
          ':.3f}s, '
        'min={:.3f}s, max={:.3f}s'.format(n,np.sum(ind),np.nanmedian(x),
        np.nanmean(x),np.nanmin(x),np.nanmax(x)))