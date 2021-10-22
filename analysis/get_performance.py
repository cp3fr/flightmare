import os
import sys

from pathlib import Path

import pandas as pd

base_path = Path(os.getcwd().split('/flightmare/')[0]+'/flightmare/')
sys.path.insert(0, base_path.as_posix())

from analysis.utils import *


path={}
path['logs']=base_path/'analysis'/'logs'
path['perf']=base_path/'analysis'/'performance'/'gate-wall'


# Load performance table.
p=pd.read_csv(path['perf']/'subject_performance.csv')

# Clean performance table.
def clean_performance_table(p):
    """Clean performance table for subsequent analyses."""
    p['model']=[n.split('-')[0] for n in p['model'].values]
    ind=[True if n != 'attbr' else False for n in p['model'].values]
    p=p.loc[ind,:]
    return p
p=clean_performance_table(p)
print(p.columns)
print(p)

# Compute average performance.
def get_average_performance(p):
    """Collect a network performance summary, i.e. mean across 10 flights of the network per trajectory"""
    ddict={}
    for subject in p['subject'].unique():
        for dataset in p['dataset'].unique():
            for model in p['model'].unique():
                for track in p['track'].unique():
                    ind=(
                        (p['subject'].values==subject) &
                        (p['dataset'].values==dataset) &
                        (p['model'].values==model) &
                        (p['track'].values==track)
                    )
                    n='subject'
                    ddict.setdefault(n,[])
                    ddict[n].append(subject)
                    n='dataset'
                    ddict.setdefault(n, [])
                    ddict[n].append(dataset)
                    n = 'model'
                    ddict.setdefault(n, [])
                    ddict[n].append(model)
                    n = 'track'
                    ddict.setdefault(n, [])
                    ddict[n].append(track)
                    n = 'num_trials'
                    ddict.setdefault(n, [])
                    ddict[n].append(np.sum(ind))
                    for i in np.arange(0,11,1):
                        n = 'sr{}'.format(i)
                        ddict.setdefault(n, [])
                        value = np.mean((p.loc[ind,'num_gates_passed'].values>=i).astype(float))
                        ddict[n].append(value)
                    for n in ['travel_distance',
                       'median_path_deviation', 'iqr_path_deviation', 'num_gates_passed','num_collisions', 'network_used',
                       'pitch_error_l1', 'pitch_error_l1-median', 'pitch_error_mse',
                       'pitch_error_mse-median', 'roll_error_l1', 'roll_error_l1-median',
                       'roll_error_mse', 'roll_error_mse-median', 'throttle_error_l1',
                       'throttle_error_l1-median', 'throttle_error_mse',
                       'throttle_error_mse-median', 'yaw_error_l1', 'yaw_error_l1-median',
                       'yaw_error_mse', 'yaw_error_mse-median',]:
                        ddict.setdefault(n, [])
                        ddict[n].append(np.nanmean(p.loc[ind,n].values))
    r=pd.DataFrame(ddict)
    r=r.sort_values(by=['model','subject','dataset'])
    return r
r=get_average_performance(p)

# Plot network success rate as function of number of gates.
def plot_success_rate_by_gate(r,
        dataset='test',
        models=['img', 'ftstr', 'encfts'],
        names=['RGB Images','Feature Tracks', 'Attention Prediction'],
        colors=['k', 'b', 'r'],
        ) -> None:
    """Plot success rate of different networks as a function of gates passed."""
    plt.figure()
    plt.gcf().set_figwidth(6)
    plt.gcf().set_figheight(3)
    icolor=0
    iname=0
    for model in models:
        ind=(
            (r['dataset'].values==dataset) &
            (r['model'].values==model)
        )
        c=colors[icolor]
        m=names[iname]
        x = np.arange(0, 11, 1)
        values=r.loc[ind,(['sr{}'.format(i) for i in x])].values
        ci=confidence_interval(values)
        y=np.mean(values,axis=0)
        plt.fill_between(x,ci[0,:],ci[1,:],color=c,alpha=0.1)
        plt.plot(x,y,'-o',color=c,label='{}'.format(m))
        icolor+=1
        iname+=1
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=3)
    plt.xticks(x)
    plt.yticks(np.arange(0,1.1,0.25))
    plt.xlabel('Gates Passed')
    plt.ylabel('Success Rate')
    plt.xlim((0.9,10.1))
    plt.ylim((0,1.025))
    plt.grid(axis='y')
    plt.tight_layout()
for n in ['train','test']:
    plot_success_rate_by_gate(r,dataset=n)
    plt.savefig(path['perf']/'plots'/('success_rate_{}.png'.format(n)))
    plt.close(plt.gcf())

# Make command prediction table.
def get_command_prediction_table(r,
        dataset='test',
        models=['img', 'ftstr', 'encfts'],
        names=['RGB Images', 'Feature Tracks', 'Attention Prediction'],
        precision=2,
        ) -> None:
    """Make a table of command prediction resusults for different networks"""
    ddict={}
    imodel=0
    for model in models:
        ind=(
            (r['dataset'].values==dataset) &
            (r['model'].values==model)
        )
        m=names[imodel]
        n='Model Name'
        ddict.setdefault(n,[])
        ddict[n].append(m)
        for cmd in ['throttle','roll','pitch', 'yaw']:
            for metric in ['mse','l1']:
                n='{}_{}'.format(cmd,metric)
                ddict.setdefault(n,[])
                values=r.loc[ind,'{}_error_{}'.format(cmd,metric)].values
                y=np.nanmean(values)
                ddict[n].append(np.round(y,precision))
        imodel+=1
    t=pd.DataFrame(ddict)
    t=t.set_index('Model Name')
    t=t.T
    t['Command'] = [n.split('_')[0].capitalize() for n in t.index]
    t['Metric'] = [n.split('_')[1].upper() for n in t.index]
    t=t.set_index(['Command','Metric']).T
    return t
for n in ['train','test']:
    t=get_command_prediction_table(r,dataset=n)
    t.to_latex(path['perf']/'plots'/('command_prediction_{}.csv'.format(n)), index=True)
    print(n.upper())
    print(t)
    print()
