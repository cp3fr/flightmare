from pyforest import *

base_path = Path(os.getcwd().split('/flightmare/')[0]+'/flightmare/')
sys.path.insert(0, base_path.as_posix())

from analysis.utils import *

"""
README


  - "mpc_nw_act"=offline evaluation (mpc in control, network predicts commands) 
  - "mpc2nw"=online evaluation (network in control) 
"""

def main(
        base_path,
        to_override = True,
        to_import_logs = True,
        to_get_performance = False,
        to_make_summary_table = False,
        to_get_subject_performance=False,
        to_get_performance_summary=False,
        to_plot_traj_3d = True,
        to_plot_state = True,
        ):
    """Processes network flight trajectory logs, checks for collisions
    and gate passes, extract performance features and makes some plots."""
    # Settings.
    collider_dict = {'gate-wall': ['gate', 'wall'], 'wall': ['wall']}
    track_dict = {'flat': './tracks/flat.csv', 'wave': './tracks/wave.csv'}
    path = {}
    path['logs'] = base_path / 'analysis' / 'logs'
    path['perf'] = base_path / 'analysis' / 'performance'

    models = [f.name for f in path['logs'].glob('*/')]

    #todo: computer freezes if more than 1 parallel process, THUS FIX
    # NUM_PARALLEL_PROCESSES=1 because plotting trajectory 3d and plotting of
    # state causes trouble, needs debugging.
    if to_import_logs:
        for m in models:
            filepaths = sorted((path['logs']/m).rglob('*.csv'))
            filepaths = [f for f in filepaths if f.name!='original.csv']
            for f in filepaths:
                import_log(filepath=f,to_override=to_override,to_plot_traj_3d=
                    to_plot_traj_3d,to_plot_state=to_plot_state)

    if to_get_performance:
        for c in collider_dict:
            p=get_performance(collider_name=c, models=models)
            o='./performance/{}/'.format(c)
            make_path(o)
            p.to_csv(o+'performance.csv',index=False)

    #todo: update this script for the updated dataset exporting
    if to_make_summary_table:
        for c in collider_dict:
            curr_path = './performance/{}/'.format(c)
            performance = pd.read_csv(curr_path+'performance.csv')
            for online_name in ['online', 'offline']:
                for trajectory_name in ['reference','other-laps','other-track',
                        'multi-laps']:
                    make_summary_table(c,curr_path,performance,online_name,
                        trajectory_name)

    if to_get_subject_performance:
        for c in collider_dict:
            get_subject_performance(base_path=base_path, collider_name=c)

    if to_get_performance_summary:
        for c in collider_dict:
            p=pd.read_csv(path['perf']/c/'subject_performance.csv')
            p=clean_performance_table(p)
            r=get_average_performance(p)
            for n in ['train','test']:
                plot_success_rate_by_gate(r,dataset=n,control='nw')
                plt.savefig(path['perf']/c/'plots'/
                    ('success_rate_{}.png'.format(n)))
                plt.close(plt.gcf())
            for n in ['train', 'test']:
                t = get_command_prediction_table(r, dataset=n,control='mpc')
                t.to_latex(path['perf']/c/'plots'/(
                    'command_prediction_{}.csv'.format(n)), index=True)
                print(n.upper())
                print(t)
                print()



if __name__ == '__main__':
    main(base_path)