from pyforest import *

base_path = Path(os.getcwd().split('/flightmare/')[0]+'/flightmare/')
sys.path.insert(0, base_path.as_posix())

from analysis.utils import *


def main(
    to_process = True,
    to_override = False,
    to_plot_traj_3d = True,
    to_plot_state = True,
    num_parallel_processes = 2,
    to_performance = False,
    to_table = False,
    to_plot_reference = False,
    to_plot_reference_with_decision = False,
    to_collect_results=False,
        ):
    """Processes network flight trajectory logs, checks for collisions
    and gate passes, extract performance features and makes some plots."""
    collider_dict = {'gate-wall': ['gate', 'wall'], 'wall': ['wall']}
    collider_names = collider_dict.keys()
    track_filepaths = {'flat': './tracks/flat.csv', 'wave': './tracks/wave.csv'}
    logfile_path = base_path/'analysis'/'logs'
    models = [f.name for f in logfile_path.glob('*/')]


    def process(
            path,
            workers=1,
            exclude=['original.csv'],
            override=False,
            ):
        """Process raw files in parallel."""
        f=sorted(path.rglob('*.csv'))
        for n in exclude:
            f=[_f for _f in f if _f.name!=n]
        map = [(_f, override) for _f in f]
        with Pool(workers) as p:
            p.starmap(process_individual_run, map)
    for m in models:
        print(logfile_path/m)
        process(path=logfile_path/m, workers=num_parallel_processes)


    # if to_process:
    #     for model in models:
    #         log_filepaths = sorted((logfile_path/model).rglob('*.csv'))
    #         log_filepaths = [f for f in log_filepaths if f.name != 'original.csv']
    #         map = [(f.as_posix(), to_override, to_plot_traj_3d, to_plot_state)
    #                for f in log_filepaths]
    #         with Pool(num_parallel_processes) as p:
    #             p.starmap(process_individual_run, map)

    #
    # if to_performance:
    #     for collider_name in collider_names:
    #         curr_feature_filename = 'features_{}.csv'.format(collider_name)
    #         outpath = './performance/{}/'.format(collider_name)
    #         outfilepath = outpath + 'performance.csv'
    #         if not os.path.exists(outpath):
    #             make_path(outpath)
    #         performance = pd.DataFrame([])
    #         for model in models:
    #             filepaths = []
    #             for w in os.walk('./process/'+model+'/'):
    #                 for f in w[2]:
    #                     if f==curr_feature_filename:
    #                         filepaths.append(os.path.join(w[0], f))
    #             for filepath in filepaths:
    #                 print('..collecting performance: {}'.format(filepath))
    #                 df =  pd.read_csv(filepath)
    #                 # Get model and run information from filepath
    #                 strings = (
    #                     df['filepath'].iloc[0]
    #                         .split('/process/')[-1]
    #                         .split('/trajectory.csv')[0]
    #                         .split('/')
    #                 )
    #                 if len(strings) == 2:
    #                     strings.insert(1, 's016_r05_flat_li01_buffer20')
    #                 # Load the yaml file
    #                 yamlpath = None
    #                 config = None
    #                 yamlcount = 0
    #                 for w in os.walk('./logs/'+model+'/'):
    #                     for f in w[2]:
    #                         if f.find('.yaml')>-1:
    #                             yamlpath = os.path.join(w[0], f)
    #                             yamlcount += 1
    #                 if yamlpath is not None:
    #                     with open(yamlpath, 'r') as stream:
    #                         try:
    #                             config = yaml.safe_load(stream)
    #                         except yaml.YAMLError as exc:
    #                             print(exc)
    #                             config = None
    #                 # Make Data dictionnairy for the output
    #                 ddict = {}
    #                 ddict['model_name'] = strings[0]
    #                 ddict['has_dda'] = int(strings[0].find('dda') > -1)
    #                 if ((strings[2].find('mpc_nw_act') > -1) &
    #                     (filepath.find('mpc_nw_act') > -1)):
    #                     ddict['has_network_used'] = 0
    #                 else:
    #                     ddict['has_network_used'] = 1
    #                 if config is not None:
    #                     ddict['has_yaml'] = 1
    #                     ddict['has_ref'] = 1
    #                     if 'no_ref' in config['train']:
    #                         ddict['has_ref'] = int(config['train']['no_ref'] == False)
    #                     ddict['has_state_q'] = 0
    #                     ddict['has_state_v'] = 0
    #                     ddict['has_state_w'] = 0
    #                     if 'use_imu' in config['train']:
    #                         if config['train']['use_imu'] == True:
    #                             ddict['has_state_q'] = 1
    #                             ddict['has_state_v'] = 1
    #                             ddict['has_state_w'] = 1
    #                             if 'imu_no_rot' in config['train']:
    #                                 if config['train']['imu_no_rot'] == True:
    #                                     ddict['has_state_q'] = 0
    #                             if 'imu_no_vels' in config['train']:
    #                                 if config['train']['imu_no_vels'] == True:
    #                                     ddict['has_state_v'] = 0
    #                                     ddict['has_state_w'] = 0
    #                     ddict['has_fts'] = 0
    #                     if 'use_fts_tracks' in config['train']:
    #                         if config['train']['use_fts_tracks']:
    #                             ddict['has_fts'] = 1
    #                     ddict['has_decfts'] = 0
    #                     ddict['has_gztr'] = 0
    #                     ddict['has_encfts'] = 0
    #                     if 'attention_fts_type' in config['train']:
    #                         if config['train']['attention_fts_type'] == 'decoder_fts':
    #                             ddict['has_decfts'] = 1
    #                             ddict['has_gztr'] = 0
    #                             ddict['has_encfts'] = 0
    #                         elif config['train']['attention_fts_type'] == 'gaze_tracks':
    #                             ddict['has_decfts'] = 0
    #                             ddict['has_gztr'] = 1
    #                             ddict['has_encfts'] = 0
    #                         elif config['train']['attention_fts_type'] == \
    #                                 'encoder_fts':
    #                             ddict['has_decfts'] = 0
    #                             ddict['has_gztr'] = 0
    #                             ddict['has_encfts'] = 1
    #                     ddict['has_attbr'] = 0
    #                     if 'attention_branching' in config['train']:
    #                         if config['train']['attention_branching'] == True:
    #                             ddict['has_attbr'] = 1
    #                     ddict['buffer'] = 0
    #                     if 'start_buffer' in config['simulation']:
    #                         ddict['buffer'] = config['simulation']['start_buffer']
    #                 else:
    #                     ddict['has_yaml'] = 0
    #                 ddict['buffer'] = float(strings[1].split('buffer')[-1]) / 10
    #                 ddict['subject'] = int(
    #                     strings[1].split('_')[0].split('-')[2].split('s')[-1])
    #                 ddict['run'] = int(strings[2].split('_')[-1])
    #                 ddict['track'] = strings[1].split('_')[0].split('-')[1]
    #                 li_string = strings[1].split('_')[3].split('li')[-1]
    #                 if li_string.find('-')>-1:
    #                     ddict['li'] = int(li_string.split('-')[0])
    #                     ddict['num_laps'] = (int(li_string.split('-')[-1]) -
    #                                          int(li_string.split('-')[0]) + 1)
    #                 else:
    #                     ddict['li'] = int(li_string)
    #                     ddict['num_laps'] = 1
    #                 if ddict['has_dda'] == 0:
    #                     if strings[2] == 'reference_mpc':
    #                         ddict['mt'] = -1
    #                         ddict['st'] = 0
    #                         ddict['repetition'] = 0
    #                     else:
    #                         ddict['mt'] = -1
    #                         ddict['st'] = int(strings[2].split('_')[1].split('switch-')[-1])
    #                         ddict['repetition'] = int(strings[2].split('_')[-1])
    #                 else:
    #                     if strings[0].find('dda_offline')>-1:
    #                         ddict['mt'] = -1
    #                         ddict['st'] = int(strings[2].split('_')[1].split('st-')[-1])
    #                         ddict['repetition'] = int(strings[2].split('_')[-1])
    #                     elif strings[2].find('mpc_eval_nw')>-1:
    #                         ddict['mt'] = -1
    #                         ddict['st'] = -1
    #                         ddict['repetition'] = 0
    #                     elif strings[2].find('mpc_nw_act')>-1:
    #                         ddict['mt'] = -1
    #                         ddict['st'] = -1
    #                         ddict['repetition'] = 0
    #                     else:
    #                         ddict['mt'] = int(strings[2].split('_')[1].split('mt-')[-1])
    #                         ddict['st'] = int(strings[2].split('_')[2].split('st-')[-1])
    #                         ddict['repetition'] = int(strings[2].split('_')[-1])
    #                 # Add data dictionnairy as output row
    #                 for k in sorted(ddict):
    #                     df[k] = ddict[k]
    #                 performance = performance.append(df)
    #         # Save perfromance dataframe
    #         performance.to_csv(outfilepath, index=False)
    #
    #
    # if to_table:
    #
    #     for collider_name in collider_names:
    #         curr_path = './performance/{}/'.format(collider_name)
    #
    #         performance = pd.read_csv(curr_path + 'performance.csv')
    #
    #         for online_name in ['online', 'offline']:
    #             for trajectory_name in ['reference', 'other-laps', 'other-track',
    #                                     'multi-laps']:
    #
    #                 print('----------------')
    #                 print(online_name, trajectory_name)
    #                 print('----------------')
    #
    #                 # Subject dictionnairy
    #                 run_dict = None
    #                 exclude_run_dict = None
    #                 if trajectory_name == 'reference':
    #                     run_dict = {
    #                     'track': 'flat',
    #                     'subject': 16,
    #                     'run': 5,
    #                     'li': 1,
    #                     'num_laps': 1,
    #                     }
    #                 elif trajectory_name == 'other-laps':
    #                     run_dict = {
    #                         'track': 'flat',
    #                         'num_laps': 1,
    #                     }
    #                     exclude_run_dict = {
    #                         'track': 'flat',
    #                         'subject': 16,
    #                         'run': 5,
    #                         'li': 1,
    #                         'num_laps': 1,
    #                     }
    #                 elif trajectory_name == 'other-track':
    #                     run_dict = {
    #                         'track': 'wave',
    #                         'num_laps': 1,
    #                     }
    #                 elif trajectory_name == 'multi-laps':
    #                     run_dict = {
    #                         'track': 'flat',
    #                     }
    #                     exclude_run_dict = {
    #                         'num_laps': 1,
    #                     }
    #
    #                 # Network general dictionnairy
    #                 if online_name=='online':
    #                     network_dict = {
    #                         'has_dda': 1,
    #                         'has_network_used': 1,
    #                     }
    #                 else:
    #                     network_dict = {
    #                         'has_dda': 1,
    #                         'has_network_used': 0,
    #                     }
    #
    #                 # Model dictionnairy
    #                 model_dicts = [
    #                     {
    #                         'name': 'Ref + RVW (Baseline)',
    #                         'specs': {
    #                             'has_ref': 1,
    #                             'has_state_q': 1,
    #                             'has_state_v': 1,
    #                             'has_state_w': 1,
    #                             'has_fts': 0,
    #                             'has_decfts': 0,
    #                             'has_attbr': 0,
    #                             'has_gztr': 0,
    #                         },
    #                     },
    #                     {
    #                         'name': 'Ref + RVW + Fts',
    #                         'specs': {
    #                             'has_ref': 1,
    #                             'has_state_q': 1,
    #                             'has_state_v': 1,
    #                             'has_state_w': 1,
    #                             'has_fts': 1,
    #                             'has_decfts': 0,
    #                             'has_attbr': 0,
    #                             'has_gztr': 0,
    #                         },
    #                     },{
    #                         'name': 'Ref + RVW + AIn',
    #                         'specs': {
    #                             'has_ref': 1,
    #                             'has_state_q': 1,
    #                             'has_state_v': 1,
    #                             'has_state_w': 1,
    #                             'has_fts': 0,
    #                             'has_decfts': 1,
    #                             'has_attbr': 0,
    #                             'has_gztr': 0,
    #                         },
    #                     },{
    #                         'name': 'Ref + RVW + Abr',
    #                         'specs': {
    #                             'has_ref': 1,
    #                             'has_state_q': 1,
    #                             'has_state_v': 1,
    #                             'has_state_w': 1,
    #                             'has_fts': 0,
    #                             'has_decfts': 0,
    #                             'has_attbr': 1,
    #                             'has_gztr': 0,
    #                         },
    #                     },{
    #                         'name': 'Ref (Baseline)',
    #                         'specs': {
    #                             'has_ref': 1,
    #                             'has_state_q': 0,
    #                             'has_state_v': 0,
    #                             'has_state_w': 0,
    #                             'has_fts': 0,
    #                             'has_decfts': 0,
    #                             'has_attbr': 0,
    #                             'has_gztr': 0,
    #                         },
    #                     },{
    #                         'name': 'Ref + Fts',
    #                         'specs': {
    #                             'has_ref': 1,
    #                             'has_state_q': 0,
    #                             'has_state_v': 0,
    #                             'has_state_w': 0,
    #                             'has_fts': 1,
    #                             'has_decfts': 0,
    #                             'has_attbr': 0,
    #                             'has_gztr': 0,
    #                         },
    #                     },{
    #                         'name': 'Ref + AIn',
    #                         'specs': {
    #                             'has_ref': 1,
    #                             'has_state_q': 0,
    #                             'has_state_v': 0,
    #                             'has_state_w': 0,
    #                             'has_fts': 0,
    #                             'has_decfts': 1,
    #                             'has_attbr': 0,
    #                             'has_gztr': 0,
    #                         },
    #                     },{
    #                         'name': 'Ref + ABr',
    #                         'specs': {
    #                             'has_ref': 1,
    #                             'has_state_q': 0,
    #                             'has_state_v': 0,
    #                             'has_state_w': 0,
    #                             'has_fts': 0,
    #                             'has_decfts': 0,
    #                             'has_attbr': 1,
    #                             'has_gztr': 0,
    #                         },
    #                     },{
    #                         'name': 'RVW (Baseline)',
    #                         'specs': {
    #                             'has_ref': 0,
    #                             'has_state_q': 1,
    #                             'has_state_v': 1,
    #                             'has_state_w': 1,
    #                             'has_fts': 0,
    #                             'has_decfts': 0,
    #                             'has_attbr': 0,
    #                             'has_gztr': 0,
    #                         },
    #                     },{
    #                         'name': 'RVW + Fts',
    #                         'specs': {
    #                             'has_ref': 0,
    #                             'has_state_q': 1,
    #                             'has_state_v': 1,
    #                             'has_state_w': 1,
    #                             'has_fts': 1,
    #                             'has_decfts': 0,
    #                             'has_attbr': 0,
    #                             'has_gztr': 0,
    #                         },
    #                     },{
    #                         'name': 'RVW + AIn',
    #                         'specs': {
    #                             'has_ref': 0,
    #                             'has_state_q': 1,
    #                             'has_state_v': 1,
    #                             'has_state_w': 1,
    #                             'has_fts': 0,
    #                             'has_decfts': 1,
    #                             'has_attbr': 0,
    #                             'has_gztr': 0,
    #                         },
    #                     },{
    #                         'name': 'RVW + ABr',
    #                         'specs': {
    #                             'has_ref': 0,
    #                             'has_state_q': 1,
    #                             'has_state_v': 1,
    #                             'has_state_w': 1,
    #                             'has_fts': 0,
    #                             'has_decfts': 0,
    #                             'has_attbr': 1,
    #                             'has_gztr': 0,
    #                         },
    #                     },
    #                 ]
    #
    #
    #
    #                 # Feature dictionnairy
    #                 if online_name=='online':
    #                     feature_dict={
    #                         'Flight Time [s]': {
    #                             'varname': 'flight_time',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': 'std',
    #                             'precision': 2
    #                             },
    #                         'Travel Distance [m]': {
    #                             'varname': 'travel_distance',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': 'std',
    #                             'precision': 2
    #                         },
    #                         'Mean Error [m]': {
    #                             'varname': 'median_path_deviation',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': 'std',
    #                             'precision': 2
    #                         },
    #                         'Gates Passed': {
    #                             'varname': 'num_gates_passed',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': 'std',
    #                             'precision': 2
    #                         },
    #                         '% Collision': {
    #                             'varname': 'num_collisions',
    #                             'track': '',
    #                             'first_line': 'percent',
    #                             'second_line': '',
    #                             'precision': 0
    #                         },
    #                     }
    #                 else:
    #                     feature_dict = {
    #                         'Throttle MSE': {
    #                             'varname': 'throttle_error_mse-median',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': '',
    #                             'precision': 3
    #                         },
    #                         'Throttle L1': {
    #                             'varname': 'throttle_error_l1-median',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': '',
    #                             'precision': 3
    #                         },
    #                         'Roll MSE': {
    #                             'varname': 'roll_error_mse-median',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': '',
    #                             'precision': 3
    #                         },
    #                         'Roll L1': {
    #                             'varname': 'roll_error_l1-median',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': '',
    #                             'precision': 3
    #                         },
    #                         'Pitch MSE': {
    #                             'varname': 'pitch_error_mse-median',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': '',
    #                             'precision': 3
    #                         },
    #                         'Pitch L1': {
    #                             'varname': 'pitch_error_l1-median',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': '',
    #                             'precision': 3
    #                         },
    #                         'Yaw MSE': {
    #                             'varname': 'yaw_error_mse-median',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': '',
    #                             'precision': 3
    #                         },
    #                         'Yaw L1': {
    #                             'varname': 'yaw_error_l1-median',
    #                             'track': '',
    #                             'first_line': 'mean',
    #                             'second_line': '',
    #                             'precision': 3
    #                         },
    #                     }
    #
    #                 # Make a table
    #                 if (run_dict is not None) and (model_dicts is not None):
    #                     table = pd.DataFrame([])
    #                     for mdict in model_dicts:
    #
    #                         #add subject dictionnairy to the model dictionnairy
    #                         mdict['specs'] = {**mdict['specs'],
    #                                           **network_dict,
    #                                           **run_dict}
    #
    #                         ddict = {}
    #                         ddict['Model'] = [mdict['name'], '']
    #
    #                         # Select performance data
    #                         # Runs to include
    #                         ind = np.array([True for i in range(performance.shape[0])])
    #                         for k, v in mdict['specs'].items():
    #                             ind = ind & (performance[k] == v)
    #                         # Runs to exclude
    #                         if exclude_run_dict is not None:
    #                             ind_exclude = np.array([True for i in range(
    #                                 performance.shape[0])])
    #                             for k, v in exclude_run_dict.items():
    #                                 ind_exclude = ind_exclude & (performance[k] == v)
    #                             # Combine run selection
    #                             ind = (ind == True) & (ind_exclude == False)
    #                         # Select current runs
    #                         curr_performance = performance.copy().loc[ind, :]
    #
    #                         ddict['Num Runs'] = [str(curr_performance.shape[0]), '']
    #
    #                         print(mdict['name'], ':', curr_performance.shape[0])
    #                         # print(curr_performance)
    #
    #                         if curr_performance.shape[0] > 0:
    #                             # Compute Average performances
    #                             for outvar in feature_dict:
    #                                 ddict.setdefault(outvar, [])
    #                                 invar = feature_dict[outvar]['varname']
    #                                 curr_vals = curr_performance[invar].values
    #                                 #first line value
    #                                 op1 = feature_dict[outvar]['first_line']
    #                                 if op1 == 'mean':
    #                                     val1 = np.nanmean(curr_vals)
    #                                 elif op1 == 'percent':
    #                                     val1 = 100 * np.mean((curr_vals > 0).astype(int))
    #                                 else:
    #                                     val1 = None
    #                                 if val1 is None:
    #                                     ddict[outvar].append('')
    #                                 else:
    #                                     ddict[outvar].append(str(np.round(val1, feature_dict[outvar][
    #                                         'precision'])))
    #                                 # second line value
    #                                 op2 = feature_dict[outvar]['second_line']
    #                                 if op2 == 'std':
    #                                     val1 = np.nanstd(curr_vals)
    #                                 else:
    #                                     val1 = None
    #                                 if val1 is None:
    #                                     ddict[outvar].append('')
    #                                 else:
    #                                     ddict[outvar].append('('+str(np.round(val1, feature_dict[outvar][
    #                                         'precision']))+')')
    #
    #                         for k in ddict:
    #                             ddict[k] = [' '.join(ddict[k])]
    #
    #                         # Append two lines to the output table
    #                         table = table.append(
    #                             pd.DataFrame(ddict,
    #                                          index=list(range(len(ddict['Model']))))
    #                         )
    #
    #                     outpath = curr_path + '/{}/'.format(online_name)
    #                     if not os.path.exists(outpath):
    #                         make_path(outpath)
    #
    #                     outfilepath = outpath + 'latex_table_{}.csv'.format(
    #                         trajectory_name)
    #                     table.to_latex(outfilepath, index=False)
    #
    #
    # if to_plot_reference:
    #     # Load track.
    #     track = pd.read_csv(track_filepaths['flat'])
    #     ndict = {
    #         'pos_x': 'px',
    #         'pos_y': 'py',
    #         'pos_z': 'pz',
    #         'rot_x_quat': 'qx',
    #         'rot_y_quat': 'qy',
    #         'rot_z_quat': 'qz',
    #         'rot_w_quat': 'qw',
    #         'dim_x': 'dx',
    #         'dim_y': 'dy',
    #         'dim_z': 'dz',
    #     }
    #     track = track.rename(columns=ndict)
    #     track = track[list(ndict.values())]
    #     track['pz'] += 0.35
    #     track['dx'] = 0.
    #     track['dy'] = 3
    #     track['dz'] = 3
    #     # Load reference and downsample to 20 Hz
    #     reference = trajectory_from_logfile(
    #         filepath=(base_path/'analysis'/'tracks'/'flat.csv').as_posix())
    #     sr = 1 / np.nanmedian(np.diff(reference.t.values))
    #     reference = reference.iloc[np.arange(0, reference.shape[0], int(sr / 20)), :]
    #     # Plot reference, track, and format figure.
    #     ax = plot_trajectory(
    #         reference.px.values,
    #         reference.py.values,
    #         reference.pz.values,
    #         reference.qx.values,
    #         reference.qy.values,
    #         reference.qz.values,
    #         reference.qw.values,
    #         axis_length=2,
    #         c='k',
    #     )
    #     ax = plot_gates_3d(
    #         track=track,
    #         ax=ax,
    #         color='k',
    #         width=4,
    #         )
    #     ax = format_trajectory_figure(
    #         ax=ax,
    #         xlims=(-15, 19),
    #         ylims=(-17, 17),
    #         zlims=(-8, 8),
    #         xlabel='px [m]',
    #         ylabel='py [m]',
    #         zlabel='pz [m]',
    #         title='',
    #         )
    #
    #     plt.axis('off')
    #     plt.grid(b=None)
    #     ax.view_init(elev=45,
    #                  azim=270)
    #     plt.gcf().set_size_inches(20,10)
    #
    #     plot_path = './plots/'
    #     if not os.path.exists(plot_path):
    #         make_path(plot_path)
    #
    #     plt.savefig(plot_path + 'reference_3d.jpg')
    #
    #
    # if to_plot_reference_with_decision:
    #     data_path = './branching_demo/'
    #     # Load track.
    #     track = pd.read_csv(data_path+'flat.csv')
    #     ndict = {
    #         'pos_x': 'px',
    #         'pos_y': 'py',
    #         'pos_z': 'pz',
    #         'rot_x_quat': 'qx',
    #         'rot_y_quat': 'qy',
    #         'rot_z_quat': 'qz',
    #         'rot_w_quat': 'qw',
    #         'dim_x': 'dx',
    #         'dim_y': 'dy',
    #         'dim_z': 'dz',
    #     }
    #     track = track.rename(columns=ndict)
    #     track = track[list(ndict.values())]
    #     track['pz'] += 0.35
    #     track['dx'] = 0.
    #     track['dy'] = 3
    #     track['dz'] = 3
    #     # Load reference and downsample to 20 Hz
    #     reference = trajectory_from_logfile(
    #         filepath=data_path+'trajectory_reference_original.csv')
    #     sr = 1 / np.nanmedian(np.diff(reference.t.values))
    #     reference = reference.iloc[np.arange(0, reference.shape[0], int(sr / 20)),
    #                 :]
    #     # Plot reference, track, and format figure.
    #     ax = plot_trajectory(
    #         reference.px.values,
    #         reference.py.values,
    #         reference.pz.values,
    #         reference.qx.values,
    #         reference.qy.values,
    #         reference.qz.values,
    #         reference.qw.values,
    #         axis_length=2,
    #         c='b',
    #         axis_colors='b',
    #     )
    #     ax = plot_gates_3d(
    #         track=track,
    #         ax=ax,
    #         color='k',
    #         width=4,
    #     )
    #     ax = format_trajectory_figure(
    #         ax=ax,
    #         xlims=(-15, 19),
    #         ylims=(-17, 17),
    #         zlims=(-8, 8),
    #         xlabel='px [m]',
    #         ylabel='py [m]',
    #         zlabel='pz [m]',
    #         title='',
    #     )
    #
    #     plt.axis('off')
    #     plt.grid(b=None)
    #     ax.view_init(elev=45,
    #                  azim=270)
    #     plt.gcf().set_size_inches(20, 10)
    #
    #     plot_path = './plots/'
    #     if not os.path.exists(plot_path):
    #         make_path(plot_path)
    #
    #     plt.savefig(plot_path + 'reference_flat_with_decision.jpg')
    #
    #
    # if to_collect_results:
    #
    #     # Some settings.
    #     collider_name = 'gate-wall'
    #
    #     # Process new logfiles.
    #     outpath = base_path / 'analysis' / 'performance' / collider_name / \
    #               'subject_performance.csv'
    #     print(outpath)
    #     if not outpath.exists():
    #         filepaths = sorted((base_path / 'analysis' / 'process').rglob(
    #             '*/features_{}.csv'.format(collider_name)))
    #         data = pd.DataFrame([])
    #         for f in filepaths:
    #             print(f.parts)
    #             df = pd.read_csv(f)
    #             df['model'] = f.parts[-4].split('_')[1]
    #             if f.parts[-3].find('trajectory') > -1:
    #                 df['track'] = f.parts[-3].split('_')[0].split('-')[1]
    #                 df['subject'] = int(
    #                     f.parts[-3].split('_')[0].split('-')[2].replace('s', ''))
    #                 df['dataset'] = f.parts[-3].split('_')[0].split('-')[3]
    #             else:
    #                 df['track'] = f.parts[-3].split('_')[2]
    #                 df['subject'] = int(f.parts[-3].split('_')[0].replace('s', ''))
    #                 df['dataset'] = 'test'
    #             df['mt'] = int(f.parts[-2].split('_')[1].split('-')[1])
    #             df['st'] = int(f.parts[-2].split('_')[2].split('-')[1])
    #             df['trial'] = int(f.parts[-2].split('_')[3])
    #             data = data.append(df)
    #         if not outpath.parent.exists():
    #             outpath.parent.mkdir(parents=True, exist_ok=True)
    #
    #         print(outpath)
    #         data.to_csv(outpath, index=False)
    #
    #     # Extract performance from logfiles.
    #     inpath = base_path / 'analysis' / 'performance' /  collider_name / 'subject_performance.csv'
    #     outpath = base_path / 'analysis' / 'performance' /  collider_name / 'summary.csv'
    #     if (inpath.exists()) & (not outpath.exists()):
    #         data = pd.read_csv(inpath)
    #         ddict = {}
    #         for model in data['model'].unique():
    #             for track in data['track'].unique():
    #                 for dataset in data['dataset'].unique():
    #                     print('--------------------------------')
    #                     for subject in data['subject'].unique():
    #                         ind = (
    #                                 (data['model'].values == model) &
    #                                 (data['track'].values == track) &
    #                                 (data['dataset'].values == dataset) &
    #                                 (data['subject'].values == subject)
    #                         )
    #                         num_samples = np.sum(ind)
    #                         num_coll_free = np.sum(
    #                             data.loc[ind, 'num_collisions'].values == 0)
    #                         num_gates_passed = np.sum(
    #                             data.loc[ind, 'num_gates_passed'].values == 11)
    #                         prop_coll_free = num_coll_free / num_samples
    #                         prop_gates_passed = num_gates_passed / num_samples
    #                         ddict.setdefault('model', [])
    #                         ddict['model'].append(model)
    #                         ddict.setdefault('track', [])
    #                         ddict['track'].append(track)
    #                         ddict.setdefault('dataset', [])
    #                         ddict['dataset'].append(dataset)
    #                         ddict.setdefault('subject', [])
    #                         ddict['subject'].append(subject)
    #                         ddict.setdefault('num_samples', [])
    #                         ddict['num_samples'].append(num_samples)
    #                         ddict.setdefault('num_coll_free', [])
    #                         ddict['num_coll_free'].append(num_coll_free)
    #                         ddict.setdefault('num_gates_passed', [])
    #                         ddict['num_gates_passed'].append(num_gates_passed)
    #                         ddict.setdefault('prop_coll_free', [])
    #                         ddict['prop_coll_free'].append(prop_coll_free)
    #                         ddict.setdefault('prop_gates_passed', [])
    #                         ddict['prop_gates_passed'].append(prop_gates_passed)
    #                         print(model, track, dataset, subject, num_samples,
    #                               num_coll_free, prop_coll_free)
    #         summary = pd.DataFrame(ddict)
    #         summary.to_csv(outpath, index=False)
    #
    #     # Plot performance tables and figure.
    #     to_plot_performance = True
    #     if to_plot_performance:
    #         inpath = base_path / 'analysis' / 'performance' /  collider_name / 'subject_performance.csv'
    #         inpath2 = base_path / 'analysis' / 'performance' /  collider_name / 'summary.csv'
    #         outpath = base_path / 'analysis' / 'performance' /  collider_name / 'plots'
    #         if inpath2.exists():
    #             # Load performance data
    #             data = pd.read_csv(inpath)
    #             summary = pd.read_csv(inpath2)
    #             # Loop over different model configurations
    #             for model in data['model'].unique():
    #                 for track in data['track'].unique():
    #                     for dataset in data['dataset'].unique():
    #                         # Determine if any data is available:
    #                         ind = ((data['model'].values == model) &
    #                                (data['track'].values == track) &
    #                                (data['dataset'].values == dataset))
    #                         if np.sum(ind) > 0:
    #                             # Make a figure that shows trajectories for all subjects
    #                             fig, axs = plt.subplots(5, 4)
    #                             fig.set_figwidth(18)
    #                             fig.set_figheight(10)
    #                             axs = axs.flatten()
    #                             i = 0
    #                             for subject in data['subject'].unique():
    #                                 # Determine Success rate
    #                                 ind = ((summary['model'].values == model) &
    #                                        (summary['track'].values == track) &
    #                                        (summary['dataset'].values == dataset) &
    #                                        (summary['subject'].values == subject))
    #                                 is_success = True
    #                                 fontweight = 'normal'
    #                                 fontcolor = 'black'
    #                                 frame_highlight = False
    #                                 gates_passed_rate = ''
    #                                 collision_free_rate = ''
    #                                 if np.sum(ind) > 0:
    #                                     _num_samples = summary.loc[
    #                                         ind, 'num_samples'].iloc[0]
    #                                     _num_coll_free = summary.loc[
    #                                         ind, 'num_coll_free'].iloc[0]
    #                                     _num_gates_passed = summary.loc[
    #                                         ind, 'num_gates_passed'].iloc[0]
    #                                     u = \
    #                                     summary.loc[ind, 'prop_gates_passed'].iloc[
    #                                         0]
    #                                     v = summary.loc[ind, 'prop_coll_free'].iloc[
    #                                         0]
    #                                     if not np.isnan(v):
    #                                         gates_passed_rate = ' | G: {}/{} ({:.0f}%)'.format(
    #                                             _num_gates_passed, _num_samples,
    #                                             u * 100)
    #                                         collision_free_rate = ' | C: {}/{} ({:.0f}%)'.format(
    #                                             _num_coll_free, _num_samples,
    #                                             v * 100)
    #                                         if (u < 1) | (v < 1):
    #                                             fontweight = 'bold'
    #                                             frame_highlight = True
    #                                             is_success = False
    #                                             fontcolor = 'red'
    #                                 # Plot trajectory
    #                                 ind = (
    #                                         (data['model'].values == model) &
    #                                         (data['track'].values == track) &
    #                                         (data['dataset'].values == dataset) &
    #                                         (data['subject'].values == subject) &
    #                                         (data['trial'].values == 0)
    #                                 )
    #                                 if np.sum(ind) > 0:
    #                                     f = (Path(data.loc[ind, 'filepath'].iloc[0])
    #                                          .parent / 'trajectory-with-gates_gate-wall_045x270.jpg')
    #                                     im = cv2.imread(f.as_posix())
    #                                     # crop image borders
    #                                     im = im[270:-340, 250:-250, :]
    #                                     # add color frame (if not full success)
    #                                     if not is_success:
    #                                         im = cv2.copyMakeBorder(im, 20, 20, 20,
    #                                                                 20,
    #                                                                 cv2.BORDER_CONSTANT,
    #                                                                 value=(
    #                                                                 255, 0, 0))
    #                                     axs[i].imshow(im)
    #                                 axs[i].axis('off')
    #                                 axs[i].set_title('s%03d' % subject +
    #                                                  gates_passed_rate +
    #                                                  collision_free_rate,
    #                                                  fontweight=fontweight,
    #                                                  color=fontcolor)
    #                                 # raise the panel counter
    #                                 i += 1
    #                             # remove axis from remaining panels
    #                             for i in range(i, axs.shape[0]):
    #                                 axs[i].axis('off')
    #                             plt.tight_layout()
    #                             # make output directory
    #                             if not outpath.exists():
    #                                 outpath.mkdir(parents=True, exist_ok=True)
    #                             # save the figure
    #                             op = (outpath / ('trajectories_{}_{}_{}.jpg'.format(
    #                                 model, track, dataset)))
    #                             fig.savefig(op.as_posix())
    #                             plt.close(fig)
    #                             fig = None
    #                             axs = None
    #                             # Pring overall success to prompt
    #                             ind = ((summary['model'].values == model) &
    #                                    (summary['track'].values == track) &
    #                                    (summary['dataset'].values == dataset))
    #                             num_samples = np.nansum(
    #                                 summary.loc[ind, 'num_samples'].values)
    #                             num_coll_free = np.nansum(summary.loc[ind,
    #                                                                   'num_coll_free'].values)
    #                             num_gates_passed = np.nansum(summary.loc[ind,
    #                                                                      'num_gates_passed'].values)
    #                             prop_coll_free = np.nan
    #                             prop_gates_passed = np.nan
    #
    #                             if num_samples > 0:
    #                                 prop_coll_free = num_coll_free / num_samples
    #                                 prop_gates_passed = num_gates_passed / num_samples
    #                                 print(
    #                                     'Success trajectories: {} {} {} | G: {}/{} ({:.0f}%) | C: {}/{} ({:.0f}%)'.format(
    #                                         model, track, dataset,
    #                                         num_gates_passed, num_samples,
    #                                         100 * prop_gates_passed,
    #                                         num_coll_free, num_samples,
    #                                         100 * prop_coll_free))
    #
    #     # Plot proportion of successful laps for each of the tracks
    #     to_plot_dist_successful_flight = True
    #     if to_plot_dist_successful_flight:
    #         inpath = base_path / 'analysis' / 'performance' /  collider_name / 'summary.csv'
    #         outpath = base_path / 'analysis' / 'performance' /  collider_name / 'success_by_subject.csv'
    #         if inpath.exists():
    #             summary = pd.read_csv(inpath)
    #             ddict = {}
    #             for subject in summary['subject'].unique():
    #                 # get laptime
    #                 f = [v for v in sorted((base_path/'analysis'/'logs').rglob(
    #                     '*original.csv'))
    #                      if v.as_posix().find('s%03d' % subject) > -1]
    #                 f=f[0]
    #                 # print('..loading reference trajectory {}'.format(f))
    #                 df = pd.read_csv(f)
    #                 laptime = (df['time-since-start [s]'].iloc[-1] -
    #                            df['time-since-start [s]'].iloc[0])
    #                 # get other performance features
    #                 ind = ((summary['subject'].values == subject) &
    #                        (summary['dataset'].values == 'train'))
    #                 num_samples = np.sum(summary.loc[ind, 'num_samples'].values)
    #                 num_gates_passed = np.sum(
    #                     summary.loc[ind, 'num_gates_passed'].values)
    #                 num_coll_free = np.sum(summary.loc[ind, 'num_coll_free'].values)
    #                 prop_gates_passed = num_gates_passed / num_samples
    #                 prop_coll_free = num_coll_free / num_samples
    #                 ddict.setdefault('subject', [])
    #                 ddict['subject'].append(subject)
    #                 ddict.setdefault('laptime', [])
    #                 ddict['laptime'].append(laptime)
    #                 ddict.setdefault('num_samples', [])
    #                 ddict['num_samples'].append(num_samples)
    #                 ddict.setdefault('num_gates_passed', [])
    #                 ddict['num_gates_passed'].append(num_gates_passed)
    #                 ddict.setdefault('num_coll_free', [])
    #                 ddict['num_coll_free'].append(num_coll_free)
    #                 ddict.setdefault('prop_gates_passed', [])
    #                 ddict['prop_gates_passed'].append(prop_gates_passed)
    #                 ddict.setdefault('prop_coll_free', [])
    #                 ddict['prop_coll_free'].append(prop_coll_free)
    #             df = pd.DataFrame(ddict)
    #             print(df)
    #             df.to_csv(outpath, index=False)
    #
    #             plt.figure()
    #             plt.gcf().set_figwidth(15)
    #             plt.gcf().set_figheight(10)
    #             plt.subplot(2, 1, 1)
    #             plt.bar(df['subject'].values - 0.15,
    #                     df['prop_gates_passed'],
    #                     width=0.3,
    #                     label='Gates')
    #             plt.bar(df['subject'].values + 0.15,
    #                     df['prop_coll_free'],
    #                     width=0.3,
    #                     label='Collisions')
    #             plt.xticks(df['subject'].values)
    #             plt.xlabel('Subject')
    #             plt.ylabel('Proportion Successful Laps')
    #             plt.legend()
    #             plt.subplot(2, 1, 2)
    #             plt.bar(df['subject'].values,
    #                     df['laptime'],
    #                     width=0.5,
    #                     label='Lap Time')
    #             plt.plot([df['subject'].min() - 0.25, df['subject'].max() + 0.25],
    #                      np.ones((2,)) * np.nanmedian(df['laptime'].values),
    #                      '--r',
    #                      lw=3,
    #                      label='Median')
    #             plt.xticks(df['subject'].values)
    #             plt.xlabel('Subject')
    #             plt.ylabel('Lap Time [s]')
    #             plt.tight_layout()
    #             plt.savefig(outpath.as_posix().replace('.csv', '.jpg'))
    #

if __name__ == '__main__':
    main()