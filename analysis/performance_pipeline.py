import os
import sys

# Add base path
base_path = os.getcwd().split('/flightmare/')[0]+'/flightmare/'
sys.path.insert(0, base_path)

from analysis.utils import *

# What to do
to_process = True
to_performance = True
to_table = True

to_override = False

to_plot_traj_3d = True
to_plot_state = True
to_plot_reference = True
to_plot_reference_with_decision = False

# Buffer time: time runs with  network were started earlier than the reference
# trajectory
buffer = 2.0
collider_names = ['gate-wall', 'wall']

# Number of parallel processes
processes = 8



collider_dict = {
    'gate-wall': ['gate', 'wall'],
    'wall': ['wall'],
}

# Chose input files and folders for processing.
track_filepaths = {
    'flat': './tracks/flat.csv',
    'wave': './tracks/wave.csv',
    }
logfile_path = './logs/'
reference_filepath = logfile_path+'reference/trajectory_reference_original.csv'
models = [
    # 'dda_flat_med_full_bf2_cf25_noref_nofts_attbr'
        ]

if len(models) == 0:
    for w in os.walk(logfile_path):
        if w[0] == logfile_path:
            models = w[1]
models = sorted(models)

# Make a dictionnairy for all possible models
all_models_dict = {}
for ref in ['ref', '']:
    for state in ['rvw', 'r', 'vw', '']:
        for fts in ['fts', '']:
            for att in ['ain', 'abr', 'gztr', '']:
                curr_name = [n for n in [ref, state, fts, att] if len(
                        n)>0]
                if len(curr_name)==0:
                    curr_name=''
                elif len(curr_name)==1:
                    curr_name=curr_name[0]
                else:
                    curr_name='+'.join(curr_name)
                all_models_dict.setdefault(curr_name,
                                           {'has_ref': 0,
                                            'has_state_q': 0,
                                            'has_state_v': 0,
                                            'has_state_w': 0,
                                            'has_fts': 0,
                                            'has_decfts': 0,
                                            'has_attbr': 0,
                                            'has_gztr': 0,
                                            })
                if ref=='ref':
                    all_models_dict[curr_name]['has_ref'] = 1
                if state=='rvw':
                    all_models_dict[curr_name]['has_state_q'] = 1
                    all_models_dict[curr_name]['has_state_v'] = 1
                    all_models_dict[curr_name]['has_state_w'] = 1
                elif state=='r':
                    all_models_dict[curr_name]['has_state_q'] = 1
                elif state=='vw':
                    all_models_dict[curr_name]['has_state_v'] = 1
                    all_models_dict[curr_name]['has_state_w'] = 1
                if fts=='fts':
                    all_models_dict[curr_name]['has_fts'] = 1
                if att == 'ain':
                    all_models_dict[curr_name]['has_decfts'] = 1
                elif att == 'abr':
                    all_models_dict[curr_name]['has_attbr'] = 1
                elif att == 'gztr':
                    all_models_dict[curr_name]['has_gztr'] = 1


# Process individual runs
if to_process:

    for model in models:

        # Make a list of input logfiles of network trajectories.
        log_filepaths = []
        for w in os.walk(os.path.join(logfile_path, model)):
            for f in w[2]:
                if (f.find('.csv') > 0) and (reference_filepath.find(f) < 0):
                    log_filepaths.append(os.path.join(w[0], f))

        # Make a map for parellel processing
        map = [(f,
                buffer,
                reference_filepath,
                track_filepaths,
                collider_dict,
                collider_names,
                to_override,
                to_plot_traj_3d,
                to_plot_state)
               for f in sorted(log_filepaths)]

        # Process the data in parallel
        with Pool(processes) as p:
            p.starmap(pl_process, map)


# Collect performance metrics across runs
if to_performance:
    for collider_name in collider_names:
        curr_feature_filename = 'features_{}.csv'.format(collider_name)
        outpath = './performance/{}/'.format(collider_name)
        outfilepath = outpath + 'performance.csv'
        if not os.path.exists(outpath):
            make_path(outpath)
        performance = pd.DataFrame([])
        for model in models:
            filepaths = []
            for w in os.walk('./process/'+model+'/'):
                for f in w[2]:
                    if f==curr_feature_filename:
                        filepaths.append(os.path.join(w[0], f))
            for filepath in filepaths:
                print('..collecting performance: {}'.format(filepath))
                df =  pd.read_csv(filepath)
                # Get model and run information from filepath
                strings = (
                    df['filepath'].iloc[0]
                        .split('/process/')[-1]
                        .split('/trajectory.csv')[0]
                        .split('/')
                )
                if len(strings) == 2:
                    strings.insert(1, 's016_r05_flat_li01_buffer20')
                # Load the yaml file
                yamlpath = None
                config = None
                yamlcount = 0
                for w in os.walk('./logs/'+model+'/'):
                    for f in w[2]:
                        if f.find('.yaml')>-1:
                            yamlpath = os.path.join(w[0], f)
                            yamlcount += 1
                if yamlpath is not None:
                    with open(yamlpath, 'r') as stream:
                        try:
                            config = yaml.safe_load(stream)
                        except yaml.YAMLError as exc:
                            print(exc)
                            config = None
                # Make Data dictionnairy for the output
                ddict = {}
                ddict['model_name'] = strings[0]
                ddict['has_dda'] = int(strings[0].find('dda') > -1)
                if ((strings[2].find('mpc_nw_act') > -1) &
                    (filepath.find('mpc_nw_act') > -1)):
                    ddict['has_network_used'] = 0
                else:
                    ddict['has_network_used'] = 1
                if config is not None:
                    ddict['has_yaml'] = 1
                    ddict['has_ref'] = 1
                    if 'no_ref' in config['train']:
                        ddict['has_ref'] = int(config['train']['no_ref'] == False)
                    ddict['has_state_q'] = 0
                    ddict['has_state_v'] = 0
                    ddict['has_state_w'] = 0
                    if 'use_imu' in config['train']:
                        if config['train']['use_imu'] == True:
                            ddict['has_state_q'] = 1
                            ddict['has_state_v'] = 1
                            ddict['has_state_w'] = 1
                            if 'imu_no_rot' in config['train']:
                                if config['train']['imu_no_rot'] == True:
                                    ddict['has_state_q'] = 0
                            if 'imu_no_vels' in config['train']:
                                if config['train']['imu_no_vels'] == True:
                                    ddict['has_state_v'] = 0
                                    ddict['has_state_w'] = 0
                    ddict['has_fts'] = 0
                    if 'use_fts_tracks' in config['train']:
                        if config['train']['use_fts_tracks']:
                            ddict['has_fts'] = 1
                    ddict['has_decfts'] = 0
                    ddict['has_gztr'] = 0
                    if 'attention_fts_type' in config['train']:
                        if config['train']['attention_fts_type'] == 'decoder_fts':
                            ddict['has_decfts'] = 1
                            ddict['has_gztr'] = 0
                        elif config['train']['attention_fts_type'] == 'gaze_tracks':
                            ddict['has_decfts'] = 0
                            ddict['has_gztr'] = 1
                    ddict['has_attbr'] = 0
                    if 'attention_branching' in config['train']:
                        if config['train']['attention_branching'] == True:
                            ddict['has_attbr'] = 1
                    ddict['buffer'] = 0
                    if 'start_buffer' in config['simulation']:
                        ddict['buffer'] = config['simulation']['start_buffer']
                else:
                    ddict['has_yaml'] = 0
                ddict['buffer'] = float(strings[1].split('buffer')[-1]) / 10
                ddict['subject'] = int(strings[1].split('_')[0].split('s')[-1])
                ddict['run'] = int(strings[1].split('_')[1].split('r')[-1])
                ddict['track'] = strings[1].split('_')[2]
                li_string = strings[1].split('_')[3].split('li')[-1]
                if li_string.find('-')>-1:
                    ddict['li'] = int(li_string.split('-')[0])
                    ddict['num_laps'] = (int(li_string.split('-')[-1]) -
                                         int(li_string.split('-')[0]) + 1)
                else:
                    ddict['li'] = int(li_string)
                    ddict['num_laps'] = 1
                if ddict['has_dda'] == 0:
                    if strings[2] == 'reference_mpc':
                        ddict['mt'] = -1
                        ddict['st'] = 0
                        ddict['repetition'] = 0
                    else:
                        ddict['mt'] = -1
                        ddict['st'] = int(strings[2].split('_')[1].split('switch-')[-1])
                        ddict['repetition'] = int(strings[2].split('_')[-1])
                else:
                    if strings[0].find('dda_offline')>-1:
                        ddict['mt'] = -1
                        ddict['st'] = int(strings[2].split('_')[1].split('st-')[-1])
                        ddict['repetition'] = int(strings[2].split('_')[-1])
                    elif strings[2].find('mpc_eval_nw')>-1:
                        ddict['mt'] = -1
                        ddict['st'] = -1
                        ddict['repetition'] = 0
                    elif strings[2].find('mpc_nw_act')>-1:
                        ddict['mt'] = -1
                        ddict['st'] = -1
                        ddict['repetition'] = 0
                    else:
                        ddict['mt'] = int(strings[2].split('_')[1].split('mt-')[-1])
                        ddict['st'] = int(strings[2].split('_')[2].split('st-')[-1])
                        ddict['repetition'] = int(strings[2].split('_')[-1])
                # Add data dictionnairy as output row
                for k in sorted(ddict):
                    df[k] = ddict[k]
                performance = performance.append(df)
        # Save perfromance dataframe
        performance.to_csv(outfilepath, index=False)


# Make output table
if to_table:

    for collider_name in collider_names:
        curr_path = './performance/{}/'.format(collider_name)
        performance = pd.read_csv(curr_path + 'performance.csv')
        for comparison_name in ['all-combos',
                                'rvw-baseline',
                                'r-baseline',
                                'vw-baseline',
                                'fts-baseline']:
            for online_name in ['online',
                                'offline']:
                for trajectory_name in ['reference',
                                        'other-laps',
                                        'other-track',
                                        'multi-laps']:

                    print('----------------')
                    print(online_name, trajectory_name)
                    print('----------------')

                    # Subject dictionnairy
                    run_dict = None
                    exclude_run_dict = None
                    if trajectory_name == 'reference':
                        run_dict = {
                        'track': 'flat',
                        'subject': 16,
                        'run': 5,
                        'li': 1,
                        'num_laps': 1,
                        }
                    elif trajectory_name == 'other-laps':
                        run_dict = {
                            'track': 'flat',
                            'num_laps': 1,
                        }
                        exclude_run_dict = {
                            'track': 'flat',
                            'subject': 16,
                            'run': 5,
                            'li': 1,
                            'num_laps': 1,
                        }
                    elif trajectory_name == 'other-track':
                        run_dict = {
                            'track': 'wave',
                            'num_laps': 1,
                        }
                    elif trajectory_name == 'multi-laps':
                        run_dict = {
                            'track': 'flat',
                        }
                        exclude_run_dict = {
                            'num_laps': 1,
                        }

                    # Network general dictionnairy
                    if online_name=='online':
                        network_dict = {
                            'has_dda': 1,
                            'has_network_used': 1,
                        }
                    else:
                        network_dict = {
                            'has_dda': 1,
                            'has_network_used': 0,
                        }

                    # Model dictionnairy
                    if comparison_name == 'rvw-baseline':
                        model_names = [
                            'ref+rvw',
                            'ref+rvw+fts',
                            'ref+rvw+ain',
                            'ref+rvw+abr',
                            'ref',
                            'ref+fts',
                            'ref+ain',
                            'ref+abr',
                            'rvw',
                            'rvw+fts',
                            'rvw+ain',
                            'rvw+abr',
                        ]
                    elif comparison_name == 'r-baseline':
                        model_names = [
                            'ref+r',
                            'ref+r+fts',
                            'ref+r+ain',
                            'ref+r+abr',
                            'ref',
                            'ref+fts',
                            'ref+ain',
                            'ref+abr',
                            'r',
                            'r+fts',
                            'r+ain',
                            'r+abr',
                        ]
                    elif comparison_name == 'vw-baseline':
                        model_names = [
                            'ref+vw',
                            'ref+vw+fts',
                            'ref+vw+ain',
                            'ref+vw+abr',
                            'ref',
                            'ref+fts',
                            'ref+ain',
                            'ref+abr',
                            'vw',
                            'vw+fts',
                            'vw+ain',
                            'vw+abr',
                        ]
                    elif comparison_name == 'fts-baseline':
                        model_names = [
                            'ref+rvw+fts',
                            'ref+rvw+fts+ain',
                            'ref+rvw+fts+abr',
                            'ref+fts',
                            'ref+fts+ain',
                            'ref+fts+abr',
                            'rvw+fts',
                            'rvw+fts+ain',
                            'rvw+fts+abr',
                        ]
                    else:
                        model_names = [n for n in all_models_dict]

                    model_dicts = []
                    for n in model_names:
                        model_dicts.append({
                            'name': n,
                            'specs': all_models_dict[n].copy()
                        })

                    # Feature dictionnairy
                    if online_name=='online':
                        feature_dict={
                            'Flight Time [s]': {
                                'varname': 'flight_time',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': 'std',
                                'precision': 2
                                },
                            'Travel Distance [m]': {
                                'varname': 'travel_distance',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': 'std',
                                'precision': 2
                            },
                            'Mean Error [m]': {
                                'varname': 'median_path_deviation',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': 'std',
                                'precision': 2
                            },
                            'Gates Passed': {
                                'varname': 'num_gates_passed',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': 'std',
                                'precision': 2
                            },
                            '% Collision': {
                                'varname': 'num_collisions',
                                'track': '',
                                'first_line': 'percent',
                                'second_line': '',
                                'precision': 0
                            },
                        }
                    else:
                        feature_dict = {
                            'Throttle MSE': {
                                'varname': 'throttle_error_mse-median',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': '',
                                'precision': 3
                            },
                            'Throttle L1': {
                                'varname': 'throttle_error_l1-median',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': '',
                                'precision': 3
                            },
                            'Roll MSE': {
                                'varname': 'roll_error_mse-median',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': '',
                                'precision': 3
                            },
                            'Roll L1': {
                                'varname': 'roll_error_l1-median',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': '',
                                'precision': 3
                            },
                            'Pitch MSE': {
                                'varname': 'pitch_error_mse-median',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': '',
                                'precision': 3
                            },
                            'Pitch L1': {
                                'varname': 'pitch_error_l1-median',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': '',
                                'precision': 3
                            },
                            'Yaw MSE': {
                                'varname': 'yaw_error_mse-median',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': '',
                                'precision': 3
                            },
                            'Yaw L1': {
                                'varname': 'yaw_error_l1-median',
                                'track': '',
                                'first_line': 'mean',
                                'second_line': '',
                                'precision': 3
                            },
                        }

                    # Make a table
                    if (run_dict is not None) and (model_dicts is not None):
                        table = pd.DataFrame([])
                        for mdict in model_dicts:

                            #add subject dictionnairy to the model dictionnairy
                            mdict['specs'] = {**mdict['specs'],
                                              **network_dict,
                                              **run_dict}

                            ddict = {}
                            ddict['Model'] = [mdict['name'], '']

                            # Select performance data
                            # Runs to include
                            ind = np.array([True for i in range(performance.shape[0])])
                            for k, v in mdict['specs'].items():
                                ind = ind & (performance[k] == v)
                            # Runs to exclude
                            if exclude_run_dict is not None:
                                ind_exclude = np.array([True for i in range(
                                    performance.shape[0])])
                                for k, v in exclude_run_dict.items():
                                    ind_exclude = ind_exclude & (performance[k] == v)
                                # Combine run selection
                                ind = (ind == True) & (ind_exclude == False)
                            # Select current runs
                            curr_performance = performance.copy().loc[ind, :]

                            ddict['Num Runs'] = [str(curr_performance.shape[0]), '']

                            print(mdict['name'], ':', curr_performance.shape[0])
                            # print(curr_performance)

                            if curr_performance.shape[0] > 0:
                                # Compute Average performances
                                for outvar in feature_dict:
                                    ddict.setdefault(outvar, [])
                                    invar = feature_dict[outvar]['varname']
                                    curr_vals = curr_performance[invar].values
                                    #first line value
                                    op1 = feature_dict[outvar]['first_line']
                                    if op1 == 'mean':
                                        val1 = np.nanmean(curr_vals)
                                    elif op1 == 'percent':
                                        val1 = 100 * np.mean((curr_vals > 0).astype(int))
                                    else:
                                        val1 = None
                                    if val1 is None:
                                        ddict[outvar].append('')
                                    else:
                                        ddict[outvar].append(str(np.round(val1, feature_dict[outvar][
                                            'precision'])))
                                    # second line value
                                    op2 = feature_dict[outvar]['second_line']
                                    if op2 == 'std':
                                        val1 = np.nanstd(curr_vals)
                                    else:
                                        val1 = None
                                    if val1 is None:
                                        ddict[outvar].append('')
                                    else:
                                        ddict[outvar].append('('+str(np.round(val1, feature_dict[outvar][
                                            'precision']))+')')

                            for k in ddict:
                                ddict[k] = [' '.join(ddict[k])]

                            # Append two lines to the output table
                            table = table.append(
                                pd.DataFrame(ddict,
                                             index=list(range(len(ddict['Model']))))
                            )

                        outpath = curr_path + '/{}/{}/'.format(online_name,
                                                               comparison_name)
                        if not os.path.exists(outpath):
                            make_path(outpath)

                        outfilepath = outpath + 'latex_table_{}.csv'.format(
                            trajectory_name)
                        table.to_latex(outfilepath, index=False)


# Plot reference trajectory with gates
if to_plot_reference:
    # Load track.
    track = pd.read_csv(track_filepaths['flat'])
    ndict = {
        'pos_x': 'px',
        'pos_y': 'py',
        'pos_z': 'pz',
        'rot_x_quat': 'qx',
        'rot_y_quat': 'qy',
        'rot_z_quat': 'qz',
        'rot_w_quat': 'qw',
        'dim_x': 'dx',
        'dim_y': 'dy',
        'dim_z': 'dz',
    }
    track = track.rename(columns=ndict)
    track = track[list(ndict.values())]
    track['pz'] += 0.35
    track['dx'] = 0.
    track['dy'] = 3
    track['dz'] = 3
    # Load reference and downsample to 20 Hz
    reference = trajectory_from_logfile(
        filepath=reference_filepath)
    sr = 1 / np.nanmedian(np.diff(reference.t.values))
    reference = reference.iloc[np.arange(0, reference.shape[0], int(sr / 20)), :]
    # Plot reference, track, and format figure.
    ax = plot_trajectory(
        reference.px.values,
        reference.py.values,
        reference.pz.values,
        reference.qx.values,
        reference.qy.values,
        reference.qz.values,
        reference.qw.values,
        axis_length=2,
        c='k',
    )
    ax = plot_gates_3d(
        track=track,
        ax=ax,
        color='k',
        width=4,
        )
    ax = format_trajectory_figure(
        ax=ax,
        xlims=(-15, 19),
        ylims=(-17, 17),
        zlims=(-8, 8),
        xlabel='px [m]',
        ylabel='py [m]',
        zlabel='pz [m]',
        title='',
        )

    plt.axis('off')
    plt.grid(b=None)
    ax.view_init(elev=45,
                 azim=270)
    plt.gcf().set_size_inches(20,10)

    plot_path = './plots/'
    if not os.path.exists(plot_path):
        make_path(plot_path)

    plt.savefig(plot_path + 'reference_3d.jpg')


# Plot reference trajecory with decision colored
if to_plot_reference_with_decision:
    data_path = './branching_demo/'
    # Load track.
    track = pd.read_csv(data_path+'flat.csv')
    ndict = {
        'pos_x': 'px',
        'pos_y': 'py',
        'pos_z': 'pz',
        'rot_x_quat': 'qx',
        'rot_y_quat': 'qy',
        'rot_z_quat': 'qz',
        'rot_w_quat': 'qw',
        'dim_x': 'dx',
        'dim_y': 'dy',
        'dim_z': 'dz',
    }
    track = track.rename(columns=ndict)
    track = track[list(ndict.values())]
    track['pz'] += 0.35
    track['dx'] = 0.
    track['dy'] = 3
    track['dz'] = 3
    # Load reference and downsample to 20 Hz
    reference = trajectory_from_logfile(
        filepath=data_path+'trajectory_reference_original.csv')
    sr = 1 / np.nanmedian(np.diff(reference.t.values))
    reference = reference.iloc[np.arange(0, reference.shape[0], int(sr / 20)),
                :]
    # Plot reference, track, and format figure.
    ax = plot_trajectory(
        reference.px.values,
        reference.py.values,
        reference.pz.values,
        reference.qx.values,
        reference.qy.values,
        reference.qz.values,
        reference.qw.values,
        axis_length=2,
        c='b',
        axis_colors='b',
    )
    ax = plot_gates_3d(
        track=track,
        ax=ax,
        color='k',
        width=4,
    )
    ax = format_trajectory_figure(
        ax=ax,
        xlims=(-15, 19),
        ylims=(-17, 17),
        zlims=(-8, 8),
        xlabel='px [m]',
        ylabel='py [m]',
        zlabel='pz [m]',
        title='',
    )

    plt.axis('off')
    plt.grid(b=None)
    ax.view_init(elev=45,
                 azim=270)
    plt.gcf().set_size_inches(20, 10)

    plot_path = './plots/'
    if not os.path.exists(plot_path):
        make_path(plot_path)

    plt.savefig(plot_path + 'reference_flat_with_decision.jpg')