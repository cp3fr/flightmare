import os
import sys

# Add base path
base_path = os.getcwd().split('/flightmare/')[0]+'/flightmare/'
sys.path.insert(0, base_path)

from analysis.utils import *

# What to do
to_process = True
to_performance = False

to_plot_traj_3d = False
to_plot_state = False
to_plot_reference = False

# Buffer time: time runs with  network were started earlier than the reference
# trajectory
buffer = 2.0

# Chose input files and folders for processing.
track_filepaths = {
    'flat': './tracks/flat.csv',
    'wave': './tracks/wave.csv',
    }
logfile_path = './logs/'
reference_filepath = (
        logfile_path +
        'resnet_test/trajectory_reference_original.csv')
models = [
    # 'dda_flat_med_full_bf2_cf25_decfts',
        ]
if len(models) == 0:
    for w in os.walk(logfile_path):
        if w[0] == logfile_path:
            models = w[1]
models = sorted(models)

# Process individual runs
if to_process:

    for model in models:

        # Make a list of input logfiles of network trajectories.
        log_filepaths = []
        for w in os.walk(os.path.join(logfile_path, model)):
            for f in w[2]:
                if (f.find('.csv') > 0) and (reference_filepath.find(f) < 0):
                    log_filepaths.append(os.path.join(w[0], f))

        # Process individual trajectories flown by the network.
        for filepath in sorted(log_filepaths):
            print('..processing {}'.format(filepath))
            # Make output folder
            data_path = (filepath
                         .replace('.csv', '/')
                         .replace('/logs/', '/process/')
                         .replace('trajectory_', '')
                         )
            make_path(data_path)
            # Copy trajectory, reference, and track files to output folder
            if not os.path.isfile(data_path + 'trajectory.csv'):
                trajectory = trajectory_from_logfile(filepath=filepath)
                # Important: compensate for buffer time (i.e. earlier start
                # of the network, before start of the reference trajectory)
                trajectory['t'] -= buffer
                trajectory.to_csv(data_path + 'trajectory.csv',
                                  index=False)
            if not os.path.isfile(data_path + 'reference.csv'):
                reference = trajectory_from_logfile(filepath=reference_filepath)
                reference.to_csv(data_path + 'reference.csv',
                                  index=False)
            if not os.path.isfile(data_path + 'track.csv'):
                if filepath.find('wave') > -1:
                    track_filepath = track_filepaths['wave']
                else:
                    track_filepath = track_filepaths['flat']

                track = track_from_logfile(filepath=track_filepath)
                # Make some adjustments
                track['pz'] += 0.35 # shift gates up in fligthmare
                track['dx'] = 0.
                track['dy'] = 3 # in the middle of inner diameter 2.5 and outer 3.0
                track['dz'] = 3 # in the middle of inner diameter 2.5 and outer 3.0
                track.to_csv(data_path + 'track.csv',
                             index=False)
            # Save gate pass and collision events to output folder
            if not os.path.isfile(data_path + 'events.csv'):
                E = get_pass_collision_events(
                    filepath_trajectory=data_path+'trajectory.csv',
                    filepath_track=data_path+'track.csv')
                E.to_csv(data_path + 'events.csv', index=False)
            # Save performance features to output folder
            if not os.path.isfile(data_path + 'features.csv'):
                P = extract_performance_features(
                    filepath_trajectory=data_path + 'trajectory.csv',
                    filepath_reference=data_path + 'reference.csv',
                    filepath_events=data_path + 'events.csv')
                P.to_csv(data_path + 'features.csv', index=False)
            # Save trajectory plot to output folder
            if to_plot_traj_3d:
                track = pd.read_csv(data_path + 'track.csv')
                trajectory = pd.read_csv(data_path + 'trajectory.csv')
                features = pd.read_csv(data_path + 'features.csv')
                for label in ['', 'valid_']:
                    for view, xlims, ylims, zlims in [
                                 [(45, 270), (-15, 19), (-17, 17), (-8, 8)],
                                 # [(0, 270), (-15, 19), (-17, 17), (-12, 12)],
                                 # [(0, 180), (-15, 19), (-17, 17), (-12, 12)],
                                 # [(90, 270), (-15, 19), (-15, 15), (-12, 12)],
                            ]:
                        outpath = (data_path + '{}trajectory_with_gates_'
                                               '{}x{}.jpg'
                                   .format(label,
                                           '%03d' % view[0],
                                           '%03d' % view[1])
                                   )
                        if not os.path.isfile(outpath):
                            if label == 'valid_':
                                ind = ((trajectory['t'].values >=
                                       features['t_start'].iloc[0]) &
                                       (trajectory['t'].values <=
                                        features['t_end'].iloc[0]))
                            else:
                                ind = np.array([True for i in range(
                                    trajectory.shape[0])])
                            ax = plot_trajectory_with_gates_3d(
                                trajectory=trajectory.iloc[ind, :],
                                track=track,
                                view=view,
                                xlims=xlims,
                                ylims=ylims,
                                zlims=zlims,
                            )

                            ax.set_title(outpath)
                            plt.savefig(outpath)
                            plt.close(plt.gcf())
                            ax=None
            # Plot the drone state
            if to_plot_state:
                if not os.path.isfile(data_path + 'state.jpg'):
                    plot_state(
                        filepath_trajectory=data_path + 'trajectory.csv',
                        filepath_reference=data_path + 'reference.csv',
                        filepath_features=data_path + 'features.csv',
                        )
                    plt.savefig(data_path + 'state.jpg')
                    plt.close(plt.gcf())


if to_performance:

    # Copy trajectory plots into the plot folder
    if to_plot_traj_3d:
        for model in models:
            outpath = './plots/trajectories/{}/'.format(model)
            if not os.path.exists(outpath):
                make_path(outpath)
            for w in os.walk('./process/{}/'.format(model)):
                for f in w[2]:
                    if f.find('trajectory_with_gates') > -1:
                        infile_path = os.path.join(w[0], f)
                        outfile_path = outpath + infile_path.replace(
                            '/trajectory_with_gates_', '_').split('/')[-1]
                        print('..copying trajectories {}'.format(outfile_path))
                        copyfile(infile_path,
                                 outfile_path)

    # Collect summaries across models and runs
    outpath = './performance/'
    outfilepath = outpath + 'performance.csv'
    if not os.path.isfile(outfilepath):
        if not os.path.exists(outpath):
            make_path(outpath)
        performance = pd.DataFrame([])
        for model in models:
            filepaths = []
            for w in os.walk('./process/'+model+'/'):
                for f in w[2]:
                    if f=='features.csv':
                        filepaths.append(os.path.join(w[0], f))
            for filepath in filepaths:
                print('..collecting performance: {}'.format(filepath), end='\r')
                performance = performance.append(pd.read_csv(filepath))
        performance.to_csv(outfilepath, index=False)




    # Add some more information to performance table
    # curr_path = df['filepath'].iloc[0]
    # # add track information
    # if curr_path.find('wave') > -1:
    #     df['track'] = 'wave'
    # else:
    #     if curr_path.find('flat') > -1:
    #         df['track'] = 'flat'
    #     else:
    #         df['track'] = 'NONE'
    #
    #
    #
    # #add subject number
    # curr_subj = int(
    #     curr_path.split('/s0')[-1].split('_')[0])
    # df['subject'] = curr_subj
    # # add run number
    # curr_run = int(
    #     curr_path.split('_r')[-1].split('_')[0])
    # df['run'] = curr_run
    # #add repetition number
    # curr_repetition = int(
    #     curr_path.split('/')[-2].split('_')[-1])
    # df['repetition'] = curr_repetition



    # # Make Performance summary tables for latex.
    # for model in models:
    #     inpath = './performance/'+model+'/summary.csv'
    #     summary = pd.read_csv(inpath)
    #
    #     odict={
    #         'Flight Time [s]': {
    #             'varname': 'flight_time',
    #             'track': '',
    #             'first_line': 'mean',
    #             'second_line': 'std',
    #             'precision': 2
    #             },
    #         'Travel Distance [m]': {
    #             'varname': 'travel_distance',
    #             'track': '',
    #             'first_line': 'mean',
    #             'second_line': 'std',
    #             'precision': 2
    #         },
    #         'Mean Error [m]': {
    #             'varname': 'median_path_deviation',
    #             'track': '',
    #             'first_line': 'mean',
    #             'second_line': 'std',
    #             'precision': 2
    #         },
    #         'Num. Gates Passed': {
    #             'varname': 'num_gates_passed',
    #             'track': '',
    #             'first_line': 'mean',
    #             'second_line': 'std',
    #             'precision': 2
    #         },
    #         '% Collision Free': {
    #             'varname': 'num_collisions',
    #             'track': '',
    #             'first_line': 'percent',
    #             'second_line': '',
    #             'precision': 0
    #         },
    #     }
    #
    #     ddict = {}
    #     ddict['Model'] = [model, '']
    #     for outvar in odict:
    #         ddict.setdefault(outvar, [])
    #         invar = odict[outvar]['varname']
    #         curr_vals = summary[invar].values
    #         #first line value
    #         op1 = odict[outvar]['first_line']
    #         if op1 == 'mean':
    #             val1 = np.nanmean(curr_vals)
    #         elif op1 == 'percent':
    #             val1 = 100 * np.mean((curr_vals > 0).astype(int))
    #         else:
    #             val1 = None
    #         if val1 is None:
    #             ddict[outvar].append('')
    #         else:
    #             ddict[outvar].append(str(np.round(val1, odict[outvar][
    #                 'precision'])))
    #         # second line value
    #         op2 = odict[outvar]['second_line']
    #         if op2 == 'std':
    #             val1 = np.nanstd(curr_vals)
    #         else:
    #             val1 = None
    #         if val1 is None:
    #             ddict[outvar].append('')
    #         else:
    #             ddict[outvar].append('('+str(np.round(val1, odict[outvar][
    #                 'precision']))+')')
    #
    #
    #
    #
    #
    #     # #todo: make the output drag and drop for latex
    #     # precision = 2
    #     # for op in ['mean', 'sd']:
    #     #     for name in names:
    #     #         tdict.setdefault(name, [])
    #     #         if op=='mean':
    #     #             tdict[name].append(
    #     #                 ('{:.%df}'%precision)
    #     #                     .format(
    #     #                     np.nanmean(summary[name].values)))
    #     #         elif op=='sd':
    #     #             tdict[name].append(
    #     #                 ('({:.%df})' % precision)
    #     #                     .format(
    #     #                     np.nanstd(summary[name].values)))
    #
    #     table = pd.DataFrame(ddict, index=list(range(len(ddict['Model']))))
    #     outpath = './performance/'+model+'/table.csv'
    #     table.to_latex(outpath, index=False)
    #




if to_plot_reference:
    # Load track.
    track = pd.read_csv(track_filepath)
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

    # # Plot flight path overlay of individual runs
    # for model in models:
    #   data_path = './process/' + model + '/'
    #   compare_trajectories_3d(
    #       reference_filepath=reference_filepath,
    #       data_path=data_path,
    #       )

    plt.show()