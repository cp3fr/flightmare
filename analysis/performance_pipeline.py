try:
    from analysis.functions.visattention import *
except:
    from functions.visattention import *

"""
Processing pipeline for visualizing and quantifying network performance as 
compared to some reference (human pilot or MPC).
"""

# Settings
to_collect = False
to_summary = True
to_plot = False

# Chose input files and folders for processing.
track_filepath = './tracks/flat.csv'
logfile_path = './logs/'
reference_filepath = (
        logfile_path +
        'resnet_test/trajectory_reference_original.csv')
models = [
    'dda_flat_med_full_bf2_cf25_decfts_ep45',
    'dda_flat_med_full_bf2_cf25_decfts_ep100'
        ]


if to_collect:

    # Make a list of input logfiles of network trajectories.
    log_filepaths = []
    for w in os.walk(logfile_path):
        for f in w[2]:
            if (f.find('.csv') > 0) and (reference_filepath.find(f) < 0):
                log_filepaths.append(os.path.join(w[0], f))

    # Process individual trajectories flown by the network.
    for filepath in log_filepaths:
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
            copyfile(filepath,
                     data_path + 'trajectory.csv')
        if not os.path.isfile(data_path + 'reference.csv'):
            copyfile(reference_filepath,
                     data_path + 'reference.csv')
        if not os.path.isfile(data_path + 'track.csv'):
            copyfile(track_filepath,
                     data_path + 'track.csv')
        # Save gate pass and collision events to output folder
        if not os.path.isfile(data_path + 'events.csv'):
            E = get_pass_collision_events(
                filepath_trajectory=filepath,
                filepath_track=track_filepath)
            E.to_csv(data_path + 'events.csv', index=False)
        # Save performance features to output folder
        if not os.path.isfile(data_path + 'features.csv'):
            P = extract_performance_features(
                filepath_trajectory=data_path + 'trajectory.csv',
                filepath_reference=data_path + 'reference.csv',
                filepath_events=data_path + 'events.csv')
            P.to_csv(data_path + 'features.csv', index=False)

        # Todo: add comparison to MPC control commands

        # Todo: Save plot of drone state comparison trajectory vs reference

        # Todo: Save plot trajectory with poses

        # Todo: Optional: Save Animation


# Todo: Make performance summary table across muliple repetitions
if to_summary:

    # Collect summaries across runs
    for model in models:
        outpath = './performance/' + model + '/'
        outfilepath = outpath + 'summary.csv'
        if not os.path.isfile(outfilepath):
            filepaths = []
            for w in os.walk('./process/'+model+'/'):
                for f in w[2]:
                    if f == 'features.csv':
                        filepaths.append(os.path.join(w[0], f))
            summary = pd.DataFrame([])
            for filepath in sorted(filepaths):
                print('collecting performance summary: {}'.format(
                    filepath
                    ))
                df = pd.read_csv(filepath)
                summary = summary.append(df)

            if not os.path.exists(outpath):
                make_path(outpath)
            summary.to_csv(
                outfilepath,
                index=False)

    #todo: Plot path deviation across time.

    # Make tables.
    for model in models:
        inpath = './performance/'+model+'/summary.csv'
        summary = pd.read_csv(inpath)
        names = [
            'flight_time',
            'travel_distance',
            'median_path_deviation',
            'iqr_path_deviation',
            'num_gates_passed',
            'num_collisions'
        ]
        tdict = {
            'model': model,
        }

        #todo: make the output drag and drop for latex
        for name in names:
            tdict[name+'_mean'] = np.nanmean(
                summary[name].values)
            tdict[name + '_sd'] = np.nanstd(
                summary[name].values)
        table = pd.DataFrame(tdict, index=[0])
        outpath = './performance/'+model+'/table.csv'
        table.to_csv(outpath, index=False)





if to_plot:
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