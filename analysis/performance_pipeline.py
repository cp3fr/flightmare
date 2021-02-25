try:
    from analysis.functions.visattention import *
except:
    from functions.visattention import *

"""
Processing pipeline for visualizing and quantifying network performance as 
compared to some reference (human pilot or MPC).
"""

# Settings
to_collect = True
to_summary = True
to_plot = False

# Chose input files and folders for processing.
track_filepath = './tracks/flat.csv'
logfile_path = './logs/'
reference_filepath = (
        logfile_path +
        'resnet_test/trajectory_reference_original.csv')
models = [
    'dda_flat_med_full_bf2_cf25_noimuai_ep80',
    'dda_flat_med_full_bf2_cf25_refonly_ep100',
    'dda_flat_med_full_bf2_cf25_noimu_ep100',
    'dda_flat_med_full_bf2_cf25_decfts_ep45',
    'dda_flat_med_full_bf2_cf25_decfts_ep100',
    'dda_0',
    'resnet_test',
        ]


if to_collect:

    for model in models:
        # Make a list of input logfiles of network trajectories.
        log_filepaths = []
        for w in os.walk(os.path.join(logfile_path, model)):
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
                trajectory = trajectory_from_logfile(filepath=filepath)
                trajectory.to_csv(data_path + 'trajectory.csv',
                                  index=False)
            if not os.path.isfile(data_path + 'reference.csv'):
                reference = trajectory_from_logfile(filepath=reference_filepath)
                reference.to_csv(data_path + 'reference.csv',
                                  index=False)
            if not os.path.isfile(data_path + 'track.csv'):
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
            if not os.path.isfile(data_path + 'trajectory_with_gates.jpg'):
                track = pd.read_csv(data_path + 'track.csv')
                trajectory = pd.read_csv(data_path + 'trajectory.csv')
                for view, xlims, ylims, zlims in [
                             [(45, 270), (-15, 19), (-17, 17), (-8, 8)],
                             [(0, 270), (-15, 19), (-17, 17), (-12, 12)],
                             [(0, 180), (-15, 19), (-17, 17), (-12, 12)],
                             [(90, 270), (-15, 19), (-15, 15), (-12, 12)],
                        ]:
                    ax = plot_trajectory_with_gates_3d(
                        trajectory=trajectory,
                        track=track,
                        view=view,
                        xlims=xlims,
                        ylims=ylims,
                        zlims=zlims,
                    )
                    outpath = (data_path + 'trajectory_with_gates_{}x{'
                                           '}.jpg'.format('%03d'%view[0],
                                                            '%03d'%view[1]))
                    ax.set_title(outpath)
                    plt.savefig(outpath)
                    plt.close(plt.gcf())
                    ax=None


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
        precision = 2
        for op in ['mean', 'sd']:
            for name in names:
                tdict.setdefault(name, [])
                if op=='mean':
                    tdict[name].append(
                        ('{:.%df}'%precision)
                            .format(
                            np.nanmean(summary[name].values)))
                elif op=='sd':
                    tdict[name].append(
                        ('({:.%df})' % precision)
                            .format(
                            np.nanstd(summary[name].values)))

        table = pd.DataFrame(tdict, index=[v for v in range(len(tdict[names[
            0]]))])
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