try:
    from analysis.functions.visattention import *
except:
    from functions.visattention import *

"""
Processing pipeline for visualizing and quantifying network performance as 
compared to some reference (human pilot or MPC).
"""

# Chose input files and folders for processing.
filepath_track = './tracks/flat.csv'
filepath_reference = './logs/resnet_test/trajectory_reference_original.csv'
folderpath = './logs/'

# Make a list of input logfiles of network trajectories.
filepaths = []
for w in os.walk(folderpath):
    for f in w[2]:
        if (f.find('.csv') > 0) and (filepath_reference.find(f) < 0):
            filepaths.append(os.path.join(w[0], f))

# Process individual trajectories flown by the network.
for filepath in filepaths:
    print('..processing {}'.format(filepath))
    # Make output folder
    outpath = filepath.replace('.csv',
                               '/').replace('/logs/',
                                            '/process/').replace('trajectory_',
                                                                 '')
    make_path(outpath)
    # Copy track, trajectory, reference files to output folder
    copyfile(filepath,
             outpath + 'trajectory.csv')
    copyfile(filepath_reference,
             outpath + 'reference.csv')
    copyfile(filepath_track,
             outpath + 'track.csv')
    # Save gate pass and collision events to output folder
    E = get_pass_collision_events(
        filepath_trajectory=filepath,
        filepath_track=filepath_track)
    E.to_csv(outpath + 'events.csv', index=False)

    # Save performance features to output folder
    P = extract_performance_features(
        filepath_trajectory=outpath + 'trajectory.csv',
        filepath_reference=outpath + 'reference.csv',
        filepath_events=outpath + 'events.csv')
    P.to_csv(outpath + 'features.csv', index=False)

    # Todo: Save plot of drone state comparison trajectory vs reference

    # Todo: Save plot trajectory with poses

    # Todo: Optional: Save Animation


# Todo: Collect features across muliple repetitions
