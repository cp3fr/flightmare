try:
    from analysis.functions.visattention import *
except:
    from functions.visattention import *

# Select Input files and folders for processing
filepath_track = './tracks/flat.csv'
filepath_reference = './logs/resnet_test/trajectory_reference_original.csv'
folderpath = './logs/'

# Collect a list of trajectory logfile paths from a given folderpath.
filepaths = []
for w in os.walk(folderpath):
    for f in w[2]:
        if (f.find('.csv') > 0) and (filepath_reference.find(f) < 0):
            filepaths.append(os.path.join(w[0], f))

# Extract trajectory, reference, and events from given logfile path.
for filepath in filepaths:
    print('..processing {}'.format(filepath))
    # Make output path
    outpath = filepath.replace(
        '.csv', '/').replace(
        '/logs/', '/process/').replace(
        'trajectory_', ''
    )
    make_path(outpath)
    # Copy track, trajectory, reference to output path
    copyfile(filepath,
             outpath + 'trajectory.csv')
    copyfile(filepath_reference,
             outpath + 'reference.csv')
    copyfile(filepath_track,
             outpath + 'track.csv')
    # Save pass and collision events to output path
    E = get_pass_collision_events(filepath_trajectory=filepath,
                                  filepath_track=filepath_track)
    E.to_csv(outpath + 'events.csv', index=False)

    # Todo: Compute performance features

    # Todo: Plot drone state comparison to reference

    # Todo: Plot trajectory with poses

    # Todo: Optional: Save an animation


# Todo: Collect features across muliple repetitions
