try:
    from analysis.functions.visattention import *
except:
    from functions.visattention import *


# Load race track information.
filepath = './tracks/flat.csv'
T = pd.read_csv(filepath)
zOffset = 0.35
T['pos_z'] += zOffset

# Define checkpoints and colliders.
gate_checkpoints = [Gate(T.iloc[i], dims=(2.5, 2.5)) for i in range(T.shape[0])]
gate_colliders = [Gate(T.iloc[i], dims=(3.5, 3.5)) for i in range(T.shape[0])]
wall_colliders = get_wall_colliders(dims=(66, 36, 9), center=(0, 0, 4.5))

# Load a trajectory
filepath = './logs/resnet_test/trajectory_reference_original.csv'
D = trajectory_from_logfile(filepath)
t = D['t'].values
px = D['px'].values
py = D['py'].values
pz = D['pz'].values

# Detect checkpoint passing and collision events.
events = {}
for key, objects in [
        ('gate_passing', gate_checkpoints),
        ('gate_collision', gate_colliders),
        ('wall_collision', wall_colliders)
        ]:
    for id in range(len(objects)):
        object = objects[id]
        for timestamp in detect_gate_passing(t, px, py, pz, object):
            if not ((key == 'gate_collision') and (timestamp in events.keys())):
                events[timestamp] = (key, id)

for k in sorted(events):
    print(k, events[k])


# Todo: save events to logfile

# Todo: extract performance features for current trajectory

# Todo: collect performance features across multiple trajectories

# Todo: plot and save performance feature summary


