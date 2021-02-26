try:
    from analysis.functions.visattention import *
except:
    from functions.visattention import *
import seaborn as sns


filepath_trajectory = './process/dda_flat_med_full_bf2_cf25_fts_decfts_ep80' \
                      '/mpc2nw_mt-152_st-20_00/reference.csv'
filepath_track = './process/dda_flat_med_full_bf2_cf25_fts_decfts_ep80' \
                      '/mpc2nw_mt-152_st-20_00/track.csv'

track = pd.read_csv(filepath_track)
trajectory = pd.read_csv(filepath_trajectory)

# Drone states
t = trajectory['t'].values
p = trajectory.loc[:, ('px', 'py', 'pz')].values
q = trajectory.loc[:, ('qx', 'qy', 'qz', 'qw')].values
# plt.figure()
# plt.plot(p[:, 0], p[:, 1])

plt.figure()

# Upcoming gates
upcoming_gates = np.empty((t.shape[0], 3))
upcoming_gates[:] = np.nan
events = get_pass_collision_events(
    filepath_trajectory=filepath_trajectory,
    filepath_track=filepath_track,
)

print(events)
for i in range(1, track.shape[0]):
    t_start = events['t'].iloc[i-1]
    t_end = events['t'].iloc[i]
    past_gate = events['object-id'].iloc[i-1]
    ind = (t>t_start) & (t<=t_end)
    upcoming_gates[ind, :] = past_gate + np.array([[1, 2, 3]])
for i in range(upcoming_gates.shape[1]):
    ind = upcoming_gates[:, i] > 9
    upcoming_gates[ind, i] -= 9

# Distances to all gates
plt.subplot(311)
distance_all = np.empty((t.shape[0], track.shape[0]))
distance_all[:] = np.nan
for i in range(track.shape[0]):
    p_gate = track.loc[:, ('px', 'py', 'pz')].iloc[i].values
    q_gate = track.loc[:, ('qx', 'qy', 'qz', 'qw')].iloc[i].values
    dp = p - p_gate
    dpn = np.linalg.norm(dp, axis=1)
    distance_all[:, i] = dpn
plt.plot(t, distance_all)
plt.xlabel('Time [s]')
plt.ylabel('Distance to Gate [m]')
plt.legend(['Gate 0', 'Gate 1', 'Gate 2', 'Gate 3', 'Gate 4', 'Gate 5',
            'Gate 6', 'Gate 7', 'Gate 8', 'Gate 9'], loc='upper right')

# Distances to next three gates
plt.subplot(312)
distance_upcoming = np.empty(upcoming_gates.shape)
distance_upcoming[:] = np.nan
for i in range(upcoming_gates.shape[1]):
    ind = np.isnan(upcoming_gates[:, i])==False
    indices = upcoming_gates[ind, i]
    p_gate = track.loc[:, ('px', 'py', 'pz')].iloc[indices, :].values
    q_gate = track.loc[:, ('qx', 'qy', 'qz', 'qw')].iloc[indices, :].values
    p_quad = p[ind, :]
    dp = p_quad - p_gate
    dpn = np.linalg.norm(dp, axis=1)
    distance_upcoming[ind, i] = dpn
plt.plot(t, distance_upcoming)
plt.xlabel('Time [s]')
plt.ylabel('Distance to Upcoming Gate [m]')
plt.legend(['+1 gate', '+2 gates', '+3 gates'], loc='upper right')

# Horizontal angle to next three gates
plt.subplot(313)
angle_upcoming = np.empty(upcoming_gates.shape)
angle_upcoming[:] = np.nan
for i in range(upcoming_gates.shape[1]):
    ind = np.isnan(upcoming_gates[:, i])==False
    indices = upcoming_gates[ind, i]
    p_gate = track.loc[:, ('px', 'py', 'pz')].iloc[indices, :].values
    q_gate = track.loc[:, ('qx', 'qy', 'qz', 'qw')].iloc[indices, :].values
    p_quad = p[ind, :]
    q_quad = q[ind, :]
    q_cam = (Rotation.from_euler('y', -30, degrees=True)
             * Rotation.from_quat(q_quad)).as_quat()
    p_cam = Rotation.from_quat(q_cam).apply(np.array([[10, 0, 0]])) + p_quad

    reference_vector = p_gate - p_quad
    target_vector = p_cam - p_quad

    angle = signed_horizontal_angle(
        reference_vector=reference_vector,
        target_vector=target_vector,
    )

    angle_upcoming[ind, i] = angle

plt.plot(t, angle_upcoming)
plt.xlabel('Time [s]')
plt.ylabel('Signed horizontal angle quad-gate [rad]')
plt.legend(['+1 gate', '+2 gates', '+3 gates'], loc='upper right')

# Pairplot across features
ind = np.isnan(upcoming_gates[:, 0])==False

df = pd.DataFrame(
    {
        'gate': upcoming_gates[ind, 0].astype(int),
        'd1': distance_upcoming[ind, 0],
        'd2': distance_upcoming[ind, 1],
        'd3': distance_upcoming[ind, 2],
        'a1': angle_upcoming[ind, 0],
        'a2': angle_upcoming[ind, 1],
        'a3': angle_upcoming[ind, 2],
    }, index=list(range(np.sum(ind)))
)
sns.pairplot(df, hue='gate')
plt.show()