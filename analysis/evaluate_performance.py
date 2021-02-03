try:
    from analysis.functions.processing_utils import *
    from analysis.functions.Animation3D import *
    from analysis.functions.Gate import *
except:
    from functions.processing_utils import *
    from functions.Animation3D import *
    from functions.Gate import *


PATH = './logs/trajectory_mpc2nw_switch-06_00.csv'
PATH = './logs/trajectory_reference_mpc.csv'
PATH = './logs/trajectory_reference_original.csv'

#read in the data
d = pd.read_csv(PATH)
print(d.columns)
#gate poses for the current track
t = pd.read_csv('./tracks/flat.csv')
#gate passing surfaces
objGatePass = [Gate(t.iloc[i], dtype='gazesim') for i in range(t.shape[0])]
#gate collision surfaces
objGateCollider = [Gate(t.iloc[i], dtype='gazesim', dims=(3.5, 3.5)) for i in range(t.shape[0])]
#wall collision surfaces
def getWallColliders(dims=(1, 1, 1), center=(0, 0, 0)):
    '''getting 3d volume wall collider objects
    dims: x, y, z dimensions in meters
    denter: x, y, z positions of the 3d volume center'''
    objWallCollider = []

    _q = (Rotation.from_euler('y', [np.pi/2]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Gate(pd.DataFrame({'pos_x': center[0], 'pos_y': center[1], 'pos_z' : center[2] - dims[2] / 2,
                                            'rot_x_quat': _q[0], 'rot_y_quat':_q[1], 'rot_z_quat':_q[2], 'rot_w_quat':_q[3],
                                            'dim_x':0, 'dim_y': dims[1], 'dim_z':dims[0]}, index=[0]).iloc[0],
                                dims=(dims[1], dims[0]), dtype='gazesim'))

    _q = (Rotation.from_euler('y', [-np.pi/2]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Gate(pd.DataFrame({'pos_x': center[0], 'pos_y': center[1], 'pos_z' : center[2] + dims[2] / 2,
                                            'rot_x_quat': _q[0], 'rot_y_quat':_q[1], 'rot_z_quat':_q[2], 'rot_w_quat':_q[3],
                                            'dim_x':0, 'dim_y':dims[1], 'dim_z':dims[0]}, index=[0]).iloc[0],
                                dims=(dims[1], dims[0]), dtype='gazesim'))

    _q = np.array([0, 0, 0, 1])
    objWallCollider.append(Gate(pd.DataFrame({'pos_x': center[0] + dims[0] / 2, 'pos_y': center[1], 'pos_z' : center[2],
                                            'rot_x_quat': _q[0], 'rot_y_quat':_q[1], 'rot_z_quat':_q[2], 'rot_w_quat':_q[3],
                                            'dim_x':0, 'dim_y':dims[1], 'dim_z':dims[2]}, index=[0]).iloc[0],
                                dims=(dims[1], dims[2]), dtype='gazesim'))

    _q = (Rotation.from_euler('z', [np.pi]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Gate(pd.DataFrame({'pos_x': center[0] - dims[0] / 2, 'pos_y': center[1], 'pos_z' : center[2],
                                            'rot_x_quat': _q[0], 'rot_y_quat':_q[1], 'rot_z_quat':_q[2], 'rot_w_quat':_q[3],
                                            'dim_x':0, 'dim_y':dims[1], 'dim_z':dims[2]}, index=[0]).iloc[0],
                                dims=(dims[1], dims[2]), dtype='gazesim'))

    _q = (Rotation.from_euler('z', [np.pi/2]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Gate(pd.DataFrame({'pos_x': center[0], 'pos_y': center[1] + dims[1] / 2, 'pos_z' : center[2],
                                            'rot_x_quat': _q[0], 'rot_y_quat':_q[1], 'rot_z_quat':_q[2], 'rot_w_quat':_q[3],
                                            'dim_x':0, 'dim_y':dims[1], 'dim_z':dims[2]}, index=[0]).iloc[0],
                                dims=(dims[0], dims[2]), dtype='gazesim'))

    _q = (Rotation.from_euler('z', [-np.pi/2]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Gate(pd.DataFrame({'pos_x': center[0], 'pos_y': center[1] - dims[1] / 2, 'pos_z' : center[2],
                                            'rot_x_quat': _q[0], 'rot_y_quat':_q[1], 'rot_z_quat':_q[2], 'rot_w_quat':_q[3],
                                            'dim_x':0, 'dim_y':dims[1], 'dim_z':dims[2]}, index=[0]).iloc[0],
                                dims=(dims[0], dims[2]), dtype='gazesim'))
    return objWallCollider
objWallCollider = getWallColliders(dims=(66, 36, 9), center=(0, 0, 4.5))
#todo: check for gate passes and collision, return event timestamps


#todo: compute metrics (number of gates passed, travel distance, valid fligth time, deviation from reference trajectory)


toAnimateTrajectory = True
if toAnimateTrajectory:
    gate_objects = objGatePass + objGateCollider + objWallCollider
    d['simulation-time-since-start [s]'] = d['time-since-start [s]'].values
    anim = Animation3D(d, Gate_objects=gate_objects, equal_lims=(-30, 30))
    anim.show()