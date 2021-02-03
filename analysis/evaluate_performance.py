try:
    from analysis.functions.preprocessing_utils import *
    from analysis.functions.processing_utils import *
    from analysis.functions.Animation3D import *
    from analysis.functions.Gate import *
    from analysis.functions.laptracker_utils import *
except:
    from functions.preprocessing_utils import *
    from functions.processing_utils import *
    from functions.Animation3D import *
    from functions.Gate import *
    from functions.laptracker_utils import *

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

def pipeline(PATH, toSaveAnimation=False):
    print(PATH)
    print('')
    #read drone state logs
    d = pd.read_csv(PATH)
    #read gate poses for the current track
    t = pd.read_csv('./tracks/flat.csv')
    #make gate passing surfaces
    objGatePass = [Gate(t.iloc[i], dtype='gazesim') for i in range(t.shape[0])]
    #make gate collision surfaces
    objGateCollider = [Gate(t.iloc[i], dtype='gazesim', dims=(3.5, 3.5)) for i in range(t.shape[0])]
    #make wall collision surfaces
    objWallCollider = getWallColliders(dims=(66, 36, 9), center=(0, 0, 4.5))
    #get drone state variables for event detection
    _t = d.loc[:, 'time-since-start [s]'].values
    _p = d.loc[:,('position_x [m]', 'position_y [m]', 'position_z [m]')].values
    #gate passing event
    evGatePass = [(i, detect_gate_passing(_t, _p, objGatePass[i])) for i in range(len(objGatePass))]
    evGatePass = [(i, v) for i, v in evGatePass if v.shape[0] > 0]
    print('gate passes:')
    print(evGatePass)
    print('')
    #gate collision event (discard the ones that are valid gate passes
    evGateCollision = []
    _tmp = [(i, detect_gate_passing(_t, _p, objGateCollider[i])) for i in range(len(objGateCollider))]
    _tmp = [(i, v) for i, v in _tmp if v.shape[0] > 0]
    for key, values in _tmp:
        new_vals = []
        for _k, _v in evGatePass:
            if _k == key:
                for value in values:
                    if value not in _v:
                        new_vals.append(_v)
        if len(new_vals) > 0:
            evGateCollision.append((key, np.array(new_vals)))
    print('gate collisions:')
    print(evGateCollision)
    print('')
    #wall collision events
    evWallCollision = [(i, detect_gate_passing(_t, _p, objWallCollider[i])) for i in range(len(objWallCollider))]
    evWallCollision = [(i, v) for i, v in evWallCollision if v.shape[0] > 0]
    print('wall collisions:')
    print(evWallCollision)
    print('')
    #save timestamps
    e = pd.DataFrame([])
    for i, v in evGatePass:
        for _v in v:
            e = e.append(pd.DataFrame({'time-since-start [s]' : _v, 'object-id' : i, 'object-name' : 'gate', 'is-collision' : 0, 'is-pass' : 1}, index = [0]))
    for i, v in evGateCollision:
        for _v in v:
            e = e.append(pd.DataFrame({'time-since-start [s]' : _v, 'object-id' : i, 'object-name' : 'gate', 'is-collision' : 1, 'is-pass' : 0}, index = [0]))
    for i, v in evWallCollision:
        for _v in v:
            e = e.append(pd.DataFrame({'time-since-start [s]' : _v, 'object-id' : i, 'object-name' : 'wall', 'is-collision' : 1, 'is-pass' : 0}, index = [0]))
    e = e.sort_values(by=['time-since-start [s]'])
    #make output folder
    outpath = './process/' + PATH.split('/logs/')[-1].split('trajectory_')[-1].split('.csv')[0] + '/'
    if os.path.exists(outpath) == False:
        make_path(outpath)
    #copy trajectory data
    copyfile(PATH, outpath + 'trajectory.csv')
    #save the events
    e.to_csv(outpath + 'events.csv', index=False)
    #compute performance metrics
    tStart = e['time-since-start [s]'].iloc[0]
    ec = e.loc[(e['is-collision'].values == 1), :]
    if ec.shape[0] > 0:
        tFirstCollision = ec['time-since-start [s]'].iloc[0]
        hasCollision = 1
        ind = e['time-since-start [s]'].values < tFirstCollision
        en = e.copy().iloc[ind, :]
    else:
        hasCollision = 0
        tFirstCollision = np.nan
        en = e.copy()
    if np.isnan(tFirstCollision):
        tEnd = np.nanmax(d['time-since-start [s]'].values)
    else:
        tEnd = tFirstCollision
    flightTime = tEnd - tStart
    numGatePasses = np.sum(en['is-pass'])
    ind = en['is-pass'].values == 1
    idGatePasses = [en.loc[ind, 'object-id'].values]
    tsGatePasses = [en.loc[ind, 'time-since-start [s]'].values]
    ind = (_t >= tStart) & (_t <= tEnd)
    flightDistance = np.nansum(np.linalg.norm(np.diff(_p[ind, :], axis=0), axis=1))
    #collect performance metrics in pandas dataframe
    p = pd.DataFrame({'time-start [s]': tStart, 'time-end [s]': tEnd, 'flight-time [s]' : flightTime, 'flight-distance [m]' : flightDistance,
                      'num-gate-passes' : numGatePasses, 'gate-id': idGatePasses, 'gate-ts': tsGatePasses, 'has-collision' : hasCollision, 'filepath' : outpath}, index=[0])
    #save performance metrics
    p.to_csv(outpath + 'performance.csv', index=False)
    #save the animation
    if toSaveAnimation:
        if os.path.isfile(outpath + 'anim.mp4') == False:
            print('..saving animation')
            gate_objects = objGatePass + objGateCollider + objWallCollider
            d['simulation-time-since-start [s]'] = d['time-since-start [s]'].values
            anim = Animation3D(d, Gate_objects=gate_objects, equal_lims=(-30, 30))
            anim.save(outpath + 'anim.mp4', writer='ffmpeg', fps=25)
            # anim.show()

PATH = './logs/'
toSaveAnimation = True

for w in os.walk(PATH):
    if w[0] == PATH:
        for f in w[2]:
            if f.find('.csv') != -1:
                pipeline(PATH=os.path.join(PATH, f), toSaveAnimation=toSaveAnimation)