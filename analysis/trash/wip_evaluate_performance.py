try:
    from analysis.trash.preprocessing_utils import *
    from analysis.trash.processing_utils import *
    from analysis.trash.Animation3D import *
    from analysis.trash.Gate import *
    from analysis.trash.laptracker_utils import *
except:
    from functions.preprocessing_utils import *
    from functions.processing_utils import *
    from functions.Animation3D import *
    from functions.Gate import *
    from functions.laptracker_utils import *

#todo: plot performance metrics

#todo: fix 2d point extraction for Gate class
#todo: update laptracker and Gate scripts in Liftoff plugin
#OK: fix gate passing event detection for gates

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

def extractFeaturesSaveAnimation(PATH, toShowAnimation=False, toSaveAnimation=False):
    print('------------------------')
    print(PATH)
    print('')
    #find switchtime
    filename = PATH.split('/')[-1]
    tSwitch = None
    for s in ['_st-', '_switch-']:
        if filename.find(s) != -1:
            tSwitch = int(filename.split(s)[-1].split('_')[0]) / 10
    #read drone state logs
    d = pd.read_csv(PATH)
    d = d.sort_values(by=['time-since-start [s]'])
    #read gate poses for the current track
    t = pd.read_csv('../tracks/flat.csv')
    #add zoffset to gate positions in flightmare
    zOffset = 0.35
    t['pos_z'] += zOffset
    #make gate passing surfaces
    objGatePass = [Gate(t.iloc[i], dtype='gazesim', dims=(2.5, 2.5)) for i in range(t.shape[0])]
    #make gate collision surfaces
    objGateCollider = [Gate(t.iloc[i], dtype='gazesim', dims=(3.5, 3.5)) for i in range(t.shape[0])]
    #make wall collision surfaces
    objWallCollider = getWallColliders(dims=(66, 36, 9), center=(0, 0, 4.5 + zOffset))
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
    outpath = '/process/'.join((PATH.split('.csv')[0] + '/').split('/logs/'))
    if os.path.exists(outpath) == False:
        make_path(outpath)
    #copy trajectory data
    copyfile(PATH, outpath + 'trajectory.csv')
    #save the events
    e.to_csv(outpath + 'events.csv', index=False)
    #compute performance metrics
    #----------------------------
    #start time: start of the flight
    tStart = e['time-since-start [s]'].iloc[0]
    #find collision events
    ec = e.loc[(e['is-collision'].values == 1), :]
    #if collisions detected
    if ec.shape[0] > 0:
        tFirstCollision = ec['time-since-start [s]'].iloc[0]
        hasCollision = 1
        ind = e['time-since-start [s]'].values < tFirstCollision
        en = e.copy().iloc[ind, :]
    #if no collisions detected
    else:
        hasCollision = 0
        tFirstCollision = np.nan
        en = e.copy()
    #end time: time of first collision or end of logging
    if np.isnan(tFirstCollision):
        tEnd = np.nanmax(d['time-since-start [s]'].values)
    else:
        tEnd = tFirstCollision
    #compute the flight times
    flightTimeTotal = tEnd - tStart
    flightTimeMPC = flightTimeTotal
    flightTimeNetwork = 0.
    if tSwitch is not None:
        flightTimeMPC = tSwitch - tStart
        flightTimeNetwork = tEnd - tSwitch
    #how many gates were passed in total
    ind = en['is-pass'].values == 1
    numGatePassesTotal = np.sum(ind)
    idGatePassesTotal = [en.loc[ind, 'object-id'].values]
    tsGatePassesTotal = [en.loc[ind, 'time-since-start [s]'].values]
    if tSwitch is not None:
        # how many gates were passed by MPC
        ind = (en['time-since-start [s]'].values <= tSwitch) & (en['is-pass'].values == 1)
        numGatePassesMPC = np.sum(ind)
        idGatePassesMPC = [en.loc[ind, 'object-id'].values]
        tsGatePassesMPC = [en.loc[ind, 'time-since-start [s]'].values]
        #how many gates were passed by the network
        ind = (en['time-since-start [s]'].values > tSwitch) & (en['time-since-start [s]'].values <= tEnd) & (en['is-pass'].values == 1)
        numGatePassesNetwork = np.sum(ind)
        idGatePassesNetwork = [en.loc[ind, 'object-id'].values]
        tsGatePassesNetwork = [en.loc[ind, 'time-since-start [s]'].values]
    else:
        # how many gates were passed by MPC
        numGatePassesMPC = numGatePassesTotal
        idGatePassesMPC = idGatePassesTotal
        tsGatePassesMPC = tsGatePassesTotal
        # how many gates were passed by the network
        numGatePassesNetwork = 0
        idGatePassesNetwork = None
        tsGatePassesNetwork = None
    #flight distance
    ind = (_t >= tStart) & (_t <= tEnd)
    flightDistanceTotal = np.nansum(np.linalg.norm(np.diff(_p[ind, :], axis=0), axis=1))
    if tSwitch is not None:
        ind = (_t >= tStart) & (_t <= tSwitch)
        flightDistanceMPC = np.nansum(np.linalg.norm(np.diff(_p[ind, :], axis=0), axis=1))
        ind = (_t > tSwitch) & (_t <= tEnd)
        flightDistanceNetwork = np.nansum(np.linalg.norm(np.diff(_p[ind, :], axis=0), axis=1))
    else:
        flightDistanceMPC = flightDistanceTotal
        flightDistanceNetwork = 0
    #collect performance metrics in pandas dataframe
    p = pd.DataFrame({'time-start [s]': tStart, 'time-switch [s]': tSwitch, 'time-end [s]': tEnd,
                      'flight-time-total [s]' : flightTimeTotal, 'flight-time-mpc [s]' : flightTimeMPC, 'flight-time-network [s]' : flightTimeNetwork,
                      'flight-distance-total [m]' : flightDistanceTotal, 'flight-distance-mpc [m]' : flightDistanceMPC, 'flight-distance-network [m]' : flightDistanceNetwork,
                      'num-gate-passes-total' : numGatePassesTotal, 'num-gate-passes-mpc' : numGatePassesMPC, 'num-gate-passes-network' : numGatePassesNetwork,
                      'gate-id-total': idGatePassesTotal, 'gate-id-mpc': idGatePassesMPC, 'gate-id-network': idGatePassesNetwork,
                      'gate-ts-total': tsGatePassesTotal, 'gate-ts-mpc': tsGatePassesMPC, 'gate-ts-network': tsGatePassesNetwork,
                      'has-collision' : hasCollision, 'filepath' : outpath}, index=[0])
    #save performance metrics
    p.to_csv(outpath + 'performance.csv', index=False)
    #save the animation
    if toSaveAnimation or toShowAnimation:
        print('..saving animation')
        gate_objects = objGatePass + objGateCollider + objWallCollider
        d['simulation-time-since-start [s]'] = d['time-since-start [s]'].values
        anim = Animation3D(d, Gate_objects=gate_objects, equal_lims=(-30, 30))
        if toSaveAnimation:
            if os.path.isfile(outpath + 'anim.mp4') == False:
                anim.save(outpath + 'anim.mp4', writer='ffmpeg', fps=25)
        if toShowAnimation:
            anim.show()

toExtractFeatures = False
toShowAnimation = False
toSaveAnimation = False
toPlotFeatures = True

PATH = '../logs/'
SELECT_MODELS = [0] #[0, 1, 2]
MODELS = ['dda_0', 'dda_offline_0', 'resnet_test'] #['dda_0', 'dda_offline_0', 'resnet_test']
MODEL_TRACKS = {'dda_0' : 'flat',
                'dda_offline_0' : 'flat',
                'resnet_test' : 'flat'}
MODEL_SWITCH_TIMES = {'dda_0' : [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                      'dda_offline_0' : [5, 10, 15, 20, 25, 30, 35, 40, 45],
                      'resnet_test' : [6, 8, 10, 12, 14]}
MODEL_FILE_STEM = {'dda_0' : 'dda_0/trajectory_mpc2nw_mt-70_st-{}_',
                   'dda_offline_0' : 'dda_offline_0/trajectory_mpc2nw_st-{}_if-60_cf-20_',
                   'resnet_test' : 'resnet_test/trajectory_mpc2nw_switch-{}_'}

if toExtractFeatures:
    for m in [MODELS[i] for i in SELECT_MODELS]:
        for w in os.walk(PATH):
            if w[0].find(m) != -1:
                for f in w[2]:
                    if f.find('.csv') != -1:
                        extractFeaturesSaveAnimation(PATH=os.path.join(w[0], f),
                                                     toShowAnimation=toShowAnimation,
                                                     toSaveAnimation=toSaveAnimation)

if toPlotFeatures:
    for m in [MODELS[i] for i in SELECT_MODELS]:
        if m in MODEL_SWITCH_TIMES.keys():
            for s in MODEL_SWITCH_TIMES[m]:
                #filename stem to look for
                stem = MODEL_FILE_STEM[m].format('%02d' % s)
                #get file paths for different model-switchtime combinations
                filepaths = []
                for w in os.walk('../process/'):
                    # print(w[0])
                    if w[0].find(stem) != -1:
                        filepaths.append(w[0] + '/')
                #make performance plot
                fig, axs = plt.subplots(2,2)
                fig.set_figwidth(20)
                fig.set_figheight(20)
                axs = axs.flatten()
                #plot flight path
                iax = 0
                for f in filepaths:
                    d = pd.read_csv(f + 'trajectory.csv')
                    p = pd.read_csv(f + 'performance.csv')
                    t0 = p['time-start [s]'].iloc[0]
                    ts = p['time-switch [s]'].iloc[0]
                    t1 = p['time-end [s]'].iloc[0]
                    for _t0, _t1, _c in [(t0, ts, 'b'), (ts, t1, 'r')]:
                        ind = (d['time-since-start [s]'].values >= _t0) & (d['time-since-start [s]'].values <= _t1)
                        px = d.loc[ind, 'position_x [m]'].values
                        py = d.loc[ind, 'position_y [m]'].values
                        axs[iax].plot(px, py, _c)
                #todo: add gates
                axs[iax].set_title(stem)
                axs[iax].set_xlim((-35, 35))
                axs[iax].set_ylim((-35, 35))

                #todo: plot drone state
                for f in filepaths:
                    d = pd.read_csv(f + 'trajectory.csv')
                    p = pd.read_csv(f + 'performance.csv')
                    iax+=1
                    cols = ['k', 'r', 'g', 'b']
                    lines = ['-', '--']
                    cmdNames = ['throttle', 'roll', 'pitch', 'yaw']
                    modelNames = ['mpc', 'nw']
                    leg = []
                    for i in range(len(cmdNames)):
                        for j in range(len(modelNames)):
                            _col = cols[i]
                            _cmdName = cmdNames[i]
                            _line = lines[j]
                            _modelName = modelNames[j]
                            n = '{}_{}'.format(_cmdName, _modelName)
                            leg.append(n)
                            ind = (d['time-since-start [s]'].values >= p['time-start [s]'].iloc[0]) & (d['time-since-start [s]'].values <= p['time-end [s]'].iloc[0])
                            x = d.loc[ind, 'time-since-start [s]'].values
                            y = d.loc[ind, n].values
                            axs[iax].plot(x, y, linestyle=_line, color=_col)
                    axs[iax].legend(leg)
                    axs[iax].set_title(f)

                # time - since - start[
                #     s], throttle_mpc, roll_mpc, pitch_mpc, yaw_mpc, throttle_nw, roll_nw, pitch_nw, yaw_nw, position_x[
                #     m], position_y[m], position_z[m], rotation_w[quaternion], rotation_x[quaternion], rotation_y[
                #     quaternion], rotation_z[quaternion], velocity_x[m], velocity_y[m], velocity_z[m], omega_x[rad / s], \
                # omega_y[rad / s], omega_z[rad / s], network_used

                #todo: plot control commands MPC and network predictions

                #todo: plot performance features

                plt.show()