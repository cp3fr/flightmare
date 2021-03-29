import os
import re
import ffmpy
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from scipy.stats import iqr

def get_logfiles(indices=None, PATH='/home/cp3fr/.config/unity3d/LuGus Studios/Liftoff/LiftoffResearchLogs/', center_on='track', sr=500):
    #relevant paths
    session_path = PATH + 'Session_Description/'
    packet_path = PATH + 'Packet_logs/'
    udp_path = PATH + 'Udp_logs/'
    #get logfile names from session logs folder
    filenames = []
    for walker in os.walk(session_path):
        if walker[0] == session_path:
            filenames = [ name.split('.')[0] for name in walker[2] if name.find('.txt') != -1]
    filenames = sorted(filenames)
    #select files of interest
    if indices is not None:
        if isinstance(indices, int):
            indices = [indices]
        filenames = sorted([filenames[i] for i in indices])
    #load logfile data and save in a list of tuples {filename, sessionlogs, packetlogs}
    outlist = []
    for filename in filenames:
        #load session logs first
        S = import_sessionlogs(session_path + filename + '.txt')
        #if packet logs available load them too
        if os.path.isfile(packet_path + filename + '.txt'):
            P = import_packetlogs(packet_path + filename + '.txt')
        elif os.path.isfile(udp_path + filename + '.csv'):
            P = pd.read_csv(udp_path + filename + '.csv')
        else:
            P = None
        #center track items and position logs
        P, S = set_origin(P, S, center_on=center_on, sr=sr)
        #put packet and session logs to output tuple
        outlist.append({'filename' : filename, 'packets' : P, 'session' : S})
    return outlist

def import_packetlogs(PATH, timezone='Europe/Paris'):
    P = {}  # temporary dictionary of current packet content
    T = {}  # final dictionary of total packet content
    #labeling/order of dimensions in the inputdata
    subvars_pos = ['x', 'y', 'z']
    subvars_quat = ['x', 'y', 'z', 'w']
    with open(PATH) as file:
        for line in file:
            varname = None
            value = None
            # find start and end tags
            re_start = re.search(r"\<.*?>", line)
            re_end = re.search(r"\</.*?>", line)
            if re_start and re_end:
                # compose variable name
                varname = re_start.group().split(' unit')[0].strip('<>')
                unit = re.search(r'\".*?"', re_start.group())
                if unit is not None:
                    unit = unit.group().strip('"[]')
                    varname += ' [{}]'.format(unit)
                # extract values as numpy arrays
                i0 = re_start.span()[1]
                i1 = re_end.span()[0]
                value = np.fromstring(line[i0:i1].strip('()'), sep=',', dtype=float).flatten()
            # save variable values to packet dictionary
            if (varname is not None) and (value is not None):
                #if more than single value, save as seperare varialbes and append x y z w to variable name
                if value.shape[0]>1:
                    if value.shape[0]==3:
                        subvars = subvars_pos
                    else:
                        subvars = subvars_quat
                    for i in range(value.shape[0]):
                        newname = varname.split(' ')[0] + '_' + subvars[i] + ' ' + varname.split(' ')[1]
                        P[newname] = value[i]
                #if single value save as such
                else:
                    P[varname] = value[0]
            # if packet index entry exists, the packet is complete, and can now be
            # saved to the final dictionary
            if 'packet-index' in P.keys():
                # update and print packet number
                curr_packet = int(P['packet-index'])
                # print message only every 10th packet to avoid printing glitches
                if curr_packet % 10 == 0:
                    print('..saving packet {}'.format(curr_packet), end='\r', flush=True)
                # save packet to total dictionary
                for key in P.keys():
                    if key in T.keys():
                        T[key].append(P[key])
                    else:
                        T[key] = [P[key]]
                P = {}
        #print total number of packets
        print('Total number of packets: {}'.format(curr_packet))
        #make pandas dataframe
        D = pd.DataFrame(T, index=list(np.arange(0, len(T['packet-index']))))
        #transform position and rotation data from left hand unity coordinate frame to right hand RPG coordinate frame
        D = unity_to_rpg_for_packet_logs(D)
        #add start timestamp
        datetime_string = PATH.split('/')[-1].split('.')[0]
        ymd = [int(val) for val in datetime_string.split('_')[0].split('-')]
        hms = [int(val) for val in datetime_string.split('_')[1].split('-')]
        ts = pd.Timestamp(ymd[0], ymd[1], ymd[2], hms[0], hms[1], hms[2]).tz_localize(timezone).timestamp()
        D['simulation-timestamp {} [s]'.format(timezone)] = D['simulation-time-since-start [s]']+ts
    return D

def import_sessionlogs(PATH, to_convert_unity_to_rpg=True):
    #import trackitem position rotation and dimensions
    D = import_trackitems(PATH, to_convert_unity_to_rpg=to_convert_unity_to_rpg)
    #todo : import drone configuration
    #todo : import camera information
    #todo : import game settings
    #todo : import flight controller settings
    #todo : import PID settings
    #todo : import input settings
    #todo : import D attenuation settings
    return D

def set_origin(D=None, S=None, center_on='track', sr=500):
    # apply position/checkpoint data centering of world frame origin to the quadrotor start position
    if center_on is not None:
        # by default start position is assumed at world origin
        start_position = np.array([0., 0., 0.])
        # if center on start requested
        if center_on == 'start':
            # center the real time packet logs
            if D is not None:
                # determine the number of samples to use (up to 1sec from beginning of the logfile)
                if D.shape[0] > sr:
                    samples = sr
                else:
                    samples = D.shape[0]
                # variable names for each axis to be centered
                names = ['position_x [m]', 'position_y [m]', 'position_z [m]']
                # loop over axes
                for i in range(len(names)):
                    # update start position based on real-time logs, median of the position within first second
                    start_position[i] = np.nanmedian(D.loc[np.arange(0, samples, 1), names[i]].values)
        # if center on track requested
        elif center_on == 'track':
            if S is not None:
                p = S.loc[:, ('checkpoint-center_x [m]', 'checkpoint-center_y [m]', 'checkpoint-center_z [m]')].values
                start_position = np.nanmin(p, axis=0) + ((np.nanmax(p, axis=0) - np.nanmin(p, axis=0)) / 2.)
        # apply centering to packet logs
        if D is not None:
            names = ['position_x [m]', 'position_y [m]', 'position_z [m]']
            # loop over axes
            for i in range(len(names)):
                # center the position data on the start position
                D[names[i]] = D.loc[:, names[i]].values - start_position[i]
        # apply centering tosession logs
        if S is not None:
            # loop over variables to center
            for varname in ['position_%s [m]', 'checkpoint-center_%s [m]']:
                # variable names for each axis
                names = [varname % 'x', varname % 'y', varname % 'z']
                # loop over axes
                for i in range(len(names)):
                    # center the position data on the start position
                    S[names[i]] = S.loc[:, names[i]].values - start_position[i]
    return D, S

def import_trackitems(PATH, timezone='Europe/Paris', to_convert_unity_to_rpg=True):
    P = {}  # temporary dictionary of current packet content
    T = {}  # final dictionary of total packet content
    #labeling/order of variables in the inputdata
    subvars_pos = ['x', 'y', 'z']
    subvars_quat = ['x', 'y', 'z', 'w']
    track_item = None
    with open(PATH) as file:
        for line in file:
            if len(line)<=1:
                track_item = None
            else:
                re_trackitem = re.search(r"\<!--Track item.*?>", line)
                if re_trackitem:
                    track_item = int(re_trackitem.group().split('#')[1].strip('->'))
            if track_item is not None:
                varname = None
                value = None
                # find start and end tags
                re_start = re.search(r"\<.*?>", line)
                re_end = re.search(r"\</.*?>", line)
                if re_start and re_end:
                    # compose variable name
                    varname = re_start.group().split(' type')[0].strip('<>')
                    if varname.find('track-item-')!=-1:
                        varname = varname.split('track-item-')[1]
                    # extract values as numpy arrays
                    i0 = re_start.span()[1]
                    i1 = re_end.span()[0]
                    value_string = line[i0:i1]
                    if value_string[0]=='(':
                        value = np.fromstring(value_string.strip('()'), sep=',', dtype=float).flatten()
                    elif value_string=='true':
                        value = True
                    elif value_string=='false':
                        value = False
                    else:
                        value = value_string
                # save variable values to packet dictionary
                if (varname is not None) and (value is not None):
                    # if more than single value, save as separate variables and append x y z w to variable name
                    if isinstance(value, str) or isinstance(value, bool):
                        P[varname] = value
                    else:
                        #select the variable unit name
                        if (varname.find('position')!=-1) or (varname.find('checkpoint-center')!=-1) or (varname.find('checkpoint-size')!=-1):
                            unitname = ' [m]'
                        elif varname.find('rotation')!=-1:
                            unitname = ' [quaternion]'
                        else:
                            unitname = ''
                        #select the order of axes of the variable
                        if value.shape[0] > 1:
                            if value.shape[0] == 3:
                                subvars = subvars_pos
                            else:
                                subvars = subvars_quat
                            for i in range(value.shape[0]):
                                newname = varname + '_' + subvars[i] + unitname
                                P[newname] = value[i]
                        # if single value save as such
                        else:
                            P[varname] = value[0]
                # if the last entry of the track items has been read, the packet is complete, and can now be saved to the final dictionary
                # Algorithm:
                # .. either not a checkpoint, thus the last variable in the list 'checkpoint'
                # .. or it is a checkpoint and last variable in the list is 'checkpoint-had-physical-object'
                if ((('checkpoint' in P.keys()) and (P['checkpoint'] == False)) or
                    (('checkpoint-has-physical-object' in P.keys()) and (P['checkpoint'] == True))):
                    #make sure checkpoint-center variable exists
                    if 'checkpoint-center_x [m]' in P.keys():
                        print('', end='\r')
                    else:
                        for val1 in ['center', 'size']:
                            for val2 in ['x','y','z']:
                                P['checkpoint-' + val1 + '_' + val2 + ' [m]'] = np.nan
                        P['checkpoint-has-physical-object'] = False
                    #add track item number
                    P['track_item'] = track_item
                    #print the current item number to prompt
                    print('..saving track item {}'.format(track_item), end='\r')
                    # save packet to total dictionary
                    for key in P.keys():
                        if key in T.keys():
                            T[key].append(P[key])
                        else:
                            T[key] = [P[key]]
                    P = {}
        # make pandas dataframe
        if 'track_item' in T.keys():
            D = pd.DataFrame(T, index=list(np.arange(0, len(T['track_item']))))
            # print total number of trackitems to prompt
            print('Total number of track items: {}'.format(D.shape[0]))
            # add start timestamp
            datetime_string = PATH.split('/')[-1].split('.')[0]
            ymd = [int(val) for val in datetime_string.split('_')[0].split('-')]
            hms = [int(val) for val in datetime_string.split('_')[1].split('-')]
            ts = pd.Timestamp(ymd[0], ymd[1], ymd[2], hms[0], hms[1], hms[2]).tz_localize(timezone).timestamp()
            D['simulation-timestamp {} [s]'.format(timezone)] = ts
        else:
            D = None
        #fix coordinate frames
        if D is not None:
            if to_convert_unity_to_rpg:
                D = unity_to_rpg_for_track_items(D)
    return D

def unity_to_rpg_for_world_frame(M):
    '''
    :param M: (n, 3) np.array, 3d coordinates in world frame (left-hand, unity format)
    :return M2: (n, 3) np.array, 3 coordinates in world frame (right-hand, rpg format)
    '''
    # convert to right hand coordinate frame (x=right, y=upward, z=backward)
    M1 = np.empty((M.shape))
    M1[:] = np.nan
    M1[:, 0] = M[:, 0]
    M1[:, 1] = M[:, 1]
    M1[:, 2] = -M[:, 2]  # invert z-axis
    # convert to world (right hand) coordinate frame (x=forward, y=left, z=upward)
    M2 = Rotation.from_euler('x', [np.pi / 2]).apply(M1)
    return M2

def unity_to_rpg_for_checkpoint_size(M):
    '''
    :param M: (n, 3) np.array, 3d coordinates checkpoint size in world frame (left-hand, unity format)
    :return M2: (n, 3) np.array, 3 coordinates checkpoint size in world frame (right-hand, rpg format)
    '''
    # convert from left hand coordinate frame (x=right, y=upward, z=forward)
    # to right hand coordinate frame (x=forward, y=left, z=upward)
    M1 = np.empty((M.shape))
    M1[:] = np.nan
    M1[:, 0] = M[:, 2]
    M1[:, 1] = M[:, 0]
    M1[:, 2] = M[:, 1]
    return M1

def unity_to_rpg_for_rpy(M):
    '''
    :param M: (n, 3) np.array, roll pitch yaw body rates in body frame (left-hand, unity format)
    :return M2: (n, 3) np.array, roll pitch yaw body rates in body frame (right-hand, rpg format)
    '''
    # convert to from left hand coordinate frame (x=right, y=up, z=forward)
    # to right hand coordinate frame (x=right, y=upward, z=backward)
    M1 = np.empty((M.shape))
    M1[:] = np.nan
    M1[:, 0] = -M[:, 0]
    M1[:, 1] = M[:, 1]
    M1[:, 2] = M[:, 2]
    return M1

def unity_to_rpg_for_body_acceleration(M):
    '''
    :param M: (n, 3) np.array, in body frame (left-hand, unity format)
    :return M2: (n, 3) np.array, in body frame (right-hand, rpg format)
    '''
    # convert to from left hand coordinate frame (x=right, y=up, z=forward)
    # to right hand coordinate frame (x=right, y=upward, z=backward)
    M1 = np.empty((M.shape))
    M1[:] = np.nan
    M1[:, 0] = -M[:, 2]
    M1[:, 1] = M[:, 0]
    M1[:, 2] = M[:, 1]
    return M1

def unity_to_rpg_for_body_angularvelocity(M):
    '''
    :param M: (n, 3) np.array, in body frame (left-hand, unity format)
    :return M2: (n, 3) np.array, in body frame (right-hand, rpg format)
    '''
    # convert to from left hand coordinate frame (x=right, y=up, z=forward)
    # to right hand coordinate frame (x=right, y=upward, z=backward)
    M1 = np.empty((M.shape))
    M1[:] = np.nan
    M1[:, 0] = -M[:, 2]
    M1[:, 1] = M[:, 0]
    M1[:, 2] = -M[:, 1]
    return M1

def unity_to_rpg_for_rotation(Q):
    '''
    :param Q: (n, 4) np.array, Quadrotor rotation quaternion body in world (left-hand, unity format)
    :return Q2: (n, 4) np.array, Quadrotor rotation quaternion body in world (right-hand, rpg format)
    '''
    # convert to right hand coordinate frame (x=right, y=upward, z=backward)
    # by reversing z-axis: Q'(w, x, y, z) = Q(w, -x, -y, z)
    Q1 = np.empty((Q.shape))
    Q1[:] = np.nan
    Q1[:, 0] = -Q[:, 0]  # invert x-component
    Q1[:, 1] = -Q[:, 1]  # invert y-component
    Q1[:, 2] = Q[:, 2]
    Q1[:, 3] = Q[:, 3]
    # apply two transforms to end up with the desired right hand world coordinate
    tf_World = Rotation.from_euler('x', [np.pi / 2])
    tf_Body = Rotation.from_euler('xy', [-np.pi / 2., np.pi / 2])
    Q2 = ((tf_World * Rotation.from_quat(Q1)) * tf_Body).as_quat()
    return Q2

def unity_to_rpg_for_packet_logs(D):
    ##================
    ## UPDATE ROTATION
    #raw quaternion for left hand coordinate frame (x=right, y=upward, z=forward)
    Q1 = D.loc[:, ('rotation_x [quaternion]', 'rotation_y [quaternion]', 'rotation_z [quaternion]', 'rotation_w [quaternion]')].values
    #updated quaternion for right hand coordinate frame (x=forward, y=left, z=upward
    Q2 = unity_to_rpg_for_rotation(Q1)
    #update rotation data
    D['rotation_x [quaternion]'] = Q2[:, 0]
    D['rotation_y [quaternion]'] = Q2[:, 1]
    D['rotation_z [quaternion]'] = Q2[:, 2]
    D['rotation_w [quaternion]'] = Q2[:, 3]
    ##============================
    # UPDATE WORLD FRAME VARIABLES
    for varname in ['position_%s [m]', 'velocity_%s [m.s-1]', 'drag_%s [N]', 'thrust_%s [N]', 'thrust-left-front-motor_%s [N]',
                 'thrust-right-front-motor_%s [N]', 'thrust-right-back-motor_%s [N]', 'thrust-left-back-motor_%s [N]',
                 'acceleration-inertial-frame_%s [m.s-2]']:
        M1 = D.loc[:, (varname % 'x', varname % 'y', varname % 'z')].values
        M2 = unity_to_rpg_for_world_frame(M1)
        D[varname % 'x'] = M2[:, 0]
        D[varname % 'y'] = M2[:, 1]
        D[varname % 'z'] = M2[:, 2]
    #todo : update body frame variables
    ##============================================
    # UPDATE BODY FRAME VARIABLE: Angular Velocity
    varname ='angularvelocity_%s [rad.s-1]'
    M1 = D.loc[:, (varname % 'x', varname % 'y', varname % 'z')].values
    M2 = unity_to_rpg_for_body_angularvelocity(M1)
    D[varname % 'x'] = M2[:, 0]
    D[varname % 'y'] = M2[:, 1]
    D[varname % 'z'] = M2[:, 2]
    ##========================================
    # UPDATE BODY FRAME VARIABLE: Acceleration
    varname = 'acceleration-body-frame_%s [m.s-2]'
    M1 = D.loc[:, (varname % 'x', varname % 'y', varname % 'z')].values
    M2 = unity_to_rpg_for_body_acceleration(M1)
    D[varname % 'x'] = M2[:, 0]
    D[varname % 'y'] = M2[:, 1]
    D[varname % 'z'] = M2[:, 2]
    ##==============================
    # UPDATE ROLL PITCH YAW COMMANDS
    varnames = ('roll [-1;1]', 'pitch [-1;1]', 'yaw [-1;1]')
    M1 = D.loc[:, varnames].values
    M2 = unity_to_rpg_for_rpy(M1)
    D[varnames[0]] = M2[:, 0]
    D[varnames[1]] = M2[:, 1]
    D[varnames[2]] = M2[:, 2]
    return D

def unity_to_rpg_for_track_items(D):
    ##============================
    # UPDATE WORLD FRAME VARIABLES
    for varname in ['position_%s [m]', 'checkpoint-center_%s [m]']:
        M1 = D.loc[:, (varname % 'x', varname % 'y', varname % 'z')].values
        M2 = unity_to_rpg_for_world_frame(M1)
        D[varname % 'x'] = M2[:, 0]
        D[varname % 'y'] = M2[:, 1]
        D[varname % 'z'] = M2[:, 2]
    ##===============================
    # UPDATE CHECKPOINT SIZE VARIABLE
    for varname in ['checkpoint-size_%s [m]']:
        M1 = D.loc[:, (varname % 'x', varname % 'y', varname % 'z')].values
        M2 = unity_to_rpg_for_checkpoint_size(M1)
        D[varname % 'x'] = M2[:, 0]
        D[varname % 'y'] = M2[:, 1]
        D[varname % 'z'] = M2[:, 2]
    ##==========================================================
    # UPDATE ROTATION FOR THE CASE THAT EULER ANGLES WERE GIVEN
    if 'rotation_x [deg]' in D.columns:
        #get the raw rotation euler angles in degrees in Lugus coordinate frame
        M1 = D.loc[:, ('rotation_x [deg]', 'rotation_y [deg]', 'rotation_z [deg]')].values
        #within Lugus coordinate system, convert from euler angles in degrees to quaternions
        M2 = Rotation.from_euler('xyz', M1, degrees=True).as_quat()
        #convert from unity to RPG frame
        M3 = unity_to_rpg_for_rotation(M2)
        #add rotation quaternions to output data
        D['rotation_x [quaternion]'] = M3[:, 0]
        D['rotation_y [quaternion]'] = M3[:, 1]
        D['rotation_z [quaternion]'] = M3[:, 2]
        D['rotation_w [quaternion]'] = M3[:, 3]
        #drop the euler angle data
        D = D.drop(columns=['rotation_x [deg]', 'rotation_y [deg]', 'rotation_z [deg]'])
    # UPDATE ROTATION FOR THE CASE THAT QUATERNIONS WERE GIVEN
    elif 'rotation_x [quaternion]' in D.columns:
        # get the raw rotation quaternions in Lugus coordinate frame
        Q1 = D.loc[:, ('rotation_x [quaternion]', 'rotation_y [quaternion]', 'rotation_z [quaternion]', 'rotation_w [quaternion]')].values
        # convert from unity to RPG frame
        Q2 = unity_to_rpg_for_rotation(Q1)
        # add rotation quaternions to output data
        D['rotation_x [quaternion]'] = Q2[:, 0]
        D['rotation_y [quaternion]'] = Q2[:, 1]
        D['rotation_z [quaternion]'] = Q2[:, 2]
        D['rotation_w [quaternion]'] = Q2[:, 3]
    return D

def getPoseRpgFromUnity(p,q):
    '''get pose in rpg frame from unity frame'''
    #position
    p = p.reshape(-1, 3) #make 2d
    p[:, 2] = -p[:, 2] #invert z axis direction
    p = Rotation.from_euler('x', [np.pi / 2]).apply(p) #rotate axes
    #rotation
    q = q.reshape(-1, 4)  #make 2d
    q[:, 0] = -q[:, 0] #invert x component
    q[:, 1] = -q[:, 1] #invert y component
    tf_W = Rotation.from_euler('x', [np.pi / 2])
    tf_B = Rotation.from_euler('xy', [-np.pi / 2, np.pi / 2])
    q = ((tf_W * Rotation.from_quat(q)) * tf_B).as_quat() #apply two transforms to end up with the desired right hand world coordinate#
    return p, q

def getPoseUnityFromRpg(p, q):
    '''get pose in unity frame from rpg frame'''
    # position
    p = p.reshape(-1, 3)  # make 2d
    p = Rotation.from_euler('x', [np.pi / 2]).inv().apply(p)  #rotate axes
    p[:, 2] = -p[:, 2]  # invert z axis direction
    # rotation
    q = q.reshape(-1, 4)  # make 2d
    tf_W = Rotation.from_euler('x', [np.pi / 2]).inv()
    tf_B = Rotation.from_euler('xy', [-np.pi / 2, np.pi / 2]).inv()
    q = ((tf_W * Rotation.from_quat(q)) * tf_B).as_quat()  # apply two transforms to end up with the desired right hand world coordinate#
    q[:, 0] = -q[:, 0]  # invert x component
    q[:, 1] = -q[:, 1]  # invert y component
    return p, q

def xmlparse(string):
    string, item = string.split('</')[0].split('<')[-1].split('>')
    varname, string = string.split(' ')
    key, string = string.split('=')
    value = string.strip('"')
    return {'name': varname, key: value, 'value': item}

def plot_pose(P, R, ax=None, l=1.):
  '''
  plot_pose(P,R,ax=None,l=1.)

  Makes a 3d matplotlib plot of poses showing the axes direction:
  x=red, y=green, z=blue

  P : np.ndarray, position [poses x axes]
  R : np.ndarray, rotation [poses x quaternions]
  '''
  # make a figure if no axis was provided
  if ax is None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
  # make sure input data is 2D, where axes are in the 2nd dimension
  P = P.reshape((-1, 3))
  R = R.reshape((-1, 4))
  # loop over poses
  for i in range(P.shape[0]):
    # current position
    p0 = P[i, :]
    # loop over dimensions and plot the axes
    for dim, col in zip(np.arange(0, 3, 1), ['r', 'g', 'b']):
      u = np.zeros((1, 3))
      u[0, dim] = 1.
      v = Rotation.from_quat(R[i, :]).apply(u)[0]
      p1 = p0 + l * v
      ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=col)

def plot_vector(P, V, ax=None, scaling=1., col='m'):
  '''
  plot_pose(P,R,ax=None,l=1.)

  Makes a 3d matplotlib plot with vector pointing from the position:
  x=red, y=green, z=blue

  P : np.ndarray, position [poses x axes]
  V : np.ndarray, vector [poses x axes]
  '''
  # make a figure if no axis was provided
  if ax is None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
  # make sure input data is 2D, where axes are in the 2nd dimension
  P = P.reshape((-1, 3))
  V = V.reshape((-1, 3))
  # loop over poses
  for i in range(P.shape[0]):
    # current position
    p = P[i, :]
    # current vector
    v = V[i, :] * scaling
    ax.plot([p[0], p[0]+v[0]], [p[1], p[1]+v[1]], [p[2], p[2]+v[2]], color=col)

def save_animation(anim, PATH, filename='anim', fps=25):
    anim.save(PATH + filename + '.mp4', writer='ffmpeg', fps=fps)
    # ff = ffmpy.FFmpeg(inputs={PATH + filename + '.mp4': None}, outputs={PATH + filename + '.gif': None})
    # ff.run()
    return None

def plot_timeseries(P, gate_objects=None, events=None):
    # time
    t = P.loc[:, ('simulation-time-since-start [s]')].values
    # position
    p = P.loc[:, ('position_x [m]', 'position_y [m]', 'position_z [m]')].values
    # velocity
    v = P.loc[:, ('velocity_x [m.s-1]', 'velocity_y [m.s-1]', 'velocity_z [m.s-1]')].values
    # velocity norm
    vn = np.linalg.norm(v, axis=1)
    # acceleration
    a = P.loc[:, ('acceleration-inertial-frame_x [m.s-2]', 'acceleration-inertial-frame_y [m.s-2]',
                  'acceleration-inertial-frame_z [m.s-2]')].values
    # acceleration norm
    an = np.linalg.norm(a, axis=1)
    # angular velocity
    av = P.loc[:, ('angularvelocity_x [rad.s-1]', 'angularvelocity_y [rad.s-1]', 'angularvelocity_z [rad.s-1]')].values
    # angular velocity norm
    avn = np.linalg.norm(av, axis=1)
    # control commands
    trpy = P.loc[:, ('throttle [0;1]', 'roll [-1;1]', 'pitch [-1;1]', 'yaw [-1;1]')].values
    # motor thrusts
    motor_thrusts = np.empty((P.shape[0], 0))
    for n1 in ['left', 'right']:
        for n2 in ['front', 'back']:
            vals = np.linalg.norm(P.loc[:, ('thrust-{}-{}-motor_x [N]'.format(n1, n2),
                                            'thrust-{}-{}-motor_y [N]'.format(n1, n2),
                                            'thrust-{}-{}-motor_z [N]'.format(n1, n2))].values, axis=1)
            motor_thrusts = np.hstack((motor_thrusts, vals.reshape((-1, 1))))
    # collective thrust
    coll_thrust = np.linalg.norm(P.loc[:, ('thrust_x [N]', 'thrust_y [N]', 'thrust_z [N]')].values, axis=1)

    #make the figure
    fig, axs = plt.subplots(9, 1)
    fig.set_figwidth(10)
    fig.set_figheight(20)

    iax = 0
    axs[iax].plot(p[:, 0], p[:, 1])
    axs[iax].set_xlim((-20, 20))
    axs[iax].set_ylim((-20, 20))
    if gate_objects is not None:
        for i in range(len(gate_objects)):
            axs[iax].plot(gate_objects[i].corners[0, :], gate_objects[i].corners[1, :], 'k')
    axs[iax].set_xlabel('position_x [m]')
    axs[iax].set_ylabel('position_y [m]')

    if events is not None:
        iax = iax + 1
        for i in np.sort(events.loc[:, ('Gate [id]')].unique()):
            ind = events.loc[:, ('Gate [id]')].values == i
            _t = events.loc[ind, ('Timestamp [s]')].values
            _e = events.loc[ind, ('Gate [id]')].values
            axs[iax].plot(_t, _e, 'x')
        axs[iax].set_xlabel('time [s]')
        axs[iax].set_ylabel('gate passing event [id]')

    iax = iax + 1
    axs[iax].plot(t, p[:, 0])
    axs[iax].plot(t, p[:, 1])
    axs[iax].plot(t, p[:, 2])
    axs[iax].set_xlabel('time [s]')
    axs[iax].set_ylabel('position [m]')
    axs[iax].legend(['x', 'y', 'z'])

    iax = iax + 1
    axs[iax].plot(t, v[:, 0])
    axs[iax].plot(t, v[:, 1])
    axs[iax].plot(t, v[:, 2])
    axs[iax].plot(t, vn)
    axs[iax].set_xlabel('time [s]')
    axs[iax].set_ylabel('velocity [m/s]')
    axs[iax].legend(['x', 'y', 'z', 'norm'])

    iax = iax + 1
    axs[iax].plot(t, a[:, 0])
    axs[iax].plot(t, a[:, 1])
    axs[iax].plot(t, a[:, 2])
    axs[iax].plot(t, an)
    axs[iax].set_xlabel('time [s]')
    axs[iax].set_ylabel('acceleration [m/s2]')
    axs[iax].legend(['x', 'y', 'z', 'norm'])
    axs[iax].set_ylim((-50, 50))

    iax = iax + 1
    axs[iax].plot(t, av[:, 0])
    axs[iax].plot(t, av[:, 1])
    axs[iax].plot(t, av[:, 2])
    axs[iax].plot(t, avn)
    axs[iax].set_xlabel('time [s]')
    axs[iax].set_ylabel('angular velocity [rad/s]')
    axs[iax].legend(['x', 'y', 'z', 'norm'])

    iax = iax + 1
    axs[iax].plot(t, motor_thrusts[:, 0])
    axs[iax].plot(t, motor_thrusts[:, 1])
    axs[iax].plot(t, motor_thrusts[:, 2])
    axs[iax].plot(t, motor_thrusts[:, 3])
    axs[iax].plot(t, coll_thrust)
    axs[iax].set_xlabel('time [s]')
    axs[iax].set_ylabel('thrust [N]')
    axs[iax].legend(['m1', 'm2', 'm3', 'm4', 'collective'])

    iax = iax + 1
    axs[iax].plot(t, trpy[:, 0])
    axs[iax].set_xlabel('time [s]')
    axs[iax].set_ylabel('Throttle Setpoint [0,+1]')
    axs[iax].legend(['throttle'])

    iax = iax + 1
    axs[iax].plot(t, trpy[:, 1])
    axs[iax].plot(t, trpy[:, 2])
    axs[iax].plot(t, trpy[:, 3])
    axs[iax].set_xlabel('time [s]')
    axs[iax].set_ylabel('Body Rate Setpoints [-1,+1]')
    axs[iax].legend(['roll', 'pitch', 'yaw'])

def angle_between(v1, v2, method='3d'):
    '''
    v1: numpy array (n x 3) of 3d vectors [x, y, z] (reference vectors to which the angle will be computed
    v2: numpy array (n x 3) of 3d vectors [x, y, z]
    type: what kind of angle should be returned (absolute, horizontal, vertical)
    '''

    #convert to 2D arrays
    if len(v1.shape)==1:
        v1 = v1.reshape(-1, v1.shape[0])
    if len(v2.shape)==1:
        v2 = v2.reshape(-1, v2.shape[0])

    #make sure data is a numpy float
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)


    #select the 3D/2D vector components of interest
    if method == 'horizontal':
        #projection of the vector into the horizontal plane
        v1 = v1[:, [0, 1]]
        v2 = v2[:, [0, 1]]
    elif method == 'vertical':
        #projection of the vector into the vertical plane
        v1 = np.hstack((np.linalg.norm(v1[:, [0, 1]], axis=1).reshape((-1, 1)), v1[:, 2].reshape((-1, 1))))
        v2 = np.hstack((np.linalg.norm(v2[:, [0, 1]], axis=1).reshape((-1, 1)), v2[:, 2].reshape((-1, 1))))

    #normalize to unit vector length
    norm_v1 = v1 / np.linalg.norm(v1, axis=1).reshape((-1, 1))
    norm_v2 = v2 / np.linalg.norm(v2, axis=1).reshape((-1, 1))

    #compute the angle between the two vectors
    #..if 3d vectors, compute the absolute angle
    if method == '3d':
        angle = np.array([np.arccos(np.clip(np.dot(norm_v1[i, :], norm_v2[i, :]), -1.0, 1.0))
                          for i in range(norm_v1.shape[0])])
    #..if 2d vectors compute signed angle fo
    else:
        #signed counterclockwise angle
        angle = np.array([np.arctan2(norm_v2[i, 1], norm_v2[i, 0]) - np.arctan2(norm_v1[i, 1], norm_v1[i, 0])
                          for i in range(norm_v1.shape[0])])

        if method == 'horizontal':
            for i in range(angle.shape[0]):
                #fix the angles above 180 degress
                if np.abs(angle[i]) > np.pi:
                    if angle[i] < 0.:
                        angle[i] = (2 * np.pi - np.abs(angle[i]))
                    else:
                        angle[i] = -(2 * np.pi - np.abs(angle[i]))
                #flip the sign
                if (np.abs(angle[i]) > 0.):
                    angle[i] = -angle[i]

    return angle

def apply_gate_fixes(S, fix_position=False, fix_order=None, fix_rotation=None, fix_dimension=None):
    #fix gate center
    if fix_position:
        S.at[:, ('position_x [m]', 'position_y [m]', 'position_z [m]')] = S.loc[:, ('checkpoint-center_x [m]',
            'checkpoint-center_y [m]', 'checkpoint-center_z [m]')].values
    #fix gate order
    if fix_order is not None:
        S = S.iloc[fix_order]
    # fix gate rotation
    if fix_rotation is not None:
        for i in range(len(fix_rotation)):
            if fix_rotation[i] != 0:
                r = S.loc[:, ('rotation_x [quaternion]', 'rotation_y [quaternion]', 'rotation_z [quaternion]',
                              'rotation_w [quaternion]')].iloc[i].values
                r = (Rotation.from_euler('z', fix_rotation[i]) * Rotation.from_quat(r)).as_quat()
                S.at[np.arange(0, S.shape[0], 1) == i, (
                'rotation_x [quaternion]', 'rotation_y [quaternion]', 'rotation_z [quaternion]',
                'rotation_w [quaternion]')] = r
    # fix gate dimensions
    if fix_dimension is not None:
        for i in range(len(fix_dimension)):
            if fix_dimension[i] is not None:
                S.at[np.arange(0, S.shape[0], 1) == i, (
                'checkpoint-size_x [m]', 'checkpoint-size_y [m]', 'checkpoint-size_z [m]')] = np.array(
                    fix_dimension[i]).astype(np.float64)
    return S

def show_gate_positions_3d(S, axis_length=3, title=None, save_to_path=None):
    #make output figure
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(20)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    #loop over gates
    for igate in range(S.shape[0]):
        #gate center
        position = S.loc[:, ('position_x [m]', 'position_y [m]', 'position_z [m]')].iloc[igate].values
        # gate rotation
        rotation = S.loc[:, ('rotation_x [quaternion]', 'rotation_y [quaternion]', 'rotation_z [quaternion]',
                             'rotation_w [quaternion]')].iloc[igate].values
        #checkpoint center
        checkpoint_center = S.loc[:, ('checkpoint-center_x [m]', 'checkpoint-center_y [m]', 'checkpoint-center_z [m]')].iloc[
            igate].values
        #checkpoint size
        checkpoint_size = S.loc[:, ('checkpoint-size_x [m]', 'checkpoint-size_y [m]', 'checkpoint-size_z [m]')].iloc[igate].values
        #loop over axes
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    #determine gate corner by: 1. add half the xyz size to checkpoint center, 2. rotate according to rotation quaternion
                    corner = Rotation.from_quat(rotation).apply(
                        np.array([x * checkpoint_size[0] / 2, y * checkpoint_size[1] / 2, z * checkpoint_size[2] / 2]))
                    #plot current corner
                    ax.plot([checkpoint_center[0] + corner[0]], [checkpoint_center[1] + corner[1]], [checkpoint_center[2] + corner[2]], 'ob')
        #plot checkpoint center
        ax.plot([checkpoint_center[0]], [checkpoint_center[1]], [checkpoint_center[2]], 'xb')
        #plot current axes
        colors = ['r', 'g', 'b']
        for iaxis in range(3):
            v = np.array([0, 0, 0])
            v[iaxis] = axis_length
            v = Rotation.from_quat(rotation).apply(v)
            ax.plot([position[0], position[0] + v[0]], [position[1], position[1] + v[1]], [position[2], position[2] + v[2]], colors[iaxis])
        #plot gate center
        ax.plot([position[0]], [position[1]], [position[2]], 'xr')
        #add a label to the gate according to its order in sessionlogs
        ax.text(checkpoint_center[0], checkpoint_center[1], checkpoint_center[2], '{}'.format(igate), None, fontsize=20)
    #determine the plot center
    center = np.nanmean(
        S.loc[:, ('checkpoint-center_x [m]', 'checkpoint-center_y [m]', 'checkpoint-center_z [m]')].values, axis=0)
    #determine plot width
    width = 15
    #set plot limits
    ax.set_xlim((center[0] - width, center[0] + width))
    ax.set_ylim((center[1] - width, center[1] + width))
    ax.set_zlim((center[2] - width, center[2] + width))
    #add axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #set title
    ax.set_title(title)
    #set viewpoint
    ax.view_init(90, -90)
    #save figure
    if save_to_path:
        fig.savefig(save_to_path)
    #show plot
    plt.show()

def get_apex_features(position_corner, rotation_corner, t, P, V=None):
    #determine number of samples and dimensions in position data
    num_samples, num_dims = P.shape
    #if 2D data, add zeros
    if num_dims == 2:
        position_corner = np.hstack((position_corner, np.array([[0.]])))
        P = np.hstack((P, np.zeros((num_samples, 1))))
        if V is not None:
            V = np.hstack((V, np.zeros((num_samples, 1))))
    #compute position relative to corner
    Pc = P - position_corner
    # apply corner rotation to the data
    #  before: x axis pointed in direction of flight
    #  after: x axis points right, so the corner overshoot is along the negative y-axis, and progress from left to right along x axis
    Pc = Rotation.from_quat(rotation_corner).apply(Pc, inverse=True)
    if V is not None:
        V = Rotation.from_quat(rotation_corner).apply(V, inverse=True)
    #compute distance relative to corner
    distance_Pc_corner = np.linalg.norm(Pc, axis=1)
    #determine sampling point of the apex (point of shortest distance to corner)
    idx = np.nanargmin(distance_Pc_corner)
    #position of the apex in native space
    position_apex = P[idx, :]
    # velocity of the apex in native space
    if V is not None:
        velocity_apex = V[idx, :]
    else:
        velocity_apex = None
    #timestamp of the apex
    time_apex = t[idx]
    #Angle between apex and corner
    u = np.array([1, 0, 0])  # reference vector
    v = Pc[idx, :].flatten() # vector pointing to apex in normalized space
    angle_apex_corner = -angle_between(u, v, 'horizontal')  # flip the angle, such that negative is before the gate and positive after
    # longitudinal distance from corner
    long_distance_apex_corner = Pc[idx, 0]
    #lateral distance from corner
    lateral_distance_apex_corner = Pc[idx, 1]
    #apex distance from corner
    total_distance_apex_corner = distance_Pc_corner[idx]
    #put it back into 2d format if necessary
    if num_dims == 2:
        position_apex = position_apex[:2]
        if velocity_apex is not None:
            velocity_apex = velocity_apex[:2]
    return (time_apex, position_apex,velocity_apex, angle_apex_corner, total_distance_apex_corner, lateral_distance_apex_corner, long_distance_apex_corner)

def get_point_of_interest_features(position_corner, rotation_corner, t, P, V=None, point_name='apex', debug_plot=False):
    #determine number of samples and dimensions in position data
    num_samples, num_dims = P.shape
    #if 2D data, add zeros
    if num_dims == 2:
        position_corner = np.hstack((position_corner, np.array([[0.]])))
        P = np.hstack((P, np.zeros((num_samples, 1))))
        if V is not None:
            V = np.hstack((V, np.zeros((num_samples, 1))))
    #apply transformation to corner being the origin and gate orientation giving the angle
    #center data on corner position
    P_norm = P - position_corner
    position_corner_norm = position_corner - position_corner
    # apply corner rotation to the data
    #  before: inertial frame
    #  after: x points from inner to outer corner of the gate
    #         y points in the direction of flight
    #         z points upward
    P_norm = Rotation.from_quat(rotation_corner).apply(P_norm, inverse=True)
    if V is not None:
        V_norm = Rotation.from_quat(rotation_corner).apply(V, inverse=True)
    else:
        V_norm= None
    #compute distance relative to corner
    distance_P_norm_corner = np.linalg.norm(P_norm, axis=1)
    #determine point of interest
    if point_name == 'apex':
        #apex: the closest point to corner
        idx = np.nanargmin(distance_P_norm_corner)
    elif point_name == 'overshoot':
        #overshoot: the point furthest away on the lateral axis
        idx = np.nanargmax(P_norm[:, 0])
    else:
        idx = None
    #if no valid point specified:
    if idx is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    else:
        #position of point in native space
        position_point = P[idx, :]
        # velocity at point in native space
        if V is not None:
            velocity_point = V[idx, :]
        else:
            velocity_point = None
        #timestamp of point
        time_point = t[idx]
        #Angle between corner-to-point and corner-x-axis
        u = np.array([1, 0, 0])  # reference vector pointing from gate inner to outer corner
        v = P_norm[idx, :].flatten() # vector pointing to overshoot point
        angle_point_corner = -angle_between(u, v, 'horizontal')  #signed angle, where negative values are before gate entering, and postive values after
        # lateral distance from corner
        lateral_distance_point_corner = P_norm[idx, 0]
        # longitudinal distance from corner
        longitudinal_distance_point_corner = P_norm[idx, 1]
        #overshoot distance from corner
        total_distance_point_corner = distance_P_norm_corner[idx]
        #put it back into 2d format if necessary
        if num_dims == 2:
            position_point = position_point[:2]
            if velocity_point is not None:
                velocity_point = velocity_point[:2]
        #make debug plot
        if debug_plot:
            plt.figure()
            plt.subplot(1,4,1)
            plt.plot(P[:, 0], P[:, 1])
            plt.plot(position_point[0], position_point[1], 'rx')
            colors = ['r', 'g', 'b']
            for i in range(3):
                vec = np.array([0, 0, 0])
                vec[i] = 1
                vec = Rotation.from_quat(rotation_corner).apply(vec)
                plt.plot([position_corner[0, 0], position_corner[0, 0] + vec[0]],
                         [position_corner[0, 1], position_corner[0, 1] + vec[1]], colors[i])
            plt.xlabel('Position X [m]')
            plt.ylabel('Position Y [m]')
            plt.title('Native Position')
            plt.legend(['Position [m]', point_name, 'Corner X axis', 'Corner Y axis', 'Corner Z axis'])
            plt.subplot(1, 4, 2)
            plt.plot(P_norm[:, 0], P_norm[:, 1])
            corner_vector = np.array([1, 0, 0])
            plt.plot([0 + corner_vector[0]], [0 + corner_vector[1]], 'r')
            plt.plot(P_norm[idx, 0], P_norm[idx, 1], 'rx')
            colors = ['r', 'g', 'b']
            for i in range(3):
                vec = np.array([0, 0, 0])
                vec[i] = 1
                plt.plot([0, 0 + vec[0]],
                         [0, 0 + vec[1]], colors[i])
            plt.xlabel('Lateral Position [m]')
            plt.ylabel('Longintudinal Position [m]')
            plt.title('Normalized Position')
            plt.subplot(1, 4, 3)
            plt.plot(t, P_norm[:, 0])
            plt.plot(t, P_norm[:, 1])
            plt.plot([np.min(t), np.max(t)], [0, 0], 'r-')
            plt.xlabel('Time [s]')
            plt.ylabel('Normalized Position [m]')
            plt.subplot(1, 4, 4)
            plt.plot(t, distance_P_norm_corner)
            plt.xlabel('Time [s]')
            plt.ylabel('Total Distance from Corner [m]')
            plt.show()

        return (time_point, position_point, velocity_point, angle_point_corner, total_distance_point_corner,
                lateral_distance_point_corner, longitudinal_distance_point_corner)