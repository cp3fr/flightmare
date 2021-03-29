import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Load example data
trajectory = pd.read_csv('trajectory.csv')
t = trajectory['t'].values
p = trajectory.loc[:, ('px', 'py', 'pz')].values
q = trajectory.loc[:, ('qx', 'qy', 'qz', 'qw')].values
vW = trajectory.loc[:, ('vx', 'vy', 'vz')].values  #linear velocity in world
                                                   # frame
wB = trajectory.loc[:, ('wx', 'wy', 'wz')].values  # angular velocity in body
                                                   # frame

# Create transformer object.
class Transformer(object):
    """
    Returns acceleration in body frame from given time, position, and
    rotation in world frame.
    """

    def __init__(self,
            smoothing_width: float=0.2
            ) -> None:
        self.smoothing_width = smoothing_width


    def update(self,
            t: np.array,
            p: np.array,
            q: np.array
            ) -> np.array:
        """
        Returns acceleration in body frame (aB) from given time (t), position
        (p), and rotation (q) in world frame.
        """
        aW = self.position2acceleration(t, p, self.smoothing_width)
        aB = self.acceleration_world2body(q, aW)
        return aB


    def smooth_signal(self,
            x: np.array,
            window_len: int=11,
            window: str='hanning'
            ) -> np.array:
        """smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
        output:
            the smoothed signal
        example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y


    def position2velocity(self,
            t: np.array,
            p: np.array,
            smoothing_width: float=0.2
            ) -> np.array:
        """
        Velocity smoothed from position data.
        Assumes regular sampling.
        """
        if t.shape[0]<2:
            return np.zeros(p.shape)
        sr = 1 / np.nanmedian(np.diff(t))
        # Compute velocity
        v = np.diff(p, axis=0) * sr
        v = np.vstack((v, v[-1:, :]))
        # Signal smoothing.
        window = "blackman"
        window_len = int(sr * smoothing_width)
        if window_len > t.shape[0]:
            window_len = t.shape[0]
        v_smooth = np.empty(v.shape)
        for i in range(v.shape[1]):
            v_smooth[:, i] = self.smooth_signal(
                v[:, i],
                window_len=window_len,
                window=window)[window_len//2 :
                               v.shape[0]+window_len//2]
        return v_smooth


    def position2acceleration(self,
            t: np.array,
            p: np.array,
            smoothing_width: float=0.2
            ) -> np.array:
        """
        Acceleration smoothed from position data. Assumes regular sampling.
        """
        if t.shape[0]<3:
            return np.zeros(p.shape)
        sr = 1 / np.nanmedian(np.diff(t))
        # Compute acceleration.
        a = np.diff(p, n=2, axis=0) * (sr ** 2)
        a = np.vstack((a, a[-1:, :]))
        a = np.vstack((a, a[-1:, :]))
        # Signal smoothing.
        window = "blackman"
        window_len = int(sr * smoothing_width)
        if window_len > t.shape[0]:
            window_len = t.shape[0]
        a_smooth = np.empty(a.shape)
        for i in range(a.shape[1]):
            a_smooth[:, i] = self.smooth_signal(
                a[:, i],
                window_len=window_len,
                window=window)[window_len//2 :
                               a.shape[0]+window_len//2]
        # Add gravity.
        a_smooth[:, 2] -= 9.80665
        return a_smooth


    def acceleration_world2body(self,
            q: np.array,
            a: np.array,
            ) -> np.array:
        """
        Returns acceleration in body frame from given rotation quaternion and
        accleration in world frame.
        """
        return Rotation.from_quat(q).inv().apply(a)
transformer = Transformer(smoothing_width=0.2)

# Compute linear acceleration in world frame (just for comparison, not actually
# needed)
aW = transformer.position2acceleration(t=t, p=p)

# Compute linear acceleration in body frame (aB)
aB = transformer.update(t=t, p=p, q=q)

# Example on how to compute aB in "real-time", which needs sequences of at
# least 4 samples, ideally of about 100-200 ms length
window = 0.200
aB_sampled = np.empty(p.shape)
wB_sampled = np.empty(p.shape)
for i in range(t.shape[0]):
    ind = (t>=(t[i]-window)) & (t<=t[i])
    curr_t = t[ind]
    curr_q = q[ind, :]
    curr_p = p[ind, :]
    # Transform recent samples of time, position and rotation in world frame
    # to acceleration in body frame
    curr_aB = transformer.update(
        t=curr_t,
        p=curr_p,
        q=curr_q)
    # Take the most recent sampling point
    aB_sampled[i, :] = curr_aB[-1, :]
    # Angular rotation is already in body frame
    wB_sampled[i, :] = wB[i, :]

# Plot the drone state variables and compare aW to aB, sampled to long series
# based output computing
fig, axs = plt.subplots(6, 1)
iax = 0
axs[iax].plot(t, p)
axs[iax].set_ylabel('Position [m]')
iax += 1
axs[iax].plot(t, vW)
axs[iax].set_ylabel('Linear_Velocity_W [m/s/s]')
iax += 1
axs[iax].plot(t, aW)
axs[iax].set_ylabel('Linear_Acceleration_W [m/s/s]')
iax += 1
axs[iax].plot(t, aB)
axs[iax].plot(t, aB_sampled)
axs[iax].set_ylabel('Linear_Acceleration_B [m/s/s]')
iax += 1
axs[iax].plot(t, q)
axs[iax].set_ylabel('Rotation_W [quaternion]')
iax += 1
axs[iax].plot(t, wB)
axs[iax].plot(t, wB_sampled)
axs[iax].set_ylabel('Angular_Velocity_B [rad/s]')
axs[iax].set_xlabel('Time [s]')

plt.show()
