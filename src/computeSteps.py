"""
computing the number of steps from data provided by accelerometer
comparing the results with real distance from the car provided by GPS
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import src.lowpass as lp
import src.peakAccelThreshold as pat
import src.adaptiveJerkPaceThreshold as asjt
from scipy.signal import find_peaks
import os
import src.peakDetection as pd
import statistics

DATA_PATH = '../data/PDR_data/'
GRAPH_PATH = '../graphs/'


def pull_data(dir_name, file_name):
    f = open(dir_name + '/' + file_name + '.csv')
    xs = []
    ys = []
    zs = []
    rs = []
    timestamps = []
    line_counter = 0
    for line in f:
        if line_counter > 0:
            value = line.split(',')
            if len(value) > 3:
                timestamps.append(float(value[-4]))
                x = float(value[-3])
                y = float(value[-2])
                z = float(value[-1])
                r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
                xs.append(x)
                ys.append(y)
                zs.append(z)
                rs.append(r)
        line_counter += 1
    return np.array(xs), np.array(ys), np.array(zs), np.array(rs), np.array(timestamps)

def data_without_mean(data):
    return np.array(data) - statistics.mean(data)

def lowpass_filter(data, cutoff, fs, order):
    return lp.butter_lowpass_filter(data, cutoff, fs, order)


def peak_accel_threshold(data, timestamps, cst):
    return pat.peak_accel_threshold(data, timestamps, cst)


def compute__peak_accel_threshold(data, timestamps, cst, step_length=0.69):
    '''
    1st method

    Calculating steps using a static threshold

    Parameters
    ----------
    data : array
        Signal full of peaks
    timestamps : array
        Array of timestamps
    without_mean: boolean
        If true, the mean of the signal is substracted
    cst : float
        The threshold to use to filter peaks
    step_length : float
        Distance travelled with one step

    Returns
    -------
    tuple (int, float)
        The couple (steps, distance)

    '''

    crossings = pat.peak_accel_threshold(data, timestamps, cst)
    steps = len(crossings)//2
    length = step_length
    distance_traveled = steps * length
    return steps, distance_traveled


def graph__peak_accel_threshold(data, timestamps, cst, step_length=0.69):
    '''
    Graph visualization for the first method

    Parameters
    ----------
    data : array
        Signal full of peaks
    timestamps : array
        Array of timestamps
    without_mean: boolean
        If true, the mean of the signal is substracted
    cst : float
        The threshold to use to filter peaks
    step_length : float
        Distance travelled with one step

    '''

    crossings = pat.peak_accel_threshold(data, timestamps, cst)
    steps = len(crossings)//2
    length = step_length
    distance_traveled = steps * length
    plt.title("Peak Acceleration Threshold: {} steps, {} m".format(steps,round(distance_traveled,2)))
    plt.xlabel("Time [sec]")
    plt.ylabel("Acceleration Norm [m/s^2]")
    plt.grid()
    plt.plot(timestamps,data,'b-', linewidth=2)
    plt.plot(timestamps, np.full(shape=len(timestamps), fill_value=cst, dtype=float),'r',linewidth=0.5)
    plt.plot(crossings.T[0], crossings.T[1], 'ro', linewidth=0.5)
    plt.savefig(GRAPH_PATH+'compute_steps_method1')
    plt.show()


def compute__adaptive_step_jerk_threshold(data, timestamps, cst, step_length=0.69):
    '''
    2nd method

    Calculating steps using a dynamic adaptive threshold

    Parameters
    ----------
    data : array
        Signal full of peaks
    timestamps : array
        Array of timestamps
    step_length : float
        Distance travelled with one step

    Returns
    -------
    (int, float, int, float)
        (steps, distance, jumps, avg)

    '''
    jumps, avg = asjt.adaptive_step_jerk_threshold(data, timestamps, cst)
    steps = len(jumps)
    distance = step_length * steps
    return steps, distance, jumps, avg


def graph__adaptive_step_jerk_threshold(data, timestamps, cst):
    '''
    Graph visualization for the 2nd method

    Parameters
    ----------
    data : array
        Signal full of peaks
    timestamps : array
        Array of timestamps
    step_length : float
        Distance travelled with one step

    '''
    steps, distance, jumps, avg = compute__adaptive_step_jerk_threshold(data, timestamps, cst)
    ts = [jump['ts'] for jump in jumps]
    val = [jump['val'] for jump in jumps]
    plt.title("Adaptive Step Jerk Threshold: {} steps, {} m".format(steps, round(distance,2)))
    plt.xlabel("Time [sec]")
    plt.ylabel("Acceleration Norm [m/s^2]")
    plt.grid()
    plt.plot(timestamps, data, 'b-', linewidth=2)
    plt.plot(ts, val, 'ro')
    plt.savefig(GRAPH_PATH+'compute_steps_method2')
    plt.show()


def compute__find_peaks(data, timestamps, distance=60, prominence=0.5, step_length=0.69):
    '''
    3rd method

    Calculating steps with finding peaks inside a signal based on peak properties

    Parameters
    ----------
    data : array
        Signal full of peaks
    timestamps : array
        Array of timestamps
    distance : int
        Required minimal horizental distance (>=1) between neighbouring peaks
    prominence : int
        Required prominence of peaks
    step_length : float
        Distance travelled with one step

    Returns
    -------
    (int, float, Array, dict)
        (steps, distance, peaks, properties)

    '''
    peaks, properties = find_peaks(data, distance=distance, prominence=prominence)
    steps = len(peaks)
    distance_traveled = steps * step_length
    return steps, distance_traveled, peaks, properties


def graph__find_peaks(data, timestamps, distance=60, prominence=0.5, step_length=0.69):
    '''
    Graph visualization for the 3rd method

    Parameters
    ----------
    data : array
        Signal full of peaks
    timestamps : array
        Array of timestamps
    distance : int
        Required minimal horizental distance (>=1) between neighbouring peaks
    prominence : int
        Required prominence of peaks
    step_length : float
        Distance traveled with one step

    '''
    steps, distance_traveled, peaks, properties = compute__find_peaks(data,timestamps,distance,prominence,step_length)
    plt.title("find_peaks method: {} steps, {} m".format(steps, round(distance_traveled,2)))
    plt.xlabel("Time [sec]")
    plt.ylabel("Acceleration Norm [m/s^2]")
    plt.grid()
    plt.plot(timestamps, data, 'b-', linewidth=2)
    plt.plot(timestamps[peaks],data[peaks],'x', color="red", label="peaks detected")
    plt.legend()
    plt.savefig(GRAPH_PATH+'compute_steps_method3')
    plt.show()


def compute__peakdetect(data, timestamps, lookahead=1, delta=3/4, step_length=0.69):
    """
    Algorithm for detecting local maximas and minimas in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    Args:
        data (array)
            signal full of peaks
        timestamps (array)
            array of timestamps
        lookahead (int, optional) Defaults to 1
            distance to look ahead from a peak candidate to
            determine if it is the actual peak
        delta (float, optional) Defaults to 3/4
            specifies a minimum difference between a peak and
            the following points, before a peak may be considered a peak. Useful
            to hinder the algorithm from picking up false peaks towards to end of
            the signal. To work well delta should be set to 'delta >= RMSnoise * 5'.
        step_length (float, optional) Defaults to 0.69
            distance traveled with one step

    Returns:
        (int, float, list, list, list)
            (steps, distance, xm, ym, timestamps_for_steps)
            where timestamps_for_steps is the list of timestamps for each detected step
    """
    _max, _min = pd.peakdetect(data,timestamps, lookahead, delta)
    xm = [p[0] for p in _max]
    ym = [p[1] for p in _max]

    # 使用 np.searchsorted 查找 xm 中时间值在 timestamps 中的对应索引
    timestamps_for_steps = [timestamps[np.searchsorted(timestamps, time)] for time in xm]

    steps = len(_max)
    distance = steps * step_length
    return steps, distance, xm, ym, timestamps_for_steps


def graph__peakdetect(data, timestamps, lookahead=1, delta=3/4, step_length=0.69):
    """
    Graph visualization for the peakdetect method

    Args:
        data (array)
            signal full of peaks
        timestamps (array)
            array of timestamps
        lookahead (int, optional) Defaults to 1
            distance to look ahead from a peak candidate to
            determine if it is the actual peak
        delta (float, optional) Defaults to 3/4
            specifies a minimum difference between a peak and
            the following points, before a peak may be considered a peak
        step_length (float, optional) Defaults to 0.69
            distance traveled with one step
    """
    steps, distance_traveled, xm, ym, timestamps_for_steps = compute__peakdetect(data, timestamps, lookahead, delta, step_length)
    plt.title("Peakdetect Method: {} steps, {} m".format(steps, round(distance_traveled,2)))
    plt.xlabel("Time121 [sec]")
    plt.ylabel("Acceleration Norm [m/s^2]")
    plt.grid()
    plt.plot(timestamps, data, 'b-', linewidth=2)
    plt.plot(timestamps_for_steps, ym, 'ro', label="detected peaks")
    plt.legend()
    plt.savefig(GRAPH_PATH+'compute_steps_method4')
    plt.show()


def compute(display_graph=1, without_mean=0):
    '''
    main function
    diplays graphs and returns computed steps of each method
    if display = 1, graphs are displayed
    '''

    GRAVITY = 9.81
    #filter requirements
    order = 4
    fs = 100
    cutoff = 2
    x_data, y_data, z_data, r_data, timestamps = pull_data(DATA_PATH, 'Accelerometer')
    #filter
    r = lowpass_filter(r_data, cutoff, fs, order)
    #mean
    if without_mean == 1:
        r = data_without_mean(r) #removing mean from data
        ZERO = 0
    else:
        ZERO = GRAVITY
    #1st method
    step1, dist1 = compute__peak_accel_threshold(r,timestamps,ZERO) #zero crossing
    if display_graph == 1: graph__peak_accel_threshold(r,timestamps,ZERO)

    #2nd method
    step2, dist2, _, _ = compute__adaptive_step_jerk_threshold(r,timestamps,ZERO) #adaptive threshold
    if display_graph == 1: graph__adaptive_step_jerk_threshold(r,timestamps,ZERO)

    #3rd method
    step3, dist3, _, _ = compute__find_peaks(r,timestamps)
    if display_graph == 1: graph__find_peaks(r,timestamps)

    #4th method
    step4, dist4, _, _,timestamps_for_steps1 = compute__peakdetect(r,timestamps)
    for i in range(0,len(timestamps_for_steps1)):
        print(f'Step {i+ 1} 对应的时间戳为{timestamps_for_steps1[i]}')
    if display_graph == 1: graph__peakdetect(r,timestamps,1,3/4)

    return step1, step2, step3, step4


def compute_step_lengths_times(display_graph=1, without_mean=0):

    GRAVITY = 9.81
    #filter requirements
    order = 4
    fs = 100
    cutoff = 2
    x_data, y_data, z_data, r_data, timestamps = pull_data(DATA_PATH, 'Accelerometer')
    #filter
    r = lowpass_filter(r_data, cutoff, fs, order)
    #mean
    if without_mean == 1:
        r = data_without_mean(r) #removing mean from data
        ZERO = 0
    else:
        ZERO = GRAVITY


    #4th method
    step4, dist4, _, _,timestamps_for_steps1 = compute__peakdetect(r,timestamps)
    print(dist4)
    if display_graph == 1: graph__peakdetect(r,timestamps,1,3/4)

    return step4, dist4, timestamps_for_steps1


if __name__ == "__main__":

    compute(display_graph=1,without_mean=0)
    #compute_step_lengths_times(display_graph=1, without_mean=0)
