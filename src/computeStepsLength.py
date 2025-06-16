import numpy as np
import matplotlib.pyplot as plt
import src.computeSteps as cSteps
import src.peakAccelThreshold as pat

DATA_PATH = r'D:\博士\IMU定位\final\PersonDeadReckoning\data1/'
GRAPH_PATH = r'D:\博士\IMU定位\final\PersonDeadReckoning\graphs1/'


def get_data(data, timestamps, start, end):
    '''
    获取指定时间段的加速度数据
    '''
    index_start = 0
    index_end = len(timestamps) - 1
    while timestamps[index_start] < start:
        index_start += 1
    while timestamps[index_end] > end:
        index_end -= 1
    return data[index_start:index_end + 1]


def weinberg_estimation(data, cst):
    '''
    Weinberg步长估算方法
    '''
    Amax = np.max(data)
    Amin = np.min(data)
    return cst * (Amax - Amin) ** (1 / 4)


def compute__adaptive_step_jerk_threshold(data, timestamps, jerk_threshold):
    '''
    自适应步伐检测方法
    使用jerk（加速度变化率）来检测步伐的起止时间
    '''
    jerk = np.diff(data)  # 计算加速度变化率（jerk）

    # 根据jerk值大于阈值来检测步伐
    step_crossings = []
    for i in range(1, len(jerk)):
        if jerk[i] > jerk_threshold and jerk[i - 1] <= jerk_threshold:
            step_crossings.append((timestamps[i], 'start'))
        if jerk[i] < -jerk_threshold and jerk[i - 1] >= -jerk_threshold:
            step_crossings.append((timestamps[i], 'end'))

    return step_crossings


def computeStepLength(data, timestamps, jerk_threshold, cst):
    '''
    结合自适应步伐检测与步长估算的方法
    '''
    # 使用自适应步伐检测方法
    step_crossings = compute__adaptive_step_jerk_threshold(data, timestamps, jerk_threshold)

    # 检测步伐对的数量
    steps = len(step_crossings) // 2

    weinberg_length_list = np.empty(steps)
    step_time_list = []  # 用于记录每个步伐的时间戳

    for i in range(steps):
        start = step_crossings[2 * i][0]
        end = step_crossings[2 * i + 1][0]

        # 提取该步伐的加速度数据
        step_data = get_data(data, timestamps, start, end)

        # 使用Weinberg方法估算步长
        weinberg_length_list[i] = weinberg_estimation(step_data, cst)

        # 添加步伐的结束时间戳
        step_time_list.append(end)

    return step_time_list, weinberg_length_list


def plot_step_length(data):
    '''
    绘制步长估算结果
    '''
    steps = len(data)
    distance = data.sum()
    plt.title("Step length estimation: {} steps, {} m".format(steps, round(distance, 2)))
    plt.xlabel("Step")
    plt.ylabel("Length [m]")
    plt.bar(range(1, steps + 1), data)
    #plt.savefig(GRAPH_PATH + 'steps_length')
    plt.show()


def compute_step_info(DATA_PATH, GRAPH_PATH, jerk_threshold=0.28, cst=0.45):
    '''
    完整的处理过程：加载数据、滤波、jerk阈值设置和步长估算
    返回每一步的时间戳和步长
    '''
    # 加载数据
    x_data, y_data, z_data, r_data, timestamps = cSteps.pull_data(DATA_PATH, 'Accelerometer')

    # 滤波
    order = 4
    fs = 50
    cutoff = 2
    r = cSteps.lowpass_filter(r_data, cutoff, fs, order)
    r = cSteps.data_without_mean(r)

    # 计算每一步的时间戳和步长
    step_time_list, weinberg_length_list = computeStepLength(r, timestamps, jerk_threshold, cst)
    # 绘制步长估算结果
    #plot_step_length(weinberg_length_list)
    # 返回步伐的时间戳和步长
    return step_time_list, weinberg_length_list


if __name__ == "__main__":
    # 计算每一步的时间戳和步长
    step_time_list, weinberg_length_list = compute_step_info(DATA_PATH, GRAPH_PATH)



