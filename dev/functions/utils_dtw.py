import matplotlib.pyplot as plt 
from fastdtw import fastdtw
import numpy as np 
import multiprocessing as mp  
from time import time

data = []
init_class = []
traces = []

def smoothen_cy5_single(traceID):

    trc = traces[traceID, 2, :]
    n_frames = traces.shape[2]
    trc_stack = np.zeros((3, n_frames-2))
    trc_stack[0] = trc[0:-2]
    trc_stack[1] = trc[1:-1]
    trc_stack[2] = trc[2:]
    smoothened = np.mean(trc_stack, axis=0)
    return smoothened

def smoothen_cy5(data):

    """
    Average every 3 frames of Cy5 emission

    input: np.array((n_traces, 4, n_frames))

    output: np.array((n_traces)

    """

    global traces 
    traces = data

    start = time()

    n_traces = traces.shape[0]
    n_frames = traces.shape[2]

    smoothened = np.zeros((n_traces, n_frames-2))
    pool = mp.Pool(mp.cpu_count())
    smoothened = np.array(pool.map(smoothen_cy5_single, iter(range(n_traces))))

    print("Time passed: " + str(time() - start))

    return smoothened

def normalize(traces):
    """
    Zero-center the traces, then scale to (0, 1) range

    input: np.array((n_traces, n_frames)), data matrix

    output: np.array((n_traces, n_frames)), scaled cy5 data matrix (:, 2, :)
    """

    start = time()

    avg_intens = np.mean(traces, axis=1)
    n_traces = traces.shape[0]
    n_frames = traces.shape[1]

    # zero-center the traces
    centered_cy5 = np.zeros((n_traces, n_frames))
    for i in range(n_traces):
        centered_cy5[i] = traces[i, :] - avg_intens[i]

    scaled_data_cy5 = np.zeros((n_traces, n_frames))

    for i in range(n_traces):
        cy5_trc = centered_cy5[i, :]
        cy5_min = cy5_trc.min()
        cy5_max = cy5_trc.max()
        if cy5_min == cy5_max:
            scaled_data_cy5[i] = np.ones(cy5_trc.shape) 
        else:
            scaled_data_cy5[i] = (cy5_trc - cy5_min) / (cy5_max - cy5_min)

    print("Time passed: " + str(time() - start))

    return scaled_data_cy5

def plot_traces(n, data, indices, time_axis):
    plt.figure(figsize=(15, n*1.2))
    if n % 2 == 1:
        n = n-1
    for i in range(n):
        plt.subplot(int(n/2), 2, i+1)
        trc = data[indices[i]]
        if len(data.shape) == 2:
            plt.plot(time_axis, trc)
        else:
            for j in range(data.shape[1]):
                plt.plot(time_axis, trc[j], color=colors[j])

        plt.xlabel('Time (s)')
        plt.ylabel('Intensity')
        plt.title('Trace ' + str(i))
        
        i = i + 1
    plt.subplots_adjust(hspace=0.7)

def dtw(a1, a2):
    return fastdtw(a1.T, a2.T)[0]

def dtw_single(m):

    trc = data[m]
    min_dist = 1e9
    curr_label = 0
    for j in range(n_classes):
        dist = dtw(trc, init_class[j])
        if dist < min_dist:
            min_dist = dist
            curr_label = j
    return curr_label

def dtw_classification_parallel(data_curr, n_classes_curr, n_frames, rep_trc_IDs):

    start = time()
    
    global init_class
    global n_classes
    global data 
    data = data_curr
    n_classes = n_classes_curr
    init_class = np.zeros((n_classes, n_frames))

    for i in range(n_classes):
        init_class[i] = data[rep_trc_IDs[i]]
        
    labels = []
    pool = mp.Pool(mp.cpu_count())
    labels = pool.map(dtw_single, iter(range(len(data))))

    print("Time passed: " + str(time() - start))
    return labels

def class_extractor(labels, n_classes):
    cls_members = {}
    for i in range(len(labels)):
        cls = labels[i]
        if cls in cls_members.keys():
            cls_members[cls].append(i)
        else:
            cls_members[cls] = [i]
    return cls_members
