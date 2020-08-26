import matplotlib.pyplot as plt 
import fastdtw
import numpy as np 
import multiprocessing as mp  
from time import time

data = []
init_class = []

def plot_traces(n, data, indices):
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

def dtw_classification(data, n_classes, n_frames, rep_trc_IDs):
    start = time()

    init_class = np.zeros((n_classes, n_frames))
    
    for i in range(n_classes):
        init_class[i] = scaled_data_cy5[rep_trc_IDs[i]]

    labels = []
    for i in range(len(data)):
        trc = data[i]
        min_dist = 1e9
        curr_label = 0
        for j in range(n_classes):
            dist = dtw(trc, init_class[j])
            if dist < min_dist:
                min_dist = dist
                curr_label = j
        labels.append(curr_label)

    print("Time passed: " + str(time() - start))
    return labels

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
