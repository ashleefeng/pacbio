from sys import executable, argv
from subprocess import check_output
from PyQt5.QtWidgets import QFileDialog, QApplication
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import h5py

"""
Utility functions for analyzing pacbio data

By XA Feng xfeng17@jhu.edu Feb 16, 2019
"""

def gui_fname(directory='E:/Ashlee/PacBio/'):
    """Open a file dialog, starting in the given directory, and return
    the chosen filename
	source: https://stackoverflow.com/questions/20790926/ipython-notebook-open-select-file-with-gui-qt-dialog
    """
    # run this exact file in a separate process, and grab the result
    file = check_output([executable, __file__, directory])
    return file.strip()

def load_data(sample_prefix):
    """
    Load pacbio dataset

    input: string, prefix for pacbio files 
    eg. ~/data/m181207_211324_42131_c000468682559900001500000112312060_s1_p0
    
    output: np.array((n_traces, 4, n_frames)), raw_traces
    """

    trc_filename = sample_prefix + '.trc.h5'
    mcd_filename = sample_prefix + '.mcd.h5'
    upd_filename = sample_prefix + '.upd.h5'
    meta_filename = sample_prefix + '.metadata.xml'

    trc_file = h5py.File(trc_filename, 'r')
    # mcd_file = h5py.File(mcd_filename, 'r')
    # upd_file = h5py.File(upd_filename, 'r')

    # Load data in trace file
    dset = trc_file['TraceData']
    raw_traces = dset['Traces']
    n_frames = raw_traces.shape[2]
    decode = dset['Codec']['Decode']
    decode_array = np.array(decode)

    # # Load data in upd file
    # upd_TD = upd_file['TraceData']
    # hole_xy_plot = upd_TD['HoleXYPlot'] # (x,y) coordinates for each well
    # hole_status = upd_TD['HoleStatus'] # fiducials

    # Extract frame rate from metadata
    meta_tree = ET.parse(meta_filename)
    root = meta_tree.getroot()
    for keyval in root.iter('{http://pacificbiosciences.com/PAP/Metadata.xsd}KeyValue'):
        if keyval.attrib['key'] == 'MovieFPS':
            fps = int(keyval.text)
            frame_rate = 1.0/fps # seconds
            break

    # For plotting
    time_axis = np.arange(n_frames)*frame_rate
    
    return raw_traces, decode_array, time_axis


def plot_one_trace(data, traceID, time_axis, foi, colors):
    
    trc = data[traceID]
    shape = data.shape
    
    if len(shape) == 2:
        plt.plot(time_axis[foi], trc[foi])
    
    else:
        n_channels = data.shape[1]
        for j in range(n_channels):
            plt.plot(time_axis[foi], trc[j][foi], color=colors[j], linewidth=0.5)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.title('Trace ' + str(traceID))

def plot_one_trace_bgs(data, traceID, time_axis, background, foi, colors, n_frames):
#     plt.figure(figsize=(15, 2))
    trc = data[traceID, :, foi]
    corrected = np.zeros((4, n_frames))
    for j in [0, 2]:
        corrected[j] = trc[j] - bg[traceID, j]
        if j > 0:
            corrected[j] = corrected[j] - leakage[0, j] * corrected[0]
        if j > 1:
            corrected[j] = corrected[j] - leakage[1, j] * corrected[1]
        if j > 2:
            corrected[j] = corrected[j] - leakage[2, j] * corrected[2]
        plt.plot(time_axis[foi], corrected[j] , c=colors[j], linewidth=.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity')
        plt.title('Trace ' + str(traceID))

def plot_traces(n, data, indices, time_axis, foi, colors):
    plt.figure(figsize=(15, n*2))
    
    for i in range(n):
        plt.subplot(n, 1, i+1)
        plot_one_trace(data, indices[i], time_axis, foi, colors)
    
    plt.subplots_adjust(hspace=0.9)

def plot_traces_caller(n, toi, indices, time_axis, foi, colors, start, end, dyes, save=True):
    plot_traces(n, toi, indices[start:end], time_axis, foi, colors)
    filename = "%s_%d-%d_most_%s_anticorrelated.png" %(sample_prefix, start, end, dyes)
    if save:
        plt.savefig(filename, dpi=200)

def decode_and_plot(time_axis, traces, traceID, decode_array, colors, foi, background, gamma, leakage, channels, lasers):
#     channels = range(4)
    trace = decode_array[traces[traceID, :, foi]]
    plt.figure(figsize=(13,4))
    
    for i in channels:
        trace[i] = trace[i] - background[traceID, i]
        trace[i] = trace[i] * gamma[i]
        for j in channels:
            trace[i] = trace[i] - trace[j] * leakage[j, i]
    
    for j in channels:
        plt.plot(time_axis[foi], trace[j], color=colors[j], linewidth=0.5, alpha=1)

    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.legend(lasers, bbox_to_anchor=(0.75, 0.5, 0.5, 0.5))
    
    # plt.figure(figsize=(13,4))
    # don = trace[0]
    # acc = trace[2]
    # fret = acc/(don + acc)
    # plt.plot(time_axis[foi], fret, linewidth=0.4)
    # plt.ylim([-0.1, 1.1])    
    # plt.xlabel('Time (s)')
    # plt.ylabel('FRET efficiency')

if __name__ == "__main__":
    directory = argv[1]
    app = QApplication([directory])
    fname = QFileDialog.getOpenFileName(None, "Select a file...", 
            directory, filter="All files (*)")
    print(fname[0])