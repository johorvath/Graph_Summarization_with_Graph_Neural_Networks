"""
plot_val_acc_epoch.py: Load tensorboard summaries and parse them; store the results in dictionary
and plot chosen results from the dictionary
"""

import argparse

import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import pickle

from glob import glob

import matplotlib.pyplot as plt

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj (name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
 
parser = argparse.ArgumentParser()
parser.add_argument('--summary_dir', type=str,dest='sd', help='path to summary writer files')  
parser.add_argument('--save_file', type=str,dest='save_file', help='save_file to store parsed results')  
parser.add_argument('--models', nargs='+', dest='models', help='List of chosen models to plot')
args = parser.parse_args()

sd = args.sd

save_file = args.save_file
dirs = glob(os.path.join(sd, "*", ""))

summaries = {}

if os.path.isfile(save_file + ".pkl"):
    summaries = load_obj( save_file )
    print("Loaded",len(summaries), "summaries!")

# Iterate over all summaries in the directory
for d in dirs:
    dirname = os.path.basename(os.path.normpath(d))

    dirname = dirname.split("summary_")[1]
    
    event_acc = EventAccumulator(d)
    event_acc.Reload()
    try:
        w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy/val'))
        w_times = list(w_times)
        w_times = [ x - w_times[0] for x in w_times ]
        step_nums = list(step_nums)
        vals = list(vals)
        tmp = {"w_times":w_times,"step_nums":step_nums,"vals":vals}
        summaries[dirname] = tmp
    except KeyError:
        print("No key")
        continue

print(len(summaries))

save_obj(summaries, save_file)

if True:
    samples = args.models    
    
    for i in samples:
        plt.plot(summaries[i]["w_times"],summaries[i]["vals"],label=i)
    plt.xlabel('time[s]')
    # Set the y axis label of the current axis.
    plt.ylabel('validation accuracy')
    # Set a title of the current axes.
    plt.title('Compare training time for 30 epochs on validation accuracy')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()