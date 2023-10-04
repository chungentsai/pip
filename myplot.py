import sys, os
from math import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

markers = [
     "+", "1", "x", "*", "P", "v", "^", "<", ">", "s"
     ]
markers = cycle(markers)

linecolors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:blue','tab:gray', 'tab:olive', 'tab:cyan']
linecolors = cycle(linecolors)

filename = sys.argv[1]


def read_records(filename):
    approx_opt = inf

    results = dict()
    f = open(filename, "r")
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n","")
    n_line = len(lines)

    d = int(lines[0])
    true_signal = np.zeros(d)
    for j in range(d):
        true_signal[j] = float(lines[j+1])

    i = d+1
    while i < n_line:
        alg_name = lines[i]
        n_epoch = int(lines[i+1])
        n_rate = int(lines[i+2])
        n_data = n_epoch * n_rate

        results[alg_name] = dict()
        results[alg_name]["n_epoch"] = np.zeros(n_data)
        results[alg_name]["elapsed_time"] = np.zeros(n_data)
        results[alg_name]["opt_error"] = np.zeros(n_data)
        results[alg_name]["normalized_l2"] = np.zeros(n_data)
        results[alg_name]["signal"] = np.zeros(d)
        results[alg_name]["marker"] = next(markers)
        results[alg_name]["linecolor"] = next(linecolors)
        
        i = i + 3
        for j in range(n_data):
            data = lines[i + j].split("\t")
            results[alg_name]["n_epoch"][j] = float(data[0])
            results[alg_name]["elapsed_time"][j] = float(data[1])
            results[alg_name]["opt_error"][j] = float(data[2])
            results[alg_name]["normalized_l2"][j] =  float(data[3])
            approx_opt = min(results[alg_name]["opt_error"][j], approx_opt)
        i = i + n_data + 1

        for j in range(d):
            results[alg_name]["signal"][j] = float(lines[i+j])

        i = i + d

    for alg_name in results.keys():
        results[alg_name]["opt_error"] = results[alg_name]["opt_error"] - approx_opt

    return results, true_signal


def main():
    results, true_signal = read_records(filename)

    directory = filename.replace("records", "figures") + "/"
    if not os.path.isdir(directory):
        os.makedirs(directory)

    algs = results.keys()

    plt.figure(1)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    algs = ["BPG", "Frank-Wolfe", "EM", "SLBOMD", "SSB", "d-sample LB-SDA", "1-sample LB-SDA", "EMD", "SPDHG"]
    for alg_name in algs:
        plt.semilogy(results[alg_name]["n_epoch"], results[alg_name]["opt_error"], label=alg_name, marker=results[alg_name]["marker"], markevery=0.1, linewidth=1, color=results[alg_name]["linecolor"])
    plt.xlabel("Number of epochs")
    plt.ylabel("Approximate optimization error")
    plt.xlim((0, 200))
    #plt.ylim([3e3, 5e3])
    plt.legend()
    plt.savefig(directory + "epoch-error.png", dpi=300, bbox_inches="tight")

    plt.figure(2)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    algs = ["BPG", "Frank-Wolfe", "EM", "SSB", "SLBOMD", "SPDHG", "d-sample LB-SDA", "EMD", "1-sample LB-SDA"]
    for alg_name in algs:
        plt.semilogy(results[alg_name]["n_epoch"], results[alg_name]["normalized_l2"], label=alg_name, marker=results[alg_name]["marker"], markevery=0.1, linewidth=1, color=results[alg_name]["linecolor"])
    plt.xlabel("Number of epochs")
    plt.ylabel("Normalized estimation error")
    plt.xlim((0, 200))
    #plt.ylim([1e-1, 1e1])
    plt.legend()
    plt.savefig(directory + "epoch-distance.png", dpi=300, bbox_inches="tight")

    plt.figure(3)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    algs = ["BPG", "SLBOMD", "SSB", "EM", "d-sample LB-SDA", "SPDHG","Frank-Wolfe", "1-sample LB-SDA", "EMD"]
    for alg_name in algs:
        plt.loglog(results[alg_name]["elapsed_time"], results[alg_name]["opt_error"], label=alg_name, marker=results[alg_name]["marker"], markevery=0.1, linewidth=1, color=results[alg_name]["linecolor"])
    plt.xlabel("Elapsed time (seconds)")
    plt.ylabel("Approximate optimization error")
    plt.ylim([1, 2e3])
    plt.legend()
    plt.savefig(directory + "time-error.png", dpi=300, bbox_inches="tight")

    plt.figure(4)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    algs = ["BPG", "SLBOMD", "SSB", "EM", "SPDHG", "d-sample LB-SDA", "Frank-Wolfe", "1-sample LB-SDA", "EMD"]
    for alg_name in algs:
        plt.loglog(results[alg_name]["elapsed_time"], results[alg_name]["normalized_l2"], label=alg_name, marker=results[alg_name]["marker"], markevery=0.1, linewidth=1, color=results[alg_name]["linecolor"])
    plt.xlabel("Elapsed time (seconds)")
    plt.ylabel("Normalized estimation error")
    #plt.ylim([1e-1, 1e1])
    plt.legend()
    plt.savefig(directory + "time-distance.png", dpi=300, bbox_inches="tight")

    plt.figure(5)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    algs = ["d-sample LB-SDA", "1-sample LB-SDA", "EMD"]
    for alg_name in algs:
        plt.plot(results[alg_name]["signal"], label=alg_name, marker=results[alg_name]["marker"], markevery=0.2, linewidth=1, color=results[alg_name]["linecolor"])
    plt.plot(true_signal, label="true", linewidth=1, color='tab:red')
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    #plt.ylim([0, 15000])
    plt.legend()
    plt.savefig(directory + "signal.png", dpi=300, bbox_inches="tight")

    #plt.show()

if __name__ == "__main__":
    main()