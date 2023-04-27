import numpy as np
import os
import sys
from six.moves import cPickle as pkl
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
sns.set_theme()

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def read_log(file):

    data = open(file).read().split("\n")
    signal = None
    local_acc, global_acc = {"train": [], "test": []}, {"test": []}
    tmp1, tmp2 = [], []
    local = True

    for ii, i in enumerate(data):

        if "FedAvg TEST RESULTS:" in i:
            local = False
            continue

        if local:

            if ("TRAINING EPOCH:") in i:
                signal = "new epoch"
                continue

            if signal == "new epoch":

                signal = "continue epoch"
                x = float(i.split("acc: ")[1].strip().split("%")[1][-5:])
                if "train" in i:
                    tmp1.append(x)
                else:
                    tmp2.append(x)
                continue 

            if signal == "continue epoch":

                if len(i) <= 1:
                    signal = None 
                    if len(tmp1) >= 1:
                        local_acc["train"].append(tmp1)
                        local_acc["test"].append(tmp2)
                        tmp1 = []
                        tmp2 = []
                    continue

                x = float(i.split("acc: ")[1].strip().split("%")[1][-5:])
                if "train" in i:
                    tmp1.append(x)
                else:
                    tmp2.append(x)
                continue 

        else:
            if "accuracy" in i:
                global_acc["test"].append(float(i.split("'accuracy': ")[1].split("%")[0][1:]))

    return local_acc, global_acc

le = [1,2,4,6,8,10, 20, 25]
plt.figure(figsize=(9,6), dpi=500)

for ii, i in enumerate(le):

    local_acc, global_acc = read_log("../log{}.log".format(i))
    personal = np.stack(local_acc["test"])
    main = np.stack(global_acc["test"])


    plt.fill_between(np.arange(len(personal)), personal.mean(1)-personal.std(1), personal.mean(1)+personal.std(1), alpha=0.1, linewidth=0.1, color="C"+str(ii))
    plt.plot(personal.mean(1), "--", linewidth=1, color="C"+str(ii))
    plt.plot(main, label="le="+str(i), linewidth=1, color="C"+str(ii))


plt.legend()
plt.ylim([0, 100])
plt.xlabel("Communication round")
plt.ylabel("Test accuracy (%)")
# plt.semilogx()
plt.annotate(r"Mean $\pm \sigma$ local model", (100, 80))
plt.annotate("Global model", (100, 35))
plt.tight_layout()
plt.savefig("a.png")

# =================================

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def equipad(x, N):

    l = len(x)
    assert N >= l 
    add = (N-l)/l 
    y = []

    for i in range(len(x)):
        y.append(x[i])
        p = add - int(add)
        if np.random.random() >= p:
            n = np.floor(add)
        else:
            n = np.ceil(add)
        y.extend([x[i]]*int(n))

    return np.stack(y)

le = [1, 2, 4, 6, 8, 10, 20, 25]
plt.figure(figsize=(9,6), dpi=500)

for ii, i in enumerate(le):

    local_acc, global_acc = read_log("../log{}.log".format(i))
    personal = np.stack(local_acc["test"])
    main = np.stack(global_acc["test"])

    # delta = main - personal.min(1)
    # delta = equipad(delta, 450)
    # delta = running_mean(delta, 50)
    # plt.plot(delta, label="le = "+str(i), alpha=0.8, color="C"+str(ii))

    # delta = main - personal.max(1)
    # delta = equipad(delta, 450)
    # delta = running_mean(delta, 50)
    # plt.plot(delta, "--", alpha=0.8, color="C"+str(ii))

    delta = personal.mean(1) - main
    delta_u = personal.mean(1) - main + personal.std(1)
    delta_l = personal.mean(1) - main - personal.std(1)

    delta = equipad(delta, 500)
    delta = running_mean(delta, 50)

    delta_u = equipad(delta_u, 500)
    delta_u = running_mean(delta_u, 50)

    delta_l = equipad(delta_l, 500)
    delta_l = running_mean(delta_l, 50)

    L = 400

    plt.plot(delta[:L], label="le = "+str(i), alpha=0.8, color="C"+str(ii))
    plt.fill_between(np.arange(len(delta_l[:L])), delta_l[:L], delta_u[:L], linewidth=.1, alpha=.1, color="C"+str(ii))

plt.legend()
plt.xlabel("Total epochs")
plt.ylabel("Delta accuracy (%)")
# plt.semilogx()
plt.tight_layout()
plt.savefig("b.png")

# ============================================

plt.figure(figsize=(9,6), dpi=500)
acc, acc1 = [], []

for ii, i in enumerate(le):

    local_acc, global_acc = read_log("../log{}.log".format(i))
    personal = np.stack(local_acc["test"])
    main = np.stack(global_acc["test"])
    acc.append(main[-1])
    acc1.append(personal[-1])

acc = np.stack(acc)
acc1 = np.stack(acc1)
plt.plot(le, acc, "-o", label="Global")
plt.fill_between(le, acc1.mean(1)-acc1.std(1), acc1.mean(1)+acc1.std(1), alpha=0.1, linewidth=0.1, color="red")
plt.plot(le, acc1.mean(1), "-o", label="Local", color="red")

plt.legend()
plt.xlabel("# Local epochs")
plt.ylabel("Convergence test accuracy (%)")
# plt.semilogx()
plt.ylim([0, 100])
plt.tight_layout()
plt.savefig("c.png")

# ==========================================

plt.figure(figsize=(9,6), dpi=500)

for ii, i in enumerate(le):

    local_acc, global_acc = read_log("../log{}.log".format(i))
    personal = np.stack(local_acc["test"])
    main = np.stack(global_acc["test"])

    delta = personal.mean(1) - main
    delta_u = personal.mean(1) - main + personal.std(1)
    delta_l = personal.mean(1) - main - personal.std(1)

    L = 400

    plt.plot(delta[:L], label="le = "+str(i), alpha=0.8, color="C"+str(ii))
    plt.fill_between(np.arange(len(delta_l[:L])), delta_l[:L], delta_u[:L], linewidth=.1, alpha=.1, color="C"+str(ii))

plt.legend()
plt.xlabel("Communcation rounds")
plt.ylabel("Delta accuracy (%)")
# plt.semilogx()
plt.tight_layout()
plt.savefig("d.png")
