import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.animation as animation


def animated_plot(x1, x2, fx):
    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(x1, x2, c=fx, cmap="viridis", edgecolor="k")
    line, = ax.plot([], [], 'r-')
    points = ax.scatter([], [], s=0.1)

    def update(i):
        line.set_data(x1[:i], x2[:i])
        points.set_offsets(np.c_[x1[:i], x2[:i]])
        return line, points

    ani = animation.FuncAnimation(fig, update, frames=len(x1), interval=50)

def data_extracter(algo_num, instance,max_index=0,method=""):
    '''return x1, x2, fx'''
    path = f"Ch4/1/Algo_{algo_num}/{instance}/history.0.txt"
    data = np.loadtxt(path)
    if method == "simple":
        return data[:, -1]
    if len(data[0])==4:
        return data[:, :-1][:,0], data[:, :-1][:,1], data[:, :-1][:,2], data[:, -1]
    if len(data[0])!=3:
        print(f"ERROR THE DIMENSION DOES NOT WORK, PLEASE CHECK THE DIMENSION OF x.")
    if max_index != 0:
        return data[:max_index, :-1][:,0], data[:max_index, :-1][:,1], data[:max_index, -1]
    
    return data[:, :-1][:,0], data[:, :-1][:,1], data[:, -1]

def Trajectory_plot(x1, x2, fx):
    plt.scatter(x1, x2, c=fx, cmap="viridis", edgecolor="k")
    plt.plot(x1, x2)
    return

def Convergence_plot(fx):
    fbest = np.array([fx[0]])
    iter_num_fbest = np.array([1])

    f0 = fx[0]
    for index, value in enumerate(fx):
        if value < f0:
            fbest = np.append(fbest, value)
            iter_num_fbest = np.append(iter_num_fbest, index+1)
            f0 = value
    return iter_num_fbest, fbest

ab = False
if ab:
    for inst in [19,37]:
        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(5,12))

        instance = inst

        for algo in [1,2]:
            x1, x2, fx = data_extracter(algo, instance)
            ax1.scatter(x1, x2, c=fx, cmap="viridis", edgecolor="k")
            ax1.plot(x1, x2, label=f'Algo_{algo}')
            ax1.set_xlabel('x_1')
            ax1.set_ylabel('x_2')
            ax1.set_title('Trajectory plot')
            ax1.legend()


            iter_num_fbest, fbest = Convergence_plot(fx)
            ax2.plot(iter_num_fbest, fbest, '.-', markersize=8, label=f'Algo_{algo}')
            ax2.set_xlabel('Number of function evaluations')
            ax2.set_ylabel('Best objective function value')
            ax2.set_title('Convergence plot')
            ax2.legend()


        plt.tight_layout()
        plt.show()


#%%

if False:
    plt.figure()
    for algo in [1,2]:
        x1, x2, x3, fx = data_extracter(algo,52)
        iter_num_fbest, fbest = Convergence_plot(fx)
        plt.plot(iter_num_fbest, fbest, '.-', markersize=8, label=f'Algo_{algo}')


    plt.xlabel('Number of function evaluations')
    plt.ylabel('Best objective function value')
    plt.title('Convergence plot')
    plt.legend()
    plt.show()

#%%

# fig, (r1, r2, l1, l2) = plt.subplots(2, 2, figsize=(6.5,12))


def facc(f_val, f0, fbest):
    return (f_val-f0) / (fbest - f0)


tau = [0.05, 0.01]

algo = [1,2]
instance = [19,37]

Nap = np.zeros((len(algo), len(instance)))
Tap = np.zeros((len(algo), len(instance)))
rap = np.zeros((len(algo), len(instance)))

def performance_profil(tau):
    
    # Best value
    fbest = np.zeros((len(instance), len(algo)))
    for ind_p, p in enumerate(instance):
        for ind_a, a in enumerate(algo):
            fx = data_extracter(a, p, method="simple")
            fbest[ind_p][ind_a] = np.min(fx)

    for ind_a, a in enumerate(algo):
        for ind_p, p in enumerate(instance):
            fx = data_extracter(a, p, method="simple")

            

            for i, f_val in enumerate(fx):
                if facc(f_val,fx[0],np.min(fbest[ind_p])) >= (1-tau):
                    Nap[ind_a, ind_p] = i
                    Tap[ind_a, ind_p] = 1
                    
                if Tap[ind_a, ind_p] == 1:
                    rap[ind_a, ind_p] = Nap[ind_a, ind_p] / np.min(Nap)
                else:
                    rap[ind_a, ind_p] = 1e6

    return


