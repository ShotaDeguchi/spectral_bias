"""
********************************************************************************
main file to execute your program
********************************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt

from config_gpu import config_gpu
from functions import *
from dnn import *

def main():
    # gpu configuration
    config_gpu(gpu_flg = 1)

    # params
    f_in   = 1
    f_out  = 1
    f_hid  = 30
    depth  = 5
    lr     = 1e-3
    opt    = "Adam"
    f_scl  = "minmax"
    d_type = "float32"
    r_seed = 1234
    n_epc  = int(5e4)

    # problem setup
    p_id = 1
    dt   = 1e-3       # sampling frequency
    nt   = int(1e3)   # samples
    t    = np.arange(0., nt * dt, dt)
    t_tf = tf.convert_to_tensor(t.reshape(-1, 1), dtype=d_type)

    if p_id == 0:
        raise NotImplementedError(">>>>> p_id")
    elif p_id == 1:
        y = func1(t)
        y_tf = func1_tf(t_tf)
        path = "./figures/problem_1/"
    elif p_id == 2:
        y = func2(t)
        y_tf = func2_tf(t_tf)
        path = "./figures/problem_2/"
    else:
        raise NotImplementedError(">>>>> p_id")

    F = np.fft.fft(y)
    freq = np.fft.fftfreq(y.size, dt)
    A = np.abs(F / (nt / 2))

    # dnn operations
    w_init = "Glorot"
    b_init = "zeros"
    act    = "tanh"
    model = DNN(
        t_tf, y_tf, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr, opt, f_scl, 
        d_type, r_seed
    )

    for n in range(n_epc):
        with tf.device("/device:GPU:0"):
            model.train_step()

        if n % int(n_epc / 10) == 0:
            y_infer = model.infer(t_tf)
            print("epoch: %d / %d, progress: %.3f" 
                % (n, n_epc, (n / n_epc * 100)))

            plt.figure(figsize=(8, 8))

            plt.subplot(2, 1, 1)
            plt.plot(t, y, label="function", alpha=.3, linestyle="-", lw = 3, c="k")
            plt.plot(t_tf, y_infer, label="dnn", alpha=.7, linestyle="--")
            plt.xlabel("t")
            plt.ylabel("f")
            plt.title("epoch:" + str(n) + "/" + str(n_epc))
            plt.ylim(-1.55, 1.55)
            plt.grid(alpha=.5)
            plt.legend(loc="upper right")

            F_infer = np.fft.fft(y_infer.numpy()[:,0])
            A_infer = np.abs(F_infer / (nt / 2))

            plt.subplot(2, 1, 2)
            plt.plot(freq[1:int(nt / 2)], A[1:int(nt / 2)], label="function", alpha=.3, linestyle="-", lw = 3, c="k")
            plt.plot(freq[1:int(nt / 2)], A_infer[1:int(nt / 2)], label="dnn", alpha=.7, linestyle="--")
            plt.xlabel("frequency")
            plt.ylabel("amplitude")
            plt.ylim(-.05, .55)
            plt.grid(alpha=.5)
            plt.legend(loc="upper right")

            plt.savefig(path + "spectral_" + str(n) + ".png")
            plt.clf()
            plt.close()

if __name__ == "__main__":
    main()

