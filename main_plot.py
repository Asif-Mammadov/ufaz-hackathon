from distutils import dist
from matplotlib import markers
import benchmarks
import matplotlib.pyplot as plt
from gsa import GSA
from gsa_optim import GSAOptim

N_RUN = 5
mean_arr = {}
best_arr = {}
gsa = {}


best1_arr = []
best2_arr = []
benchmark_list = {"ackley": benchmarks.ackley, 
"rastrigin" : benchmarks.rastrigin, 
"rosenbrock" : benchmarks.rosenbrock, 
"schwefel" : benchmarks.schwefel}

pop_size = 30
iter_num = 100
dim = 30
seed = 40
distance = 10
import numpy as np

for i in range(100):
  for benchmark_name, benchmark in benchmark_list.items():
    gsa = {
      "GSA" : GSAOptim(benchmark=benchmark, pop_size=pop_size, iter_num=iter_num, dim=dim, distance=distance),
      "GSA-optim-1" : GSAOptim(benchmark=benchmark, imp_type=1, pop_size=pop_size, iter_num=iter_num, dim=dim, distance=distance),
      "GSA-optim-2" : GSAOptim(benchmark=benchmark, imp_type=2, pop_size=pop_size, iter_num=iter_num, dim=dim, distance=distance),
      "GSA-optim-3" : GSAOptim(benchmark=benchmark, imp_type=3, pop_size=pop_size, iter_num=iter_num, dim=dim, distance=distance),
    }

    for key in gsa:
      mean_arr[key] = []
      best_arr[key] = []

    for _ in range(N_RUN):
      print("---------------\n")
      print("Sprint:",  _)
      gsa = {
        "GSA" : GSAOptim(benchmark=benchmark, pop_size=pop_size, iter_num=iter_num, dim=dim, distance=dim),
        "GSA-optim-1" : GSAOptim(benchmark=benchmark, imp_type=1, pop_size=pop_size, iter_num=iter_num, dim=dim, distance=dim),
        "GSA-optim-2" : GSAOptim(benchmark=benchmark, imp_type=2, pop_size=pop_size, iter_num=iter_num, dim=dim, distance=dim),
        "GSA-optim-3" : GSAOptim(benchmark=benchmark, imp_type=3, pop_size=pop_size, iter_num=iter_num, dim=dim, distance=dim),
      }
      if len(mean_arr['GSA']) == 0:
        for key in gsa:
          avr, best = np.array(gsa[key].run())
          mean_arr[key] = avr.copy()
          best_arr[key] = best.copy()
      else:
        for key in gsa:
          avr, best = np.array(gsa[key].run())
          mean_arr[key] += avr.copy()
          best_arr[key] += best.copy()
      for key in gsa:
        mean_arr[key] /= 10
        best_arr[key] /= 10

    fig, ax = plt.subplots(2, sharex=True)
    ax[1].set_xlabel("N iterations")
    ax[0].set_ylabel("Average fit")
    ax[0].set_title("GSA Evaluation (" + benchmark_name + ").")
    ax[1].set_ylabel("Best fit")
    for key in gsa.keys():
      ax[0].plot(mean_arr[key], label=key)

    for key in gsa.keys():
      ax[1].plot(best_arr[key])

    fig.legend()
    from datetime import datetime

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")

    plt.savefig("image/" + "gsa_" + dt_string + benchmark_name)