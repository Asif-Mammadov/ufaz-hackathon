import benchmarks
from gsa_optim import GSAOptim

# X = np.random.rand(100, 2) * 300
a = GSAOptim(benchmark=benchmarks.schwefel, iter_num=20, distance=10, pop_size=20)
# avg, bests = a.run()

# fig, ax = plt.subplots(2)

# ax[0].plot(avg, label='avg')
# ax[1].plot(bests, label='best')
# plt.show()
a.animate(save=True)
