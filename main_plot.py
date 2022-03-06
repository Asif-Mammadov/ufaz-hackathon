import benchmarks
import matplotlib.pyplot as plt
from gsa import GSA


# X = np.random.rand(100, 2) * 300
a = GSA(benchmark=benchmarks.ackley, iter_num=20, distance=10, pop_size=20)

avg, bests = a.run()

fig, ax = plt.subplots(2)

ax[0].plot(avg, label='avg')
ax[1].plot(bests, label='best')
plt.show()