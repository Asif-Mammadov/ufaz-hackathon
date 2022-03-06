from matplotlib.collections import BrokenBarHCollection
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

class GSA:
  def __init__(self, X, benchmark, iter_num=10, dim=2):
    self.X = X
    self.benchmark = benchmark
    self.iter_num = iter_num
    self.dim = dim
    self.G = 0.9
    self.alpha = 0.1
    self.pop_size = len(X)
    self.v = np.zeros((self.pop_size, self.dim))
    self.lastbest = None
    self.iter = 0
    self.avg = []
    self.bests = []

  def fit(self):
    # Create fit values from the set
    self.fit_values = []
    for val in self.X:
      self.fit_values.append(self.benchmark(val))
    self.fit_values = np.array(self.fit_values)
    print("X: ", self.X, "\n\nFit_values: ", self.fit_values)

  def find_best(self):
    # Calculate the best - min of fit values
    self.best = np.min(self.fit_values)
    return self.best

  def find_worst(self):
    # Calculate the worst - max of fit values
    self.worst = np.max(self.fit_values)
    return self.worst

  def find_G(self, t):
    # Calculate the gravitational constant
    self.G =  self.G * np.exp(-1 * self.alpha * t/self.iter_num)

  def find_mass(self):
    # Calculate mass for each particle
    self.m = (self.fit_values - self.worst) / (self.best - self.worst)

  def find_inertia_mass(self):
    # Calculate inertia mass for each particle
    self.M = self.m / (np.sum(self.m))

  def find_force(self):
    self.f = np.full((self.pop_size, 2), None)
    values = self.fit_values.copy()
    values.sort(axis=0)

    # Iterate through particles and calculate force
    for i in range(self.pop_size):
      f = None
      for fit_value in self.fit_values:
        j = int(np.where(values == fit_value)[0])
        # Apply force formula and update value
        num = float(self.M[i] * self.M[j])
        denum = np.sqrt(np.sum((self.X[i] - self.X[j]) ** 2)) + np.finfo('float').eps
        val = self.G * (num / denum) * (self.X[j] - self.X[i])
        f = val if f is None else f + val

      self.f[i] = np.random.uniform(0, 1) * f

  def find_accel(self):
    # Calculate values for acceleration
    self.a = np.full((self.pop_size, 2), None)
    for i in range(self.pop_size):
      self.a[i] = 0 if self.M[i] == 0 else self.f[i] / self.M[i]

  def find_velocity(self):
    # Set values for particle velocity
    self.v = np.random.uniform(0, 1) * self.v + self.a

  def find_pos(self):
    # Set values for particle position
    self.X = self.X + self.v
    
  def update_lastbest(self):
    # Save the last best solution before next iteration
    best = np.min(self.fit_values)
    i = int(np.where(self.fit_values == best)[0])
    benchmarked_x = self.benchmark(self.X[i])
    if self.lastbest is None or benchmarked_x < self.lastbest:
      self.lastbest = benchmarked_x

  def run_tmp(self):
    self.fit()
    self.find_best()
    self.find_worst()
    self.find_G(0)
    self.find_mass()
    self.find_inertia_mass()
    self.find_force()
    self.find_accel()
    self.find_velocity()
    self.find_pos()

    self.iter += 1
    self.update_lastbest()
    self.avg.append(self.fit_values.mean())
    self.bests.append(self.lastbest)
    return self.X.T

  def run(self):
    while self.iter < self.iter_num:
      self.run_tmp()

    # plt.plot(self.bests, '-o')    
    plt.plot(self.bests, '-o')    
    plt.show()
    #   XT = self.X.T
    #   self.fit(rosenbrock)
    #   self.find_best()
    #   self.find_worst()
    #   self.find_G(0)
    #   self.find_mass()
    #   self.find_inertia_mass()
    #   self.find_force()
    #   self.find_accel()
    #   self.find_velocity()
    #   self.find_pos()
    #   self.update_lastbest()

  def animate(self, save=False, save_count=100):
    def update(data):
      points.set_xdata(data[0])
      points.set_ydata(data[1])
      minx = data[0].min()
      maxx = data[0].max()
      miny = data[1].min()
      maxy = data[1].max()

      # Set limiting points for x and y axes
      xlow = minx * 1.5 if minx < 0 else minx * 0.5
      xhigh = maxx * 0.5 if maxx < 0 else maxx * 1.5
      ylow = miny * 1.5 if miny < 0 else miny * 0.5
      yhigh = maxy * 0.5 if maxy < 0 else maxy * 1.5
      
      ax.set_xlim(xlow, xhigh)
      ax.set_ylim(ylow, yhigh)
      title = ax.text(0.5,0.85, "iter " + str(self.iter), bbox={'facecolor':'w',  'pad':5},
                      transform=ax.transAxes, ha="center")
      return points

    def generate_points():
      while True:
        yield self.run_tmp()
    
    fig, ax = plt.subplots()
    XT = self.X.T
    points, = ax.plot(XT[0], XT[1], 'o')

    ani = animation.FuncAnimation(fig, update, generate_points, interval=300, save_count=save_count, repeat=False)
    if save:
      ani.save('animation.gif', fps=4, writer='imagemagick')
    else:
      plt.show()

# Benchmarking method
def rosenbrock(args):
  x, y = args
  a = 1
  b = 100
  return (a - x) ** 2 + b * (y - x ** 2) ** 2

X = np.random.rand(100, 2) * 300
a = GSA(X, benchmark=rosenbrock, iter_num=100)
# a.run()
a.animate(save=True)


