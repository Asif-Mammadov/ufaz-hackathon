import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import benchmarks
from datetime import datetime

class GSAOptim:
  def __init__(self, benchmark, pop_size=30, iter_num=100, dim=3, distance=10, imp_type=None, seed=None):
    if seed is not None:
      np.random.seed(seed)
    self.X = np.random.rand(pop_size, dim) * distance
    self.distance = distance
    self.benchmark = benchmark
    self.iter_num = iter_num
    self.dim = dim
    self.G = 0.9
    self.alpha = 0.1
    self.pop_size = pop_size
    self.v = np.zeros((self.pop_size, self.dim))
    self.lastbest = None
    self.curbest = None
    self.iter = 0
    self.avg = []
    self.median = []
    self.std = []
    self.bests = []
    self.imp_type = imp_type

  def fit(self):
    # Create fit values from the set
    self.fit_values = []
    for val in self.X:
      self.fit_values.append(self.benchmark(val))
    self.fit_values = np.array(self.fit_values)
    # print("X: ", self.X, "\n\nFit:", self.fit_values)

  def find_best(self):
    # Calculate the best - min of fit values
    return np.min(self.fit_values)

  def find_worst(self):
    # Calculate the worst - max of fit values
    return np.max(self.fit_values)

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
    self.f = np.full((self.pop_size, self.dim), None)
    values = self.fit_values.copy()
    values.sort(axis=0)

    # Iterate through particles and calculate force
    for i in range(self.pop_size):
      f = None
      for fit_value in self.fit_values:
        # print(np.where(values == fit_value))
        j = int(np.where(values == fit_value)[0])
        # Apply force formula and update value
        num = float(self.M[i] * self.M[j])
        denum = np.sqrt(np.sum((self.X[i] - self.X[j]) ** 2)) + np.finfo('float').eps
        val = self.G * (num / denum) * (self.X[j] - self.X[i])
        f = val if f is None else f + val

      self.f[i] = np.random.uniform(0, 1) * f

  def find_accel(self):
    # Calculate values for acceleration
    self.a = np.full((self.pop_size, self.dim), None)
    for i in range(self.pop_size):
      self.a[i] = 0 if self.M[i] == 0 else self.f[i] / self.M[i]

  def find_velocity(self):
    # Set values for particle velocity
    self.v = np.random.uniform(0, 1) * self.v + self.a

  def find_pos(self):
    # Set values for particle position
    self.X = self.X + self.v

  def improve(self):
    # Use the best-so-far methods to improve the algorithm
    best = self.find_best()
    if self.curbest is None or best < self.curbest:
      self.curbest = best
    else:
      if self.imp_type == 1:
        self.worst = self.curbest
      elif self.imp_type == 2:
        self.best = self.curbest
      elif self.imp_type == 3:
        index = int(np.random.rand() * len(self.fit_values))
        self.fit_values[index] = self.curbest

  def update_lastbest(self):
    # Save the last best solution before next iteration
    best = np.min(self.fit_values)
    i = int(np.where(self.fit_values == best)[0])
    benchmarked_x = self.benchmark(self.X[i])
    if self.lastbest is None or benchmarked_x < self.lastbest:
      self.lastbest = benchmarked_x

  def run_tmp(self):
    self.fit()
    self.best = self.find_best()
    self.worst = self.find_worst()
    self.find_G(0)
    self.find_mass()
    self.find_inertia_mass()
    self.find_force()
    self.find_accel()
    self.find_velocity()
    self.find_pos()
    if self.imp_type is not None:
      self.improve()

    self.iter += 1
    self.update_lastbest()
    self.avg.append(self.fit_values.mean())
    self.std.append(self.fit_values.std())
    self.median.append(np.median(self.fit_values))
    self.bests.append(self.lastbest)
    return self.X.T

  def run(self):
    print("GSA Optim")
    print("Pop_size:", self.pop_size)
    print("Iter num:", self.iter_num)
    print("Dimension:", self.dim)
    print("Distance", self.distance)
    while self.iter < self.iter_num:
      self.run_tmp()
    return self.avg, self.median, self.std, self.bests 

  def animate(self, save=False, save_count=100):
    def update(data):
      # points.set_xdata(data[0])
      # points.set_ydata(data[1])
      points.set_offsets(np.c_[data[0], data[1]])
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
      title = ax.text(0.5,0.85, "Shwefel. Iteration: " + str(self.iter), bbox={'facecolor':'w',  'pad':5},
                      transform=ax.transAxes, ha="center")
      return points

    def generate_rand_col():
      colors = []
      for _ in range(20):
        red = np.random.rand()
        green = np.random.rand()
        blue = np.random.rand()
        colors.append((red, green, blue))
      
      return colors

    def generate_points():
      while True:
        yield self.run_tmp()

    fig, ax = plt.subplots()
    XT = self.X.T
    colors = generate_rand_col()
    points = ax.scatter(XT[0], XT[1], marker='o', c=colors)

    ani = animation.FuncAnimation(fig, update, generate_points, interval=300, save_count=save_count, repeat=False)
    if save:

      # datetime object containing current date and time
      now = datetime.now()

      # dd/mm/YY H:M:S
      dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
      ani.save('animation' + dt_string + 'shwefel' + '.gif', fps=8, writer='imagemagick')
    else:
      plt.show()
