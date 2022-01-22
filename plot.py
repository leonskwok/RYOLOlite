
import scipy.signal
import os
import numpy as np
import matplotlib.pyplot as plt


def loss_plot():
  traindata1 = np.loadtxt('logs/RTiny_GhostBottle_all_G212-21_09-55/loss.txt')[:, 0]
  traindata2 = np.loadtxt('logs/RTiny12-25_14-55/loss.txt')[:, 0]
#   valdata = np.loadtxt(os.path.join(savepath, 'val_loss.txt'))
  iters = range(len(traindata1))

  plt.figure()
  # plt.plot(iters, traindata1, 'red', linewidth=2, label='train loss')
  # plt.plot(iters, traindata2, 'blue', linewidth=2, label='train loss')
#   plt.plot(iters, valdata, 'coral', linewidth=2, label='val loss')
  try:
      if len(traindata1) < 25:
          num = 5
      else:
          num = 15

      plt.plot(iters, scipy.signal.savgol_filter(traindata1, num, 3),
               'green', linestyle='--', linewidth=2, label='smooth train loss')
      plt.plot(iters, scipy.signal.savgol_filter(traindata2, num, 3),
               '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
  except:
      pass

  plt.grid(True)
  plt.xlabel('Iteration')
  plt.ylabel('Loss')

  plt.legend(loc="upper right")
  plt.savefig("loss_figure.png")


if __name__ == "__main__":
  loss_plot()
