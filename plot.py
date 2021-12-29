
import scipy.signal
import os
import numpy as np
import matplotlib.pyplot as plt


def loss_plot():
  savepath = 'results/yolov4_ghostbottle'
  traindata = np.loadtxt(os.path.join(savepath, 'loss.txt'))[:, 0]
#   valdata = np.loadtxt(os.path.join(savepath, 'val_loss.txt'))
  iters = range(len(traindata))

  plt.figure()
  plt.plot(iters, traindata, 'red', linewidth=2, label='train loss')
#   plt.plot(iters, valdata, 'coral', linewidth=2, label='val loss')
#   try:
#       if len(traindata) < 25:
#           num = 5
#       else:
#           num = 15

#       plt.plot(iters, scipy.signal.savgol_filter(traindata, num, 3),
#                'green', linestyle='--', linewidth=2, label='smooth train loss')
#       # plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3),
#       #          '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
#   except:
#       pass

  plt.grid(True)
  plt.xlabel('Iteration')
  plt.ylabel('Loss')

  plt.legend(loc="upper right")
  plt.savefig(os.path.join(savepath, "loss_figure.png"))


if __name__ == "__main__":
  loss_plot()
