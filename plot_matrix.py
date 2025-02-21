import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from Methods.TIN.method_linear import TIN
from matplotlib import rcParams
rcParams['font.family']='sans-serif'
rcParams['font.sans-serif']=['Calibri']

datafile = r'./datasets/s4_Dog_data/answer_test.csv'
truth_file=r'./datasets/s4_Dog_data/truth.csv'
tin = TIN(datafile, truth_file=truth_file,beta_1=0.15,beta_2=0.05,beta_3=0.8)
tin.run()
data = tin.pi_wp


class_list = ['0','1','2','3']



for pattern in range(6):
  matrix = np.zeros(shape=(4, 4))
  for i in range(len(class_list)):
    for j in range(len(class_list)):
      matrix[i][j] = data[pattern][class_list[i]][class_list[j]]

  plt.subplot(2, 3, pattern+1)
  plt.imshow(matrix, interpolation='nearest', cmap=cm.YlOrRd)
  title = 'pattern ' + str(pattern+1)
  plt.title(title)

plt.tight_layout()
plt.savefig('pattern.pdf')
# plt.show()