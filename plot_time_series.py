import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np

# define exp index
exp_index = 2

# load the time series
data = pd.read_csv(r'./raw_collected_data/demo{}.csv'.format(exp_index), header=None).to_numpy()
print(data.shape)
timesteps = data.shape[1] - 1
time_per_step = 60 / timesteps * 1000   # in the unit of millisecond
print(time_per_step)
window_size = 15

# load the prediction
predictions = pd.read_csv(r'./results/prediction{}.csv'.format(exp_index), header=None).to_numpy()[:,1:]



if __name__ == "__main__":
    for index in range(9):
        print(index)
        time_series = data[index][1:]
        prediction_tag = predictions[index,:]
        fig = plt.figure()
        ims = []
        for i in range(timesteps-window_size):
            series = time_series[i: i+window_size]
            predict_val = prediction_tag[i]
            if predict_val > 0.5:
                c = 'red'
                linewidth = 20
            else:
                c = 'blue'
                linewidth = 2
            im = plt.plot(np.arange(window_size), series, color=c, linewidth=linewidth)
            ims.append(im)
        ani = animation.ArtistAnimation(fig, ims, interval=time_per_step, repeat_delay=0)
        ani.save("./results/test_results/test_{}.gif".format(index),writer='pillow')
