import os
import imageio

duration = .1

# input_folder = 'figures_gd'
# n_frames = 60
# output = 'gd.gif'

# input_folder = 'figures_lagd'
# n_frames = 600
# output = 'lagd.gif'

# input_folder = 'figures_adam'
# n_frames = 250
# output = 'adam.gif'

# input_folder = 'figures_rmsprop'
# n_frames = 150
# output = 'rmsprop.gif'

input_folder = 'figures_adabound'
n_frames = 300
output = 'adabound.gif'

images = []
for i in range(n_frames):
	images.append( imageio.imread(os.path.join(input_folder, '{}.png'.format(i))) )

kargs = {'duration':duration}
imageio.mimsave(os.path.join(input_folder,output), images, 'GIF', **kargs)