# Learning to Model Remainder in Supervised Learning
Visualizing the convergence on a 3D example problem, i.e., z=f(x,y). 

![](generated_gif/gd.gif)

Download the gif of the GD convergence via this [link](generated_gif/gd.gif).

## Usage

Visualize the convergence with a specific optimization method. The options of optimizer_name are gd, lagd, adam, rmsprop, and adabound.
```bash
python visualize_convergence.py --opt optimizer_name
```

Generate a gif rendering a convergence process.
```bash
python gen_gif.py
```

Plot the loss curve of the convergence.
```bash
python plot_z.py
```