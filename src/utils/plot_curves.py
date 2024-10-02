import os
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra
def plot_curves(x_list=None, y_list=None, plot_name="unamed_plot", output_directory=None):
    """
    Plots curves given by x_list and y_list to a file in the specified output directory.
    
    :param x_list: List of arrays for the x values of each curve.
    :param y_list: List of arrays for the y values of each curve.
    :param plot_name: Name of the plot file (without extension).
    :param output_directory: Directory where the plot will be saved.
    """
    if x_list is None:
        x_list = [range(len(y)) for y in y_list]

    if output_directory is None:
        # Get the current working directory from Hydra (the output directory).
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        output_directory = hydra_cfg['runtime']['output_dir']

    # Ensure that the plot name is unique by adding .png extension.
    plot_path = os.path.join(output_directory, f"{plot_name}.png")

    # Create the plot
    plt.figure()
    for x, y in zip(x_list, y_list):
        plt.plot(x, y)

    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title(plot_name)

    # Save the plot, overwriting if it exists.
    plt.savefig(plot_path)
    plt.close()


