# 2020ws-graph-summarization

This is the repository for the paper "Graph Summarization with Graph Neural Networks" by Johannes Horvath, Maximilian Blasi and Manuel Freudenreich under the supervision of David Richerby and Ansgar Scherp at the Ulm University. It containes the source code, the scientific paper and the technical report.

## Software

Three Anaconda environment are supplied in etc/:
* conda_env_server_a.yml: this can be used to run the preprocessing pipeline (no CUDA required)
* conda_env_server_a_plot.yml: this can be used to plot the class distributions (special versions needed of numpy and matplotlib for pgf)
* conda_env_server_c.yml: this can be used to train the models on a GPU (CUDA required)

### ***Prerequisites*** 

* PyTorch (https://pytorch.org/get-started/locally/)
* PyTorch Geometric (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
* Tensorboard
* Tensorflow (for Tensorboard)
* (Matplotlib)

Also see Anaconda yaml files in etc/.


### Run

Please refer to our technical report and especially the manual section.
