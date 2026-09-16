# A recreation and extension of [Latham et al. (2000)](https://pubmed.ncbi.nlm.nih.gov/10669496/)

This is a recreation and extension of the neuronal dynamics model presented in
[Latham et al. (2000)](https://pubmed.ncbi.nlm.nih.gov/10669496/). Check
[Results.ipynb](./Results.ipynb) for a recreation of the results, and
[Watts-Strogatz.ipynb](./Watts-Strogatz.ipynb) for an extension where I trade
the local connectivity model examined in the paper for a directed
[Watts-Strogatz
model](https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model) and examine
the robustness of the claims in this regime.

## Installation

The Jupyter notebooks are populated with the key results so no code needs to be
run, however, if you should want to run the simulation yourself:

1. Clone the directory `git clone https://github.com/hectorBrown/neuronsim`.
2. Enter the directory and use `uv sync` to install the dependencies.
3. Run `uv run jupyter-lab` to open a jupyter-lab client, and run the cells.

Skeleton caching code is provided within the notebooks for a gzip-compressed
pickle cache, but this is not handled automatically.
