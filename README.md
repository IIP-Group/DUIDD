# DUIDD: Deep-Unfolded Interleaved Detection and Decoding for MIMO Wireless Systems

This simulator implements the experiments of the paper “DUIDD: Deep-Unfolded Interleaved Detection and Decoding for MIMO Wireless Systems,” 
*R. Wiesmayr, C. Dick, 
J. Hoydis, and C. Studer, Procs. Asilomar Conf. Signals, Syst., Comput., Oct. 2022, available at https://arxiv.org/abs/2212.07816*

The simulations are implemented with [NVIDIA Sionna](https://github.com/NVlabs/sionna) Release v0.11 and own extensions.

Parts of the code are also based on
- *R. Wiesmayr, G. Marti, C. Dick, H. Song, and C. Studer
“Bit Error and Block Error Rate Training for ML-Assisted
Communication,” arXiv:2210.14103, 2022*, available at https://arxiv.org/abs/2210.14103
- *C. Studer, S. Fateh, and D. Seethaler, “ASIC Implementation of Soft-Input Soft-Output MIMO Detection Using MMSE Parallel Interference
Cancellation,” IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 1754–1765, July 2011, available at https://www.csl.cornell.edu/~studer/papers/11JSSC-mmsepic.pdf*

If you are using this simulator (or parts of it) for a publication, you must cite the above-mentioned references and clearly mention this in your paper.

## Running simulations
Please have your Python environment ready with NVIDIA Sionna v0.11, as the code was developed and tested for this version.

The main simulation script `simulation_script.py` is located in `./scr` and contains multiple simulation parameters that can be modified at will.
The script trains the specified signal processing models and evaluates a performance benchmark. At the end, bit error rate and block error rate curves are plotted and saved.

> The ray-tracing channels utilized by our script can be downloaded from [here](https://iis-nextcloud.ee.ethz.ch/s/PMDAyWzc6kXwqMS).

Before running the simulations, the following directories have to be created:
- `./data/weights/` for saving the trained model weights
- `./results` for the simulation results (BER and BLER curves), which are saved as `.csv` and `.pickle` files
- If you want to use the ray-tracing channels, download and place them under `./data/channels/`

## Version history

- Version 0.1: [wiesmayr@iis.ee.ethz.ch](wiesmayr@iis.ee.ethz.ch) - initial version for GitHub release