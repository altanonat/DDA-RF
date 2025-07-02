# Data-Driven Modeling of the Rabinovich-Fabrikant System: EDMD, SINDy, and ANN

This repository contains MATLAB implementations of three data-driven modeling techniques — **Extended Dynamic Mode Decomposition (EDMD)**, **Sparse Identification of Nonlinear Dynamical Systems (SINDy)**, and **Artificial Neural Networks (ANN)** — applied to the **Rabinovich-Fabrikant** dynamical system.

## Description

The goal of this project is to demonstrate and compare the effectiveness of various machine learning-based techniques in approximating nonlinear dynamical systems using simulation data. The Rabinovich-Fabrikant system is chosen as a testbed due to its rich nonlinear and potentially chaotic behavior.

Implemented methods include:
- **EDMD** for Koopman-based linear lifting
- **SINDy** for sparse regression of governing equations
- **ANN** for neural operator learning of dynamics

## Structure

📁 EDMD/
📁 SINDy/
📁 ANN/
📄 README.md

Each folder contains method-specific scripts and results for learning the dynamics from simulated data of the Rabinovich-Fabrikant system.

## Acknowledgments

Portions of the SINDy and ANN implementations are adapted from the following textbook:

> **Brunton, S. L., & Kutz, J. N. (2022).** *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control.* Cambridge University Press.

The official companion MATLAB code repository for the book can be found here:  
🔗 [https://github.com/dynamicslab/databook_matlab](https://github.com/dynamicslab/databook_matlab)

Please consider referring to the book and its repository for further insights, theoretical background, and broader applications of these techniques.

## License

This project is released under the MIT License.

## Contact

For questions, suggestions, or collaborations, feel free to open an issue or contact the maintainer v
