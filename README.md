[![Upload Python Package](https://github.com/MrTarantoga/MDL-Density-Histogram/actions/workflows/python-publish.yml/badge.svg?event=release)](https://github.com/MrTarantoga/MDL-Density-Histogram/actions/workflows/python-publish.yml)
[![Python application test](https://github.com/MrTarantoga/MDL-Density-Histogram/actions/workflows/python-app.yml/badge.svg)](https://github.com/MrTarantoga/MDL-Density-Histogram/actions/workflows/python-app.yml)

# MDL Optimal Histogram Density Estimation

This package provides a Cython-accelerated implementation of the **Minimum Description Length (MDL) optimal histogram density estimation** algorithm from Kontkanen & Myllymaki (2007). It uses information-theoretic principles to automatically determine optimal variable-width bins for density estimation.

![Freedman-Diaconis vs. MDL-Optimization](https://github.com/MrTarantoga/MDL-Density-Histogram/blob/main/gmm5_idx_3.png)

## Features
- **MDL Principle**: Uses stochastic complexity for model selection
- **Dynamic Programming**: Efficient O(E²·K_max) optimization (cache parametric complexity computation, speed up)
- **Variable-Width Bins**: Adapts to data density variations
- **Automatic Bin Count**: No manual parameter tuning required (except maximum bin count to consider $K_{max}$ and data resolution $\epsilon$)
- **Cython Acceleration**: Critical operations compiled to C

## Installation
You can install the package using pip:
```bash
pip install MDL-Density-Histogram
```
Alternatively, you can install it from source by cloning the repository and running:
```bash
# From project root directory
pip install .
```

Requires:
- Python 3.11+
- NumPy
- Cython
- C compiler (GCC/Clang/MSVC)

## Usage Example
```python
import numpy as np
from mdl_density_hist import mdl_optimal_histogram

# Generate sample data
data = np.random.normal(0, 1, 1000)

# Compute optimal histogram
cut_points = mdl_optimal_histogram(data, epsilon=0.1)

# Visualize result
import matplotlib.pyplot as plt
plt.hist(data, bins=cut_points, density=True)
plt.title('MDL Optimal Histogram')
plt.show()
```

## Parameters
- `data`: Input array (1D numpy array)
- `epsilon`: Quantization precision (default: 0.1)
- `K_max`: Maximum number of bins (default: min(n, 50))

## Algorithm Highlights
- Uses **Ramanujan's factorial approximation** for efficient parametric complexity
- Chache parameteric complexity to speed up computation.

## Paper Citation
Kontkanen, P., & Myllymäki, P. (2007).  
*MDL Histogram Density Estimation*  
Journal of Machine Learning Research 8 (2007) 2007-2038  
[PDF](https://proceedings.mlr.press/v2/kontkanen07a/kontkanen07a.pdf)

## License
Apache 2.0 License - See LICENSE file

## Project Structure
```
src/
├── mdl_density_hist/
│   ├── __init__.py
│   └── mdl_hist.pyx  # Core Cython implementation
└── pyproject.toml
```

## Performance Notes
- Precomputed parametric complexity using dynamic programming
- Memory-optimized array operations via NumPy
- Candidate cut point pruning for reduced search space

## Future Work
- Add visualization tools for complexity curves

For implementation details, see the [paper](https://proceedings.mlr.press/v2/kontkanen07a/kontkanen07a.pdf) and inline code comments.
