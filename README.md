[![Upload Python Package](https://github.com/MrTarantoga/MDL-Density-Histogram/actions/workflows/python-publish.yml/badge.svg?event=release)](https://github.com/MrTarantoga/MDL-Density-Histogram/actions/workflows/python-publish.yml)

# MDL Optimal Histogram Density Estimation

This package provides a Cython-accelerated implementation of the **Minimum Description Length (MDL) optimal histogram density estimation** algorithm from Kontkanen & Myllymaki (2007). It uses information-theoretic principles to automatically determine optimal variable-width bins for density estimation.

![Example Histogram](docs/gm6_example.png)

## Features
- **MDL Principle**: Uses stochastic complexity for model selection
- **Dynamic Programming**: Efficient O(E²·K_max) optimization
- **Variable-Width Bins**: Adapts to data density variations
- **Automatic Bin Count**: No manual parameter tuning required
- **Cython Acceleration**: Critical path operations compiled to C

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
from mdl_density_hist.mdl_hist import mdl_optimal_histogram

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
- Implements **Equation 22** for candidate cut point generation
- Uses **Ramanujan's factorial approximation** for efficient parametric complexity
- Dynamic programming table optimization with caching
- Handles edge cases through implicit boundary conditions

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
- Add support for multidimensional histograms
- Implement Bayesian Information Criterion (BIC) comparison
- Add visualization tools for complexity curves

For implementation details, see the [paper](https://proceedings.mlr.press/v2/kontkanen07a/kontkanen07a.pdf) and inline code comments.
