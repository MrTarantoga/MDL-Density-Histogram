from setuptools import setup, Extension
# Use Cython.Build instead of Cython.Distutils
from Cython.Build import build_ext
import numpy as np

# Define the source directory relative to setup.py
SRC_DIR = "src/mdl_density_hist"

# Define the Cython extension
ext_1 = Extension(
    name="mdl_density_hist.mdl_hist",
    sources=[SRC_DIR + "/mdl_hist.pyx"],
    libraries=[],
    include_dirs=[np.get_include()]
)

EXTENSIONS = [ext_1]

# Setup configuration focused only on building the extension
# Remove the __name__ == "__main__" guard
setup(
    # Removed packages and package_dir
    cmdclass={"build_ext": build_ext},
    ext_modules=EXTENSIONS
)
