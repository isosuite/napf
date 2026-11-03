<p align="center"><img src="https://github.com/tataratat/napf/raw/main/docs/source/_static/napf.png" width="50%" title="nurbs"></p>

**napf - nanoflann wrappers for python and maybe fortran**

[![main](https://github.com/tataratat/napf/actions/workflows/main.yml/badge.svg)](https://github.com/tataratat/napf/actions/workflows/main.yml)
[![PyPI version](https://badge.fury.io/py/napf.svg)](https://badge.fury.io/py/napf)


## python
As [nanoflann](https://github.com/jlblancoc/nanoflann) offers template classes, separate classes are implemented in `napf` for each ___{datatype, distance metric}___. All the search functions are equipped with multi-thread execution. Uses [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) for data input and output.
Currently, the combinations of following options are supported:
- `data type`: {__double__, __float__, __int__, __long__}  _(corresponds to {np.float64, np.float32, np.int32, np.int64})_
- `distance metric`: {__L1__, __L2__}
Note that functions return squared distances, when you use the __L2__ metric.

### quick start
__install with pip:__
```
pip install --upgrade pip
pip install napf
```
*Note: in case your system requires a dynamic build, you need a C++17 compatible C++ compiler. To make sure a correct compiler is chosen, set `export CC=<your-c-compiler> CXX=<your-c++-compiler>`.*

```python
import napf
import numpy as np

data = <data in 2D array>
queries = <query points in 2D array>

kdt = napf.KDT(tree_data=data, metric=1)

distances, indices = kdt.knn_search(
    queries=queries,
    kneighbors=3,
    nthread=4,
)
...
```

## fortran
If you need fortran bindings, please let us know by creating an [issue](https://gthub.com/tataratat/napf/issues).

## Documentation
This package uses a `sphinx` based documentation. An online version of the documentation can be found at [napf - documentation](https://tataratat.github.io/napf/).

If you want to build the documentation yourself use the following commands in the package root directory.
```bash
pip install -r ./docs/requirements.txt
sphinx-build -W -b html docs/source docs/build
```

You will find the documentation in the docs/build folder.
