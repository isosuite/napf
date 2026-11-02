import numpy as np

from napf import _napf as core  # noqa: F401

np2napf_dtypes = {
    "int32": "i",
    "int64": "l",
    "float32": "f",
    "float64": "d",
}


def validate_metric_input(metric):
    """
    internal use fn for metric validation

    Parameters
    -----------
    metric: int or str

    Returns
    --------
    m: int
      validated metric casted to int.
    """
    try:
        m = int(metric)
        assert m in [1, 2]

    except ValueError:
        mstr = str(metric).lower()
        print(mstr)
        assert mstr.startswith("l1") or mstr.startswith("l2")
        m = int(mstr[-1])
        assert m in [1, 2]

    except BaseException:
        raise ValueError(
            "KDT only supports 1 or 2 (alternatively 'L1' or 'L2') "
            "as metric input."
        )

    return m


def enforce_contiguous(array, dtype=None):
    """
    If input is an instance / subclass of np.ndarray, this will check
    if they are configuous. If so, returns same object, else turns makes it
    contiguous and returns.

    Parameters
    ----------
    array: array-like

    Returns
    -------
    contiguous_array: array-like
    """
    if isinstance(array, np.ndarray):
        if array.flags["C_CONTIGUOUS"] and (
            dtype is None or dtype is array.dtype
        ):
            return array
        return np.ascontiguousarray(array, dtype=dtype)

    return array


def core_class_str_and_data(tree_data, metric):
    """
    Returns class name of current setting.
    Also checks if it is valid dtype.

    Parameters
    -----------
    tree_data: (n, dim) np.ndarray
    metric: int or str

    Returns
    --------
    core_class_str: str
    core_class_data: (n, dim) np.ndarray
    """
    arr = enforce_contiguous(
        tree_data, dtype=getattr(tree_data, "dtype", None)
    )
    dtypestr = str(arr.dtype)
    if dtypestr not in np2napf_dtypes:
        raise TypeError(f"Sorry, `napf` does not support ({dtypestr}) dtypes.")

    if arr.ndim != 2:
        raise ValueError(
            "Please make sure your input array is 2D! "
            "Given array's dim is {arr.ndim}."
        )

    # extract info
    data_t = np2napf_dtypes[dtypestr]
    metric = validate_metric_input(metric)

    return f"KDT{data_t}L{metric}", arr


class KDT:
    """
    `napf` is implemented as template, thus, there are separate classes
    for each {data_type, dim, metric}.
    Currently following combinations are supported:
    data_type: {double, int}
    dim: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20}
    metric: {L1, L2}

    Given tree_data, creates corresponding core kdt class.
    Tree is initialized using `newtree()`.


    Parameters
    -----------
    tree_data: (n, dim) np.ndarray
      Default is None. {double, float, int, long}
    metric: int or str
      Default is 2 and distance will be a squared euklidian distance.
      Valid options are {1, l1, L1, 2, l2, L2}.
    leaf_size: int
    nthread: int
      Default thread count for all multi-thread-

    Returns
    --------
    core_obj: KDT{data_t}D{dim}L{metric}
    """

    __slots__ = (
        "_core_tree",
        "_dtype",
        "_nthread",
    )

    def __init__(self, tree_data, metric=2, leaf_size=10, nthread=1):
        """
        Init
        """
        self.newtree(tree_data, metric, leaf_size, nthread)
        self.nthread = nthread

    @property
    def nthread(self):
        """
        Returns saved default value for nthread.

        Parameters
        -----------
        None

        Returns
        --------
        nthread: int
        """
        return self._nthread

    @nthread.setter
    def nthread(self, nthread_):
        """
        Sets default value for nthread.

        Parameters
        -----------
        nthread_: int

        Returns
        --------
        None
        """
        self._nthread = nthread_

    @property
    def core_tree(self):
        """
        Returns initialized core tree, if there's any.

        Parameters
        -----------
        None

        Returns
        --------
        core_tree: napf.core.KDT*
        """
        if hasattr(self, "_core_tree"):
            return self._core_tree

        else:
            return None

    @property
    def tree_data(self):
        """
        Returns data used to initialize core tree. Read only

        Parameters
        -----------
        None

        Returns
        --------
        tree_data: (n, d) np.ndarray
          {double, float, int, long}
        """
        coretree = self.core_tree
        if coretree is not None:
            return coretree.tree_data

        else:
            return None

    @property
    def dtype(self):
        """
        Returns dtype of current tree
        """
        return self._dtype

    def newtree(self, tree_data, metric=2, leaf_size=10, nthread=1):
        """
        Given 2D array-like tree_data, it:
          1. makes sure data is a contiguous array
          2. builds a corresponding kdt.

        Note that core kdt objects also have `newtree()` function with the same
        parameters, and build a new kdt each time it is called.

        Parameters
        -----------
        tree_data: (n, d) np.ndarray
          {double, float, int, long}

        """
        core_cls, tdata = core_class_str_and_data(
            np.ascontiguousarray(tree_data), metric
        )  # checks and raises error
        # we can call newtree() function of the core class,
        # if _core_tree already exists.
        # However, creating a new kdt should not add significant overhead.
        self._core_tree = eval(f"core.{core_cls}(tdata, leaf_size, nthread)")
        self._dtype = tdata.dtype

    def knn_search(self, queries, kneighbors, nthread=None):
        """
        k-nearest-neighbor search.

        Parameters
        -----------
        queries: (m, d) np.ndarray
          Data type will be casted to the same type as `tree_data`.
        kneighbors: int
        nthread: int
          Default is None and will use self.nthread.

        Returns
        --------
        ids_and_distances: tuple
          ((m, kneighbors) np.ndarray - double dists,)
           (m, kneighbors) np.ndarray - uint ids)
        """
        if nthread is None:
            nthread = self.nthread

        return self.core_tree.knn_search(
            enforce_contiguous(queries, self.dtype), kneighbors, nthread
        )

    def query(self, queries, nthread=None):
        """
        scipy-like KDTree query call.
        Same as `knn_search(queries, 1, nthreads)`

        Parameters
        -----------
        queries: (m, d) np.ndarray
          Data type will be casted to the same type as `tree_data`.
        nthread: int
          Default is None and will use self.nthread.

        Returns
        --------
        distances_and_ids: tuple
          ((m, 1) np.ndarray - double dists,)
           (m, 1) np.ndarray - uint ids)
        """
        if nthread is None:
            nthread = self.nthread

        return self.core_tree.query(
            enforce_contiguous(queries, self.dtype), nthread
        )

    def radius_search(self, queries, radius, return_sorted, nthread=None):
        """
        Searches for neighbors in given radius.

        Parameters
        -----------
        queries: (m, d) np.ndarray
          Data type will be casted to the same type as `tree_data`.
        radius: float
        return_sorted: bool
        nthread: int
          Default is None and will use self.nthread

        Returns
        --------
        ids_and_distances: tuple
          ((m, 1) np.ndarray - uint ids,
           (m, 1) np.ndarray - double dists)
        """
        if nthread is None:
            nthread = self.nthread

        return self.core_tree.radius_search(
            enforce_contiguous(queries, self.dtype),
            radius,
            return_sorted,
            nthread,
        )

    def rknn_search(self, queries, radius, n_nearest, nthread=None):
        """
        Searches for k-nearest neighbors within the radius.
        With insufficient neighbors, rest of the return values will have dummy
        filled in - they will be the maximum value of each data type.
        If the dtype is signed, it will have a negative value.

        Parameters
        ----------
        queries: (m, d) np.ndarray
        radius: float
        n_nearest: int
        nthread: int

        Returns
        -------
        ids_and_distances: tuple
          ((m, 1) np.ndarray - uint ids,
           (m, 1) np.ndarray - double dists)
        """
        if nthread is None:
            nthread = self.nthread

        return self.core_tree.rknn_search(
            enforce_contiguous(queries, self.dtype),
            radius,
            n_nearest,
            nthread,
        )

    def query_ball_point(self, queries, radius, return_sorted, nthread=None):
        """
        scipy-like KDTree query_ball_point call.

        Parameters
        ----------
        queries: (m, d) np.ndarray
          Data type will be casted to the same type as `tree_data`.
        radius: float
        return_sorted: bool
        nthread: int
          Default is None and will use self.nthread

        Returns
        -------
        ids: list
          list of np.array
        """
        if nthread is None:
            nthread = self.nthread

        return self.core_tree.query_ball_point(
            enforce_contiguous(queries, self.dtype),
            radius,
            return_sorted,
            nthread,
        )

    def radii_search(self, queries, radii, return_sorted, nthread=None):
        """
        Similar to `radius_search`, but you can specify radius for each query.

        Parameters
        -----------
        queries: (m, d) np.ndarray
          Data type will be casted to the same type as `tree_data`.
        radii: (m,) np.ndarray
        return_sorted: bool
        nthread: int
          Default is None and will use self.nthread

        Returns
        --------
        ids_and_distances: tuple
          ((m, 1) np.ndarray - uint ids,
           (m, 1) np.ndarray - double dists)
        """
        # input size check
        if len(queries) != len(radii):
            raise ValueError(
                f"Input size mismatch between queries ({len(queries)}) "
                f" and radii ({len(radii)})."
                "They should be the same."
            )

        if nthread is None:
            nthread = self.nthread

        return self.core_tree.radii_search(
            enforce_contiguous(queries, self.dtype),
            enforce_contiguous(radii, self.dtype),
            return_sorted,
            nthread,
        )

    def unique_data_and_inverse(
        self,
        radius,
        return_unique=True,
        return_intersection=True,
        nthread=None,
    ):
        """
        Finds unique tree data with in given radius tolerance.


        Parameters
        ----------
        radius: float
        return_unique: bool
          Default is True. Otherwise, will be an empty array return.
        return_intersection: bool
          Default is True, Otherwise, will be an empty UIntVectorVector return.
        nthread:int

        Returns
        -------
        unique_data: np.ndarray
          Empty if return_unique is False.
          Same as kdt.tree_data[unique_ids].
        unique_ids: np.ndarray
          Indices of unique entries from tree data.
          First occurrence is considered unique.
        inverse_ids: np.ndarray
          Indices to reconstruct original tree_data with
          unique_data. kdt.tree_data == unique_data[inverse_ids]
        intersection: UIntVectorVector
          Empty if return_intersection is False.
          Intersection of each data with respect to all the others.
        """
        if nthread is None:
            nthread = self.nthread

        (
            original_inverse,
            intersection,
        ) = self.core_tree.tree_data_unique_inverse(
            radius, return_intersection, nthread
        )

        unique_ids, inverse_ids = np.unique(
            original_inverse, return_inverse=True
        )

        if return_unique:
            return (
                self.core_tree.tree_data[unique_ids],
                unique_ids,
                inverse_ids,
                intersection,
            )
        else:
            return np.array(), unique_ids, inverse_ids, intersection
