import itertools
import unittest

import numpy as np

import napf


def loop_all_and_test(dims, data_type, metrics, test_func):
    for dim, data_t, metric in itertools.product(dims, data_type, metrics):
        test_func(dim, data_t, metric)


class InitAndQueryTest(unittest.TestCase):
    def test_init_and_query(self):
        dims = list(range(1, 21))
        data_type = ["float64", "float32", "int64", "int32"]
        metrics = [1, 2]

        def test_func(dim, data_t, metric):
            # try to initialize the tree with
            n_data = 100
            randata = (np.random.random((n_data, dim)) * n_data).astype(data_t)
            tree_data = np.vstack(
                (randata, np.array([[-1] * dim], dtype=data_t))
            ).astype(
                data_t
            )  # make sure one more time

            kdt = napf.KDT(tree_data, metric)

            # init test
            qname = type(kdt.core_tree).__qualname__
            assert kdt.core_tree.dim == dim, f"wrong dim init for {qname}"
            assert (
                kdt.core_tree.tree_data.dtype == data_t
            ), f"wrong dtype init for {qname}"
            assert (
                kdt.core_tree.metric == metric
            ), f"wrong metric init for {qname}"
            assert kdt.dim == dim, f"wrong kdt.dim for {qname}"

            # query test should be all be dist = 0 and id = n_data
            dist, ids = kdt.query([[-1] * dim], nthread=1)
            assert np.isclose(dist[0], 0), f"wrong dist query for {qname}"
            assert ids[0] == n_data, f"wrong index query for {qname}"

            # test knn_search
            dist, ids = kdt.knn_search(kdt.tree_data, 1, nthread=2)
            assert np.isclose(
                dist.sum(), 0
            ), f"wrong dist query for {qname}, dim {dim}"

            # skip integer types for index check
            # as it is too easy for them to have duplicates.
            # with default options of nanoflann, this will return smaller index
            if data_t.startswith("int"):
                return

            assert np.all(
                ids.ravel() == np.arange(len(kdt.tree_data))
            ), f"wrong index query for {qname}, dim {dim}"

        loop_all_and_test(dims, data_type, metrics, test_func)

    def test_rknn(self):
        dims = list(range(1, 21))
        data_type = ["float64", "float32"]
        metrics = [1, 2]

        def test_func(dim, data_t, metric):
            n_data = 100
            random_data = (np.random.random((n_data, dim)) * n_data).astype(
                data_t
            )

            # create exactly two matches
            tree_data = np.vstack((random_data, random_data)).astype(data_t)

            kdt = napf.KDT(tree_data, metric)

            nn = 1
            ids, dists = kdt.rknn_search(random_data, 1e-10, nn)
            assert ids.shape[1] == nn
            assert dists.shape[1] == nn
            # we can't guarantee ids but distance should be all zero
            assert np.isclose(dists.sum(), 0)

            # with two queries, also zero dist
            nn = 2
            ids, dists = kdt.rknn_search(random_data, 1e-10, nn)
            assert dists.shape[1] == nn
            assert ids.shape[1] == nn
            assert np.isclose(dists.sum(), 0)

            # with five, only first two queries are zero and last three are
            # dummy values
            nn = 5
            ids, dists = kdt.rknn_search(random_data, 1e-10, 5)
            assert dists.shape[1] == nn
            assert ids.shape[1] == nn
            assert np.isclose(dists[:, :2].sum(), 0)
            # dummy dist is negative for floats
            assert (dists[:, 2:] < 0).all()
            # dummy id must be bigger than n_data, as long as n_data does
            # not overflow
            assert (abs(ids[:, 2:]) > n_data * 2).all()

        loop_all_and_test(dims, data_type, metrics, test_func)

    def test_rknn_multithread_dim_ne_nn(self):
        """Regression test: multi-threaded rknn with dim != n_nearest."""
        pts = np.random.RandomState(42).random((200, 3)).astype(np.float64)
        kdt = napf.KDT(pts, metric=2)
        # dim = 3, n_nearest = 1, nthread = 4
        ids, dists = kdt.rknn_search(pts, 1.0, 1, nthread=4)
        self.assertEqual(ids.shape, (200, 1))
        self.assertEqual(dists.shape, (200, 1))
        self.assertTrue((ids.ravel() == np.arange(200)).all())
        self.assertTrue(np.allclose(dists, 0.0))

        # dim = 3, n_nearest = 5, nthread = 4
        ids, dists = kdt.rknn_search(pts, 10.0, 5, nthread=4)
        self.assertEqual(ids.shape, (200, 5))
        self.assertEqual(dists.shape, (200, 5))
        self.assertTrue((ids[:, 0] == np.arange(200)).all())
        self.assertTrue(np.allclose(dists[:, 0], 0.0))

    def test_radii_search(self):
        """Regression test: radii_search must not accumulate all results into query 0."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        kdt = napf.KDT(pts)
        q = np.array([[0.01, 0.0], [1.01, 1.0]], dtype=np.float64)
        radii = np.array([0.1, 0.1], dtype=np.float64)

        for nthread in [1, 2]:
            ids, dists = kdt.radii_search(
                q, radii, return_sorted=True, nthread=nthread
            )
            self.assertEqual(len(ids), 2)
            self.assertEqual(len(dists), 2)
            self.assertEqual(list(ids[0]), [0])
            self.assertEqual(list(ids[1]), [1])
            self.assertTrue(np.isclose(dists[0][0], 0.0001))
            self.assertTrue(np.isclose(dists[1][0], 0.0001))

    def test_radius_search(self):
        """Test radius_search with single and multi-threading."""
        pts = np.zeros((15, 3), dtype=np.float64)
        pts[:, 0] = np.arange(15) / 10.0
        kdt = napf.KDT(pts, metric=1)
        queries = pts[:7] + [0.04, 0, 0]

        for nthread in [1, 2]:
            ids, dists = kdt.radius_search(
                queries, 0.061, return_sorted=True, nthread=nthread
            )
            self.assertEqual(len(ids), 7)
            self.assertEqual(len(dists), 7)
            for i in range(7):
                self.assertEqual(list(ids[i]), [i, i + 1])
                self.assertTrue(np.allclose(dists[i], [0.04, 0.06]))

    def test_query_ball_point(self):
        """Test query_ball_point."""
        pts = np.array([[0.0, 0.0], [0.01, 0.0], [1.0, 1.0]], dtype=np.float64)
        kdt = napf.KDT(pts)

        for nthread in [1, 2]:
            res = kdt.query_ball_point(
                pts[:2], 0.05, return_sorted=True, nthread=nthread
            )
            self.assertEqual(len(res), 2)
            self.assertEqual(list(res[0]), [0, 1])
            self.assertEqual(list(res[1]), [0, 1])

    def test_unique_data_and_inverse(self):
        """Test unique_data_and_inverse with both return_intersection modes."""
        pts = np.array(
            [[0.0, 0.0], [0.001, 0.0], [1.0, 1.0], [1.001, 1.0]],
            dtype=np.float64,
        )
        kdt = napf.KDT(pts)

        for nthread in [1, 2]:
            # With intersection
            u_pts, u_ids, inv_ids, inter = kdt.unique_data_and_inverse(
                0.01,
                return_unique=True,
                return_intersection=True,
                nthread=nthread,
            )
            self.assertEqual(len(u_ids), 2)
            self.assertEqual(len(inter), 4)
            self.assertTrue(np.allclose(u_pts, pts[u_ids]))

            # Without intersection (uses MinIndexCallback fast-path)
            empty_u, u_ids, inv_ids, inter = kdt.unique_data_and_inverse(
                0.01,
                return_unique=False,
                return_intersection=False,
                nthread=nthread,
            )
            self.assertEqual(empty_u.shape, (0, 2))
            self.assertEqual(len(u_ids), 2)
            self.assertEqual(len(inter), 0)


if __name__ == "__main__":
    unittest.main()
