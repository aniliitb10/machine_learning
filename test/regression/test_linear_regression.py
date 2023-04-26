import unittest

import numpy as np
import numpy.testing as npt

import regression.linear_regression as lr


class LinearRegressionTest(unittest.TestCase):
    """
    All these examples are taken from coursera # Supervised Machine Learning course
    # Regression and Classification >> Week 2 >> Optional Lab: Multiple linear regression
    """

    def test_compute_cost_simple(self):
        x = np.array([[2104, 5, 1, 45],
                      [1416, 3, 2, 40],
                      [852, 2, 1, 35]])
        y = np.array([460, 232, 178])
        w = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
        b = 785.1811367994083

        self.assertAlmostEqual(lr.compute_cost(x, y, w, b), 1.5578904880036537e-12)

    def test_compute_gradient(self):
        x = np.array([[2104, 5, 1, 45],
                      [1416, 3, 2, 40],
                      [852, 2, 1, 35]])
        y = np.array([460, 232, 178])
        w = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
        b = 785.1811367994083
        dj_dw, dj_db = lr.compute_gradient(x, y, w, b)
        npt.assert_almost_equal(dj_dw, [-0.002726235812, -0.000006271973, -0.000002217456, -0.000069240340])
        self.assertAlmostEqual(dj_db, -1.673925169143331e-06)

    def test_gradient_descent(self):
        x = np.array([[2104, 5, 1, 45],
                      [1416, 3, 2, 40],
                      [852, 2, 1, 35]])
        y = np.array([460, 232, 178])
        w = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
        # b = 785.1811367994083
        initial_w = np.zeros_like(w)
        initial_b = 0.
        iterations = 100_000  # course uses 1000 iterations, but changed it to get better calculations
        alpha = 5.0e-7
        w_final, b_final, _ = lr.gradient_descent(x, y, initial_w, initial_b, alpha, iterations)
        npt.assert_almost_equal(w_final, [0.24224154, 0.28821169, -0.85520022, -1.57622854])
        self.assertAlmostEqual(b_final, -0.04168502)
        self.assertAlmostEqual(lr.compute_cost(x, y, w_final, b_final), 563.25375720)


if __name__ == '__main__':
    unittest.main()
