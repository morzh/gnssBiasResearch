from pyLOps import Diagonal
import numpy as np
from pylops.utils import dottest
from scipy.sparse.linalg import lsqr
import pytest

from numpy.testing import assert_array_almost_equal


# @pytest.mark.parametrize("par", [(par1), (par2)])
def test_Diagonal(par):
    """Dot-test and inversion for diagonal operator
    """
    d = np.arange(par['nx']) + 1.

    Dop = Diagonal(d)
    assert dottest(Dop, par['nx'], par['nx'], complexflag=0 if par['imag'] == 1 else 3)

    x = np.ones(par['nx'])
    xlsqr = lsqr(Dop, Dop * x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)
