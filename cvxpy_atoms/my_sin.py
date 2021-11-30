from cvxpy.atoms.affine.sum import *
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.atom import Atom
import numpy as np
import scipy.sparse as sp
import scipy as scipy

def my_sin(expr):
    """Extracts the diagonal from a matrix or makes a vector a diagonal matrix.

    Parameters
    ----------
    expr : Expression or numeric constant
        A vector or square matrix.

    Returns
    -------
    Expression
        An Expression representing the diagonal vector/matrix.
    """
    expr = AffAtom.cast_to_const(expr)
    return atom_my_sin(expr)

class atom_my_sin(Atom):

    _allow_complex = False
    def __init__(self, x):
        super(atom_my_sin, self).__init__(x)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the sum of the entries of x squared over y.
        """
        # theta = np.float64(values[0])
        return -np.sin(values[0])+1

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] <= np.pi]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return [np.cos(values[0])]

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return 1, 1

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return True, False

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def validate_arguments(self):
        """Check dimensions of arguments.
        """
        if not self.args[0].is_scalar():
            raise ValueError("Argument of my_sin must be a scalar.")
        super(atom_my_sin, self).validate_arguments()

    def is_quadratic(self):
        """Quadratic if x is affine and y is constant.
        """
        return False

    def is_qpwa(self):
        """Quadratic of piecewise affine if x is PWL and y is constant.
        """
        return False

    def is_atom_quasiconvex(self):
        return True

    def is_atom_quasiconcave(self):
        return False