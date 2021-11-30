"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy.atoms.atom import Atom
import numpy as np
import cvxpy
import scipy.sparse as sp
import scipy as scipy
import cvxpy.lin_ops.lin_utils as lu

def so2(expr):
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
    # expr = AffAtom.cast_to_const(expr)
    return atom_exp_so2(expr)

class atom_exp_so2(Atom):

    _allow_complex = False
    def __init__(self, x):
        super(atom_exp_so2, self).__init__(x)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the sum of the entries of x squared over y.
        """
        theta = np.float64(values[0])
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])#.reshape(2, 2)

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return [np.array([[-np.sin(values[0]), -np.cos(values[0])], [np.cos(values[0]), -np.sin(values[0])]])@self.numeric(values)]

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return 2, 2

    def validate_arguments(self):
        """Check dimensions of arguments.
        """
        if not self.args[0].is_scalar():
            raise ValueError("Argument to exp(SO2) must be a scalar.")
        super(atom_exp_so2, self).validate_arguments()

    # def canonicalize(self):

    def graph_implementation(self, arg_objs, shape, data=None):
        # return ([lu.], [])

