from cvxpy.expressions.expression import Expression
from cvxpy.atoms.norm import norm
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.reshape import reshape


def so2_expr(x):
    """Total variation of a vector, matrix, or list of matrices.

    Uses L1 norm of discrete gradients for vectors and
    L2 norm of discrete gradients for matrices.

    Parameters
    ----------
    value : Expression or numeric constant
        The value to take the total variation of.
    args : Matrix constants/expressions
        Additional matrices extending the third dimension of value.

    Returns
    -------
    Expression
        An Expression representing the total variation.
    """
    x = Expression.cast_to_const(x)

    if x.size > 1:
        raise ValueError("SO2 operator cannot take a non-scalar argument.")
    else:
        args = map(Expression.cast_to_const, args)
        SO2 = [[cos(x), -sin(x)], [sin(x), cos(x)]]
        return SO2