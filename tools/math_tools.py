"""
Mathematical computation tools using SymPy and NumPy.
Provides symbolic math, equation solving, calculus, and matrix operations.
"""

import sympy
import numpy as np
from typing import Any, Optional, List
from utils import setup_logger

logger = setup_logger("math_tools")


def calculate(expression: str) -> Optional[float]:
    """
    Evaluate mathematical expression.

    Args:
        expression: Mathematical expression string

    Returns:
        Numerical result of evaluation
    """
    try:
        result = sympy.sympify(expression)
        return float(result.evalf())
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        return None


def solve_equation(equation: str, variable: str = 'x') -> List[float]:
    """
    Solve equation symbolically.

    Args:
        equation: Equation string (e.g., "x**2 - 4")
        variable: Variable to solve for

    Returns:
        List of solutions
    """
    try:
        x = sympy.Symbol(variable)
        eq = sympy.sympify(equation)
        solutions = sympy.solve(eq, x)
        return [float(sol.evalf()) for sol in solutions]
    except Exception as e:
        logger.error(f"Equation solving error: {e}")
        return []


def differentiate(expression: str, variable: str = 'x') -> str:
    """
    Compute derivative.

    Args:
        expression: Expression to differentiate
        variable: Variable to differentiate with respect to

    Returns:
        String representation of derivative
    """
    try:
        x = sympy.Symbol(variable)
        expr = sympy.sympify(expression)
        derivative = sympy.diff(expr, x)
        return str(derivative)
    except Exception as e:
        logger.error(f"Differentiation error: {e}")
        return ""


def integrate(expression: str, variable: str = 'x') -> str:
    """
    Compute integral.

    Args:
        expression: Expression to integrate
        variable: Variable to integrate with respect to

    Returns:
        String representation of integral
    """
    try:
        x = sympy.Symbol(variable)
        expr = sympy.sympify(expression)
        integral = sympy.integrate(expr, x)
        return str(integral)
    except Exception as e:
        logger.error(f"Integration error: {e}")
        return ""


def matrix_operations(operation: str, matrix_a: list, matrix_b: list = None) -> Any:
    """
    Perform matrix operations.

    Args:
        operation: Operation type (determinant, inverse, eigenvalues, multiply)
        matrix_a: First matrix as nested list
        matrix_b: Second matrix for operations like multiply

    Returns:
        Result of matrix operation
    """
    try:
        A = np.array(matrix_a)

        if operation == "determinant":
            return float(np.linalg.det(A))

        elif operation == "inverse":
            return np.linalg.inv(A).tolist()

        elif operation == "eigenvalues":
            return np.linalg.eigvals(A).tolist()

        elif operation == "multiply" and matrix_b:
            B = np.array(matrix_b)
            return np.matmul(A, B).tolist()

        else:
            logger.warning(f"Unknown operation or missing matrix_b: {operation}")
            return None

    except Exception as e:
        logger.error(f"Matrix operation error: {e}")
        return None
