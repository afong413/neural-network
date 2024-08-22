from collections.abc import Callable


class CalcFunction(Callable):  # MARK: CalcFunction
    """
    A function with a derivative. I stopped using `sympy` because of
    functions like ReLU (`max(x,0)`) that it couldn't differentiate.
    """

    def __init__(
        self,
        f: Callable,
        df: Callable,
    ):
        self._f = f
        self._df = df

    def __call__(self, *args, **kwargs):
        """
        The function.
        """
        return self._f(*args, **kwargs)

    def d(self, *args, **kwargs):
        """
        The derivative of the function.
        """
        return self._df(*args, **kwargs)
