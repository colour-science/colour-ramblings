# coding: utf-8
from __future__ import division, unicode_literals

import numpy as np
import re
from contextlib import contextmanager
from copy import deepcopy
from collections import OrderedDict
from operator import (
    add, div, mul, pow, sub, iadd, idiv, imul, ipow, isub)
from pandas import Series

from colour import (
    CaseInsensitiveMapping,
    CubicSplineInterpolator,
    Extrapolator,
    LinearInterpolator,
    PchipInterpolator,
    SpragueInterpolator,
    as_numeric,
    is_numeric,
    tsplit,
    tstack,
    warning)

np.set_printoptions(formatter={'float': '{:0.8f}'.format})

# TODO: Is this the right name for the interpolator? Should we call the default
# null value `default`?
class NullInterpolator(object):
    """

    Parameters
    ----------
    x : ndarray
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y : ndarray
        Dependent and already known :math:`y` variable values to
        interpolate.

    Methods
    -------
    __call__


    Examples
    --------
    """

    def __init__(self,
                 x=None,
                 y=None,
                 absolute_tolerance=10e-7,
                 relative_tolerance=10e-7,
                 default=np.nan):
        self._x = None
        self.x = x
        self._y = None
        self.y = y
        self._absolute_tolerance = None
        self.absolute_tolerance = absolute_tolerance
        self._relative_tolerance = None
        self.relative_tolerance = relative_tolerance
        self._default = None
        self.default = default

        self._validate_dimensions()

    @property
    def x(self):
        """
        Property for **self._x** private attribute.

        Returns
        -------
        array_like
            self._x
        """

        return self._x

    @x.setter
    def x(self, value):
        """
        Setter for **self._x** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(np.float_)

            assert value.ndim == 1, (
                '"x" independent variable must have exactly one dimension!')

        self._x = value

    @property
    def y(self):
        """
        Property for **self._y** private attribute.

        Returns
        -------
        array_like
            self._y
        """

        return self._y

    @y.setter
    def y(self, value):
        """
        Setter for **self._y** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(np.float_)

            assert value.ndim == 1, (
                '"y" dependent variable must have exactly one dimension!')

        self._y = value

    @property
    def relative_tolerance(self):
        """
        Property for **self._relative_tolerance** private attribute.

        Returns
        -------
        numeric
            self._relative_tolerance
        """

        return self._relative_tolerance

    @relative_tolerance.setter
    def relative_tolerance(self, value):
        """
        Setter for **self._relative_tolerance** private attribute.

        Parameters
        ----------
        value : numeric
            Attribute value.
        """

        if value is not None:
            assert is_numeric(value), (
                '"relative_tolerance" variable must be a "numeric"!')

        self._relative_tolerance = value

    @property
    def absolute_tolerance(self):
        """
        Property for **self._absolute_tolerance** private attribute.

        Returns
        -------
        numeric
            self._absolute_tolerance
        """

        return self._absolute_tolerance

    @absolute_tolerance.setter
    def absolute_tolerance(self, value):
        """
        Setter for **self._absolute_tolerance** private attribute.

        Parameters
        ----------
        value : numeric
            Attribute value.
        """

        if value is not None:
            assert is_numeric(value), (
                '"absolute_tolerance" variable must be a "numeric"!')

        self._absolute_tolerance = value

    @property
    def default(self):
        """
        Property for **self._default** private attribute.

        Returns
        -------
        numeric
            self._default
        """

        return self._default

    @default.setter
    def default(self, value):
        """
        Setter for **self._default** private attribute.

        Parameters
        ----------
        value : numeric
            Attribute value.
        """

        if value is not None:
            assert is_numeric(value), (
                '"default" variable must be a "numeric"!')

        self._default = value

    def __call__(self, x):
        """
        Evaluates the interpolator at given point(s).


        Parameters
        ----------
        x : numeric or array_like
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        float or ndarray
            Interpolated value(s).
        """

        x = np.atleast_1d(x).astype(np.float_)

        xi = as_numeric(self._evaluate(x))

        return xi

    def _evaluate(self, x):
        """
        Performs the interpolator evaluation at given points.

        Parameters
        ----------
        x : ndarray
            Points to evaluate the interpolant at.

        Returns
        -------
        ndarray
            Interpolated points values.
        """

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        index = nearest_index(self._x, x)
        values = self._y[index]
        values[~np.isclose(self._x[index],
                           x,
                           rtol=self._absolute_tolerance,
                           atol=self._relative_tolerance)] = self._default

        return values

    def _validate_dimensions(self):
        """
        Validates variables dimensions to be the same.
        """

        if len(self._x) != len(self._y):
            raise ValueError(
                ('"x" independent and "y" dependent variables have different '
                 'dimensions: "{0}", "{1}"').format(len(self._x),
                                                    len(self._y)))

    def _validate_interpolation_range(self, x):
        """
        Validates given point to be in interpolation range.
        """

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if below_interpolation_range.any():
            raise ValueError('"{0}" is below interpolation range.'.format(x))

        if above_interpolation_range.any():
            raise ValueError('"{0}" is above interpolation range.'.format(x))


INTERPOLATORS = CaseInsensitiveMapping({
    'Cubic Spline': CubicSplineInterpolator,
    'Linear': LinearInterpolator,
    'Null': NullInterpolator,
    'Pchip': PchipInterpolator,
    'Sprague': SpragueInterpolator})


def nearest_index(a, b):
    index = np.searchsorted(a, b)

    return np.where(np.abs(b - a[index - 1]) < np.fabs(b - a[index]),
                    index - 1,
                    index)


def nearest(a, b):
    a = np.asarray(a)

    return a[nearest_index(a, b)]


@contextmanager
def ndarray_write(a):
    a.setflags(write=True)

    yield a

    a.setflags(write=False)


def fill_nan(a, method='Interpolation', default=np.nan):
    mask = np.isnan(a)

    if method.lower() == 'interpolation':
        a[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            a[~mask])
    elif method.lower() == 'constant':
        a[mask] = default

    return a


def is_pandas_installed():
    try:
        import pandas

        return True
    except ImportError:
        return False


def unpack_data(data=None, domain=None):
    domain_f, range_f, name_f = None, None, None
    if isinstance(data, Signal):
        domain_f = data.domain
        range_f = data.range
    if (isinstance(data, tuple) or
            isinstance(data, list) or
            isinstance(data, np.ndarray)):
        data = np.asarray(data)
        if data.ndim == 1:
            range_f = data
        elif data.ndim == 2:
            domain_f, range_f = tsplit(data)
        else:
            raise ValueError('"data" must be a 1d or 2d array-like variable!')
    elif (isinstance(data, dict) or
              isinstance(data, OrderedDict)):
        domain_f, range_f = tsplit(sorted(data.items()))
    elif is_pandas_installed():
        if isinstance(data, Series):
            domain_f = data.index.values
            range_f = data.values
            name_f = data.name

    domain_f = domain_f if domain is None else domain

    return domain_f, range_f, name_f


class Signal(object):
    def __init__(self, data=None, domain=None, **kwargs):

        interpolator = kwargs.get('interpolator')
        interpolator_args = kwargs.get('interpolator_args')
        extrapolator = kwargs.get('extrapolator')
        extrapolator_args = kwargs.get('extrapolator_args')
        name = kwargs.get('name')

        self._domain = None
        self._range = None
        self._interpolator = LinearInterpolator
        self._interpolator_args = {}
        self._extrapolator = Extrapolator
        self._extrapolator_args = {
            'method': 'Constant',
            'left': np.nan,
            'right': np.nan}
        self._name = '{0} ({1})'.format(self.__class__.__name__, id(self))

        domain_f, range_f, name_f = unpack_data(data, domain)
        name_f = name_f if name is None else name

        self.domain = domain_f
        self.range = range_f
        self.interpolator = interpolator
        self.interpolator_args = interpolator_args
        self.extrapolator = extrapolator
        self.extrapolator_args = extrapolator_args
        self.name = name_f

        self._create_function()

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, value):
        if value is not None:
            if not np.all(np.isfinite(value)):
                warning('"domain" variable is not finite, '
                        'unpredictable results may occur!\n{0}'.format(value))

            # TODO: `self.domain` is a copy of `value` to avoid side effects,
            # Is it a smart way to avoid them?
            value = np.copy(np.asarray(value))

            if self._range is not None:
                assert value.size == self._range.size, (
                    '"domain" and "range" variables must have same size!')

            value.setflags(write=False)
            self._domain = value
            self._create_function()

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, value):
        if value is not None:
            if not np.all(np.isfinite(value)):
                warning('"range" variable is not finite, '
                        'unpredictable results may occur!\n{0}'.format(value))

            # TODO: `self.range` is a copy of `value` to avoid side effects,
            # Is it a smart way to avoid them?
            value = np.copy(np.asarray(value))

            if self._domain is not None:
                assert value.size == self._domain.size, (
                    '"domain" and "range" variables must have same size!')

            value.setflags(write=False)
            self._range = value
            self._create_function()

    @property
    def interpolator(self):
        return self._interpolator

    @interpolator.setter
    def interpolator(self, value):
        if value is not None:
            # TODO: Check for interpolator capabilities.
            self._interpolator = value
            self._create_function()

    @property
    def interpolator_args(self):
        return self._interpolator_args

    @interpolator_args.setter
    def interpolator_args(self, value):
        if value is not None:
            assert type(value) in (dict, OrderedDict), (
                ('"{0}" attribute: "{1}" type is not '
                 '"dict" or "OrderedDict"!').format(
                    'interpolator_args', value))

            self._interpolator_args = value
            self._create_function()

    @property
    def extrapolator(self):
        return self._extrapolator

    @extrapolator.setter
    def extrapolator(self, value):
        if value is not None:
            # TODO: Check for extrapolator capabilities.
            self._extrapolator = value
            self._create_function()

    @property
    def extrapolator_args(self):
        return self._extrapolator_args

    @extrapolator_args.setter
    def extrapolator_args(self, value):
        if value is not None:
            assert type(value) in (dict, OrderedDict), (
                ('"{0}" attribute: "{1}" type is not '
                 '"dict" or "OrderedDict"!').format(
                    'extrapolator_args', value))

            self._extrapolator_args = value
            self._create_function()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if value is not None:
            assert type(value) in (str, unicode), (  # noqa
                ('"{0}" attribute: "{1}" type is not '
                 '"str" or "unicode"!').format('name', value))
            self._name = value

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, value):
        raise AttributeError(
            '"{0}" attribute is read only!'.format('function'))

    def _create_function(self):
        if self._domain is not None and self._range is not None:
            with ndarray_write(self._domain), ndarray_write(self._range):
                # TODO: Providing a writeable copy of both `self.domain` and `
                # self.range` to the interpolator to avoid issue regarding `MemoryView`
                # being read-only.
                # https://mail.python.org/pipermail/cython-devel/2013-February/003384.html
                self._function = self._extrapolator(
                    self._interpolator(
                        np.copy(self._domain),
                        np.copy(self._range),
                        **self._interpolator_args),
                    **self._extrapolator_args)
        else:
            def _undefined_signal_interpolator_function(*args, **kwargs):
                raise RuntimeError(
                    'Underlying signal interpolator function does not exists, '
                    'please ensure you defined both '
                    '"domain" and "range" variables!')

            self._function = _undefined_signal_interpolator_function

    def __str__(self):
        try:
            return str(tstack((self.domain, self.range)))
        except TypeError:
            return super(Signal, self).__str__()

    def __repr__(self):
        try:
            representation = repr(tstack((self.domain, self.range)))
            representation = representation.replace(
                'array', self.__class__.__name__)
            representation = representation.replace(
                '       [', '{0}['.format(
                    ' ' * (len(self.__class__.__name__) + 2)))

            return representation
        except TypeError:
            return super(Signal, self).__repr__()

    def __getitem__(self, x):
        if type(x) is slice:
            return self._range[x]
        else:
            return self._function(x)

    def __setitem__(self, x, value):
        if type(x) is slice:
            with ndarray_write(self._range):
                self._range[x] = value
        else:
            with ndarray_write(self._domain), ndarray_write(self._range):
                x = np.atleast_1d(x)
                value = np.resize(value, x.shape)

                # Matching domain, replacing existing `self.range`.
                mask = np.in1d(x, self._domain)
                x_m = x[mask]
                indexes = np.searchsorted(self._domain, x_m)
                self._range[indexes] = value[mask]

                # Non matching domain, inserting into existing `self.domain`
                # and `self.range`.
                x_nm = x[~mask]
                indexes = np.searchsorted(self._domain, x_nm)
                if indexes.size != 0:
                    self._domain = np.insert(self._domain, indexes, x_nm)
                    self._range = np.insert(self._range, indexes, value[~mask])

        self._create_function

    def __contains__(self, x):
        return np.all(np.where(
            np.logical_and(
                x >= np.min(self._domain),
                x <= np.max(self._domain)),
            True,
            False))

    def __eq__(self, x):
        if isinstance(x, self.__class__):
            if all((np.array_equal(self._domain, x.domain),
                    np.array_equal(self._range, x.range),
                    isinstance(self._interpolator, x.interpolator.__class__),
                    self._interpolator_args == x.interpolator_args,
                    isinstance(self._extrapolator, x.extrapolator.__class__),
                    self._extrapolator_args == x.extrapolator_args)):
                return True

        return False

    def __neq__(self, x):
        return not (self == x)

    def _fill_domain_nan(self, method='Interpolation', default=0):
        with ndarray_write(self._domain):
            self._domain = fill_nan(self._domain, method, default)
            self._create_function()

    def _fill_range_nan(self, method='Interpolation', default=0):
        with ndarray_write(self._range):
            self._range = fill_nan(self._range, method, default)
            self._create_function()

    def _arithmetical_operation(self, x, operator, in_place=False):
        operator, ioperator = {
            '+': (add, iadd),
            '-': (sub, isub),
            '*': (mul, imul),
            '/': (div, idiv),
            '**': (pow, ipow)}[operator]

        if in_place:
            if isinstance(x, self.__class__):
                with ndarray_write(self._domain), ndarray_write(self._range):
                    self[self._domain] = operator(self._range, x[self._domain])

                    exclusive_or = np.setxor1d(self._domain, x.domain)
                    self[exclusive_or] = np.full(exclusive_or.shape, np.nan)
                    # Previous implementation, fails with interpolation.
                    # self[x.domain] = operator(self[x.domain], x.range)
            else:
                with ndarray_write(self._range):
                    self.range = ioperator(self.range, x)

            return self
        else:
            copy = ioperator(self.copy(), x)

            return copy

    def __iadd__(self, x):
        return self._arithmetical_operation(x, '+', True)

    def __add__(self, x):
        return self._arithmetical_operation(x, '+')

    def __isub__(self, x):
        return self._arithmetical_operation(x, '-', True)

    def __sub__(self, x):
        return self._arithmetical_operation(x, '-')

    def __imul__(self, x):
        return self._arithmetical_operation(x, '*', True)

    def __mul__(self, x):
        return self._arithmetical_operation(x, '*')

    def __idiv__(self, x):
        return self._arithmetical_operation(x, '/', True)

    def __div__(self, x):
        return self._arithmetical_operation(x, '/')

    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __ipow__(self, x):
        return self._arithmetical_operation(x, '**')

    def __pow__(self, x):
        return self._arithmetical_operation(x, '**')

    def copy(self):
        return deepcopy(self)

    def fill_nan(self, method='Interpolation', default=0):
        self._fill_domain_nan(method, default)
        self._fill_range_nan(method, default)

    def uncertainty(self, x):
        n = nearest(self._domain, x)

        return np.abs(x - n)


def test_empty_object_initialisation():
    cs1 = Signal()
    try:
        print(cs1[0])
    except RuntimeError as error:
        print(error)

    domain = np.arange(0, 1000, 100)
    cs1 = Signal(domain=domain)
    try:
        print(cs1[0])
    except RuntimeError as error:
        print(error)

    range = np.linspace(1, 10, domain.size)
    cs1 = Signal(range, domain)
    assert cs1[0] == 1

    try:
        cs1 = Signal(range, [])
    except AssertionError as error:
        print(error)


def test_object_initialisation():
    domain = np.arange(0, 1000, 100)
    domain_a = np.linspace(0, 1, 10)
    range = np.linspace(1, 10, domain.size)

    data = zip(domain, range)

    cs1 = Signal(range, domain)
    assert re.match('Signal \(\d+\)', cs1.name)
    np.testing.assert_array_equal(cs1.range, range)
    np.testing.assert_array_equal(cs1.domain, domain)

    cs1 = Signal(data)
    np.testing.assert_array_equal(cs1.range, range)
    np.testing.assert_array_equal(cs1.domain, domain)

    cs1 = Signal(data, domain_a)
    np.testing.assert_array_equal(cs1.range, range)
    np.testing.assert_array_equal(cs1.domain, domain_a)

    cs1 = Signal(Signal(data))
    np.testing.assert_array_equal(cs1.range, range)
    np.testing.assert_array_equal(cs1.domain, domain)

    cs1 = Signal(Series(range, domain))
    np.testing.assert_array_equal(cs1.range, range)
    np.testing.assert_array_equal(cs1.domain, domain)

    cs1 = Signal(Series(range, domain, name='D65'))
    assert cs1.name == 'D65'
    np.testing.assert_array_equal(cs1.range, range)
    np.testing.assert_array_equal(cs1.domain, domain)


def test_copy_operations():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)
    cs2 = cs1.copy()
    assert id(cs1) != id(cs2)


def test_getter():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)
    np.testing.assert_approx_equal(cs1[150.25], 2.5025)

    np.testing.assert_array_almost_equal(
        cs1[np.linspace(100, 400, 10)],
        [2., 2.33333333, 2.66666667, 3., 3.33333333,
         3.66666667, 4., 4.33333333, 4.66666667, 5.])

    np.testing.assert_array_almost_equal(
        cs1[0:3],
        [1., 2., 3.])


def test_setter():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)

    cs1[10] = np.pi
    np.testing.assert_array_almost_equal(
        cs1.domain,
        [0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    np.testing.assert_array_almost_equal(
        cs1.range,
        [1, 3.14159265, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    cs1[(200, 300)] = np.pi
    np.testing.assert_array_almost_equal(
        cs1.domain,
        [0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    np.testing.assert_array_almost_equal(
        cs1.range,
        [1, 3.14159265, 2, 3.14159265, 3.14159265, 5, 6, 7, 8, 9, 10])

    cs1[(0, 850)] = np.pi
    np.testing.assert_array_almost_equal(
        cs1.domain,
        [0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 850, 900])
    np.testing.assert_array_almost_equal(
        cs1.range,
        [3.14159265, 3.14159265, 2, 3.14159265, 3.14159265, 5, 6, 7, 8, 9,
         3.14159265, 10])

    cs1[0:9] = -np.pi
    np.testing.assert_array_almost_equal(
        cs1.domain,
        [0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 850, 900])
    np.testing.assert_array_almost_equal(
        cs1.range,
        [-3.14159265, -3.14159265, -3.14159265, -3.14159265,
         -3.14159265, -3.14159265, -3.14159265, -3.14159265,
         -3.14159265, 9, 3.14159265, 10])

    cs1[(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)] = np.pi
    np.testing.assert_array_almost_equal(
        cs1.domain,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200,
         300, 400, 500, 600, 700, 800, 850, 900])
    np.testing.assert_array_almost_equal(
        cs1.range,
        [-3.14159265, 3.14159265, 3.14159265, 3.14159265,
         3.14159265, 3.14159265, 3.14159265, 3.14159265,
         3.14159265, 3.14159265, 3.14159265, -3.14159265,
         -3.14159265, -3.14159265, -3.14159265, -3.14159265,
         -3.14159265, -3.14159265, 9, 3.14159265, 10])


def test_contains():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)
    assert 110 in cs1
    assert (110, 1000) not in cs1


def test_equality():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)
    cs2 = cs1.copy()
    assert cs1 == cs2

    cs1[0] = 0.1
    assert cs1 != cs2

    cs2 = cs1.copy()
    cs1.interpolator = NullInterpolator
    assert cs1 != cs2

    cs1.interpolator = LinearInterpolator
    assert cs1 == cs2


def test_NullInterpolator():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain, interpolator=NullInterpolator)
    assert cs1[100] == 2

    np.testing.assert_array_equal(cs1[100.1, 500], [np.nan, 6.])

    assert cs1[100.0000001] == 2


def test_uncertainty():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)

    np.testing.assert_array_equal(
        cs1.uncertainty((0.1, 150, 200)),
        [0.1, 50, 0])


def test_arithmetical_operations_with_matching_domain():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)
    cs2 = Signal(range, domain)

    cs1 += cs2
    np.testing.assert_array_almost_equal(cs1.domain, domain)
    np.testing.assert_array_almost_equal(cs1.range, range * 2)

    cs1 += 1
    np.testing.assert_array_almost_equal(cs1.domain, domain)
    np.testing.assert_array_almost_equal(cs1.range, range * 2 + 1)

    cs1 -= np.ones(domain.size)
    np.testing.assert_array_almost_equal(cs1.domain, domain)
    np.testing.assert_array_almost_equal(cs1.range, range * 2)

    cs2 = cs1 + cs1
    np.testing.assert_array_almost_equal(cs2.range, range * 4)


def test_arithmetical_operations_with_mismatching_domain():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)
    cs2 = Signal(range, domain + 400)
    cs1 += cs2
    np.testing.assert_array_almost_equal(
        cs1.domain,
        np.array(
            [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
             1300]))
    np.testing.assert_array_almost_equal(
        cs1.range,
        np.array(
            [np.nan, np.nan, np.nan, np.nan, 6, 8, 10, 12, 14, 16, np.nan,
             np.nan, np.nan, np.nan]))

    cs1 += 1
    np.testing.assert_array_almost_equal(
        cs1.range,
        np.array(
            [np.nan, np.nan, np.nan, np.nan, 7, 9, 11, 13, 15, 17, np.nan,
             np.nan, np.nan, np.nan]))

    cs1 -= np.ones(cs1.domain.size)
    np.testing.assert_array_almost_equal(
        cs1.range,
        np.array(
            [np.nan, np.nan, np.nan, np.nan, 6, 8, 10, 12, 14, 16, np.nan,
             np.nan, np.nan, np.nan]))

    cs2 = cs1 + cs2
    np.testing.assert_array_almost_equal(
        cs2.range,
        np.array(
            [np.nan, np.nan, np.nan, np.nan, 7, 10, 13, 16, 19, 22, np.nan,
             np.nan, np.nan, np.nan]))

    cs1.interpolator = CubicSplineInterpolator
    cs2 = cs1 + cs1
    np.testing.assert_array_almost_equal(
        cs2.range,
        np.array(
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))

    cs1.fill_nan('constant')
    cs2 = cs1 + cs1
    np.testing.assert_array_almost_equal(
        cs2.range,
        np.array(
            [0, 0, 0, 0, 12, 16, 20, 24, 28, 32, 0, 0, 0, 0]))


# Spectral Power Distribution

from colour.colorimetry.dataset.illuminants.spds import (
    ILLUMINANTS_RELATIVE_SPDS_DATA)


class Spectrum(Signal):
    def __init__(self, *args, **kwargs):
        # TODO: Define relevant default for spectral computations.
        settings = {
            'interpolator': PchipInterpolator,
            'extrapolator_args': {'left': None, 'right': None}}

        settings.update(kwargs)

        super(Spectrum, self).__init__(*args, **settings)

    @property
    def wavelengths(self):
        return self.domain

    @wavelengths.setter
    def wavelengths(self, value):
        self.domain = value

    @property
    def values(self):
        return self.range

    @values.setter
    def values(self, value):
        self.range = value

    @property
    def title(self):
        if self._title is not None:
            return self._title
        else:
            return self._name

    @title.setter
    def title(self, value):
        if value is not None:
            assert type(value) in (str, unicode), (  # noqa
                ('"{0}" attribute: "{1}" type is not '
                 '"str" or "unicode"!').format('title', value))
        self._title = value

    def normalise(self, factor=100):
        self.range = (self.range * (1 / np.max(self.range))) * factor

        return self


def test_repr():
    s1 = Spectrum(ILLUMINANTS_RELATIVE_SPDS_DATA['A'])
    print(repr(s1))


def test_normalise():
    s1 = Spectrum(ILLUMINANTS_RELATIVE_SPDS_DATA['C'])
    print(s1.normalise())


test_empty_object_initialisation()
test_object_initialisation()
test_copy_operations()
test_getter()
test_setter()
test_contains()
test_equality()
test_NullInterpolator()
test_uncertainty()
test_arithmetical_operations_with_matching_domain()
test_arithmetical_operations_with_mismatching_domain()
test_repr()
test_normalise()
