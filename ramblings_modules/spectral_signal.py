# coding: utf-8
from __future__ import division, unicode_literals

from contextlib import contextmanager
from copy import deepcopy
from collections import Iterator, Mapping, OrderedDict, Sequence
from operator import (add, mul, pow, sub, iadd, imul, ipow, isub)

try:
    from operator import div, idiv
except ImportError:
    from operator import truediv, itruediv

    div = truediv
    idiv = itruediv
import numpy as np
import re
from pandas import DataFrame, Series
from pprint import pprint

from colour import (CaseInsensitiveMapping, CubicSplineInterpolator,
                    Extrapolator, LinearInterpolator, PchipInterpolator,
                    SpragueInterpolator, as_numeric, is_numeric, is_string,
                    tsplit, tstack, warning)

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
        values[~np.isclose(
            self._x[index],
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
                 'dimensions: "{0}", "{1}"').format(
                     len(self._x), len(self._y)))

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
    'Cubic Spline':
    CubicSplineInterpolator,
    'Linear':
    LinearInterpolator,
    'Null':
    NullInterpolator,
    'Pchip':
    PchipInterpolator,
    'Sprague':
    SpragueInterpolator
})


def nearest_index(a, b):
    index = np.searchsorted(a, b)

    return np.where(
        np.abs(b - a[index - 1]) < np.fabs(b - a[index]), index - 1, index)


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
            np.flatnonzero(mask), np.flatnonzero(~mask), a[~mask])
    elif method.lower() == 'constant':
        a[mask] = default

    return a


def is_pandas_installed():
    try:
        import pandas

        return True
    except ImportError:
        return False


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
            'right': np.nan
        }
        self._name = ('{0} ({1})'.format(self.__class__.__name__, id(self))
                      if name is None else name)

        self.domain, self.range = self.signal_unpack_data(data, domain)

        self.interpolator = interpolator
        self.interpolator_args = interpolator_args
        self.extrapolator = extrapolator
        self.extrapolator_args = extrapolator_args

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
            assert type(value) in (dict, OrderedDict), ((
                '"{0}" attribute: "{1}" type is not '
                '"dict" or "OrderedDict"!').format('interpolator_args', value))

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
            assert type(value) in (dict, OrderedDict), ((
                '"{0}" attribute: "{1}" type is not '
                '"dict" or "OrderedDict"!').format('extrapolator_args', value))

            self._extrapolator_args = value
            self._create_function()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if value is not None:
            assert is_string(value), (  # noqa
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
                        np.copy(self._range), **self._interpolator_args),
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
            representation = representation.replace('array',
                                                    self.__class__.__name__)
            representation = representation.replace('       [', '{0}['.format(
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
        return np.all(
            np.where(
                np.logical_and(x >= np.min(self._domain), x <=
                               np.max(self._domain)), True, False))

    def __eq__(self, x):
        if isinstance(x, self.__class__):
            if all([
                    np.array_equal(self._domain, x.domain),
                    np.array_equal(self._range, x.range),
                    self._interpolator is x.interpolator,
                    self._interpolator_args == x.interpolator_args,
                    self._extrapolator is x.extrapolator,
                    self._extrapolator_args == x.extrapolator_args
            ]):
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
            '**': (pow, ipow)
        }[operator]

        if in_place:
            if isinstance(x, self.__class__):
                with ndarray_write(self._domain), ndarray_write(self._range):
                    self[self._domain] = operator(self._range, x[self._domain])

                    exclusive_or = np.setxor1d(self._domain, x.domain)
                    self[exclusive_or] = np.full(exclusive_or.shape, np.nan)
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

    @staticmethod
    def signal_unpack_data(data=None, domain=None):
        domain_upk, range_upk = None, None
        if isinstance(data, Signal):
            domain_upk = data.domain
            range_upk = data.range
        elif (issubclass(type(data), Sequence) or
              isinstance(data, (tuple, list, np.ndarray, Iterator))):
            data = tsplit(list(data) if isinstance(data, Iterator) else data)
            assert data.ndim in (1, 2), (
                'User "data" must be a 1d or 2d array-like variable!')
            if data.ndim == 1:
                domain_upk, range_upk = np.linspace(0, 1, data.size), data
            else:
                domain_upk, range_upk = data
        elif (issubclass(type(data), Mapping) or
              isinstance(data, (dict, OrderedDict))):
            domain_upk, range_upk = tsplit(sorted(data.items()))
        elif is_pandas_installed():
            if isinstance(data, Series):
                domain_upk = data.index.values
                range_upk = data.values

        if domain is not None and range_upk is not None:
            assert len(domain) == len(range_upk), (
                'User "domain" is not compatible with unpacked range!')
            domain_upk = domain

        return domain_upk, range_upk

    def copy(self):
        return deepcopy(self)

    def fill_nan(self, method='Interpolation', default=0):
        self._fill_domain_nan(method, default)
        self._fill_range_nan(method, default)

    def uncertainty(self, x):
        n = nearest(self._domain, x)

        return np.abs(x - n)


def test_signal_empty_object_initialisation():
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


def test_signal_object_initialisation():
    domain = np.arange(0, 1000, 100)
    domain_a = np.linspace(0, 1, 10)
    range = np.linspace(1, 10, domain.size)

    data = list(zip(domain, range))

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

    cs1 = Signal(Series(range, domain), name='D65')
    assert cs1.name == 'D65'
    np.testing.assert_array_equal(cs1.range, range)
    np.testing.assert_array_equal(cs1.domain, domain)


def test_signal_copy_operations():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)
    cs2 = cs1.copy()
    assert id(cs1) != id(cs2)


def test_signal_getter():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)
    np.testing.assert_approx_equal(cs1[150.25], 2.5025)

    np.testing.assert_array_almost_equal(cs1[np.linspace(100, 400, 10)], [
        2., 2.33333333, 2.66666667, 3., 3.33333333, 3.66666667, 4., 4.33333333,
        4.66666667, 5.
    ])

    np.testing.assert_array_almost_equal(cs1[0:3], [1., 2., 3.])


def test_signal_setter():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)

    cs1[10] = np.pi
    np.testing.assert_array_almost_equal(
        cs1.domain, [0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    np.testing.assert_array_almost_equal(
        cs1.range, [1, 3.14159265, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    cs1[(200, 300)] = np.pi
    np.testing.assert_array_almost_equal(
        cs1.domain, [0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    np.testing.assert_array_almost_equal(cs1.range, [
        1, 3.14159265, 2, 3.14159265, 3.14159265, 5, 6, 7, 8, 9, 10
    ])

    cs1[(0, 850)] = np.pi
    np.testing.assert_array_almost_equal(
        cs1.domain, [0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 850, 900])
    np.testing.assert_array_almost_equal(cs1.range, [
        3.14159265, 3.14159265, 2, 3.14159265, 3.14159265, 5, 6, 7, 8, 9,
        3.14159265, 10
    ])

    cs1[0:9] = -np.pi
    np.testing.assert_array_almost_equal(
        cs1.domain, [0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 850, 900])
    np.testing.assert_array_almost_equal(cs1.range, [
        -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265,
        -3.14159265, -3.14159265, -3.14159265, -3.14159265, 9, 3.14159265, 10
    ])

    cs1[(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)] = np.pi
    np.testing.assert_array_almost_equal(cs1.domain, [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500, 600, 700,
        800, 850, 900
    ])
    np.testing.assert_array_almost_equal(cs1.range, [
        -3.14159265, 3.14159265, 3.14159265, 3.14159265, 3.14159265,
        3.14159265, 3.14159265, 3.14159265, 3.14159265, 3.14159265, 3.14159265,
        -3.14159265, -3.14159265, -3.14159265, -3.14159265, -3.14159265,
        -3.14159265, -3.14159265, 9, 3.14159265, 10
    ])


def test_signal_contains():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)
    assert 110 in cs1
    assert (110, 1000) not in cs1


def test_signal_equality():
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


def test_signal_NullInterpolator():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain, interpolator=NullInterpolator)
    assert cs1[100] == 2

    np.testing.assert_array_equal(cs1[100.1, 500], [np.nan, 6.])

    assert cs1[100.0000001] == 2


def test_signal_uncertainty():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)

    np.testing.assert_array_equal(
        cs1.uncertainty((0.1, 150, 200)), [0.1, 50, 0])


def test_signal_arithmetical_operations_with_matching_domain():
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


def test_signal_arithmetical_operations_with_mismatching_domain():
    domain = np.arange(0, 1000, 100)
    range = np.linspace(1, 10, domain.size)

    cs1 = Signal(range, domain)
    cs2 = Signal(range, domain + 400)
    cs1 += cs2
    np.testing.assert_array_almost_equal(cs1.domain,
                                         np.array([
                                             0, 100, 200, 300, 400, 500, 600,
                                             700, 800, 900, 1000, 1100, 1200,
                                             1300
                                         ]))
    np.testing.assert_array_almost_equal(cs1.range,
                                         np.array([
                                             np.nan, np.nan, np.nan, np.nan, 6,
                                             8, 10, 12, 14, 16, np.nan, np.nan,
                                             np.nan, np.nan
                                         ]))

    cs1 += 1
    np.testing.assert_array_almost_equal(cs1.range,
                                         np.array([
                                             np.nan, np.nan, np.nan, np.nan, 7,
                                             9, 11, 13, 15, 17, np.nan, np.nan,
                                             np.nan, np.nan
                                         ]))

    cs1 -= np.ones(cs1.domain.size)
    np.testing.assert_array_almost_equal(cs1.range,
                                         np.array([
                                             np.nan, np.nan, np.nan, np.nan, 6,
                                             8, 10, 12, 14, 16, np.nan, np.nan,
                                             np.nan, np.nan
                                         ]))

    cs2 = cs1 + cs2
    np.testing.assert_array_almost_equal(cs2.range,
                                         np.array([
                                             np.nan, np.nan, np.nan, np.nan, 7,
                                             10, 13, 16, 19, 22, np.nan,
                                             np.nan, np.nan, np.nan
                                         ]))

    cs1.interpolator = CubicSplineInterpolator
    cs2 = cs1 + cs1
    np.testing.assert_array_almost_equal(
        cs2.range,
        np.array([
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        ]))

    cs1.fill_nan('constant')
    cs2 = cs1 + cs1
    np.testing.assert_array_almost_equal(
        cs2.range, np.array([0, 0, 0, 0, 12, 16, 20, 24, 28, 32, 0, 0, 0, 0]))


# Spectral Power Distribution

from colour.colorimetry.dataset.illuminants.spds import (
    ILLUMINANTS_RELATIVE_SPDS_DATA)


class Spectrum(Signal):
    def __init__(self, *args, **kwargs):
        # TODO: Define relevant default for spectral computations.
        settings = {
            'interpolator': PchipInterpolator,
            'extrapolator_args': {
                'left': None,
                'right': None
            }
        }

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
            assert is_string(value), (  # noqa
                ('"{0}" attribute: "{1}" type is not '
                 '"str" or "unicode"!').format('title', value))
        self._title = value

    def normalise(self, factor=100):
        self.range = (self.range * (1 / np.max(self.range))) * factor

        return self


def test_signal_repr():
    s1 = Spectrum(ILLUMINANTS_RELATIVE_SPDS_DATA['A'])
    print(repr(s1))


def test_signal_normalise():
    s1 = Spectrum(ILLUMINANTS_RELATIVE_SPDS_DATA['C'])
    s1_n = s1.normalise()
    np.testing.assert_almost_equal(
        s1_n.values,
        np.array([
            0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00805867,
            0.16117334, 0.32234668, 1.24909340, 2.17584012, 3.90845354,
            5.64106697, 8.01837376, 10.39568055, 13.86090741, 17.24554759,
            22.16133452, 26.59360142, 32.17019905, 38.19808204, 44.45966637,
            51.01136272, 57.86928842, 64.95285680, 72.14924651, 79.05552422,
            85.26069788, 90.57941816, 94.89080506, 97.91280522, 99.48424531,
            99.92747200, 99.60512531, 99.20219196, 99.36336530, 99.76629865,
            100.00000000, 99.84688533, 99.05713595, 97.26811185, 94.20581836,
            90.33765815, 86.21162060, 82.44016440, 79.62768958, 78.08848416,
            77.99178016, 78.97493755, 80.53831896, 82.27899105, 83.76984447,
            84.77717785, 85.15593521, 84.85776453, 83.89878314, 82.44016440,
            80.70755097, 78.81376420, 76.90386010, 75.10677734, 73.51116125,
            72.28624386, 71.58513982, 71.23861713, 71.06938512, 70.99685712,
            70.96462245, 70.91627045, 70.80344911, 70.75509711, 70.90821178,
            71.07744379, 71.07744379, 70.83568378, 70.28769442, 69.54629704,
            68.74043033, 67.69280361, 66.25030220, 64.63051011, 63.05101136,
            61.48762995, 59.92424853, 58.34474978, 56.73301636, 55.04069627,
            53.42896285, 51.89781610, 50.60842937, 49.56080264, 48.51317592,
            47.70730921, 47.14320251, 46.82085583, 46.74026916, 46.90144250,
            47.14320251, 47.62672254
        ]))


test_signal_empty_object_initialisation()
test_signal_object_initialisation()
test_signal_copy_operations()
test_signal_getter()
test_signal_setter()
test_signal_contains()
test_signal_equality()
test_signal_NullInterpolator()
test_signal_uncertainty()
test_signal_arithmetical_operations_with_matching_domain()
test_signal_arithmetical_operations_with_mismatching_domain()

# test_signal_repr()
# test_signal_normalise()


class MultiSignal(Signal):
    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(MultiSignal, self).__init__(data, domain, **kwargs)

        self._signals = self.multi_signal_unpack_data(data, domain, labels)

    @property
    def domain(self):
        if len(self._signals) != 0:
            return self._signals[self.labels[0]].domain

    @domain.setter
    def domain(self, value):
        if value is not None:
            for signal in self._signals:
                signal.domain = value

    @property
    def range(self):
        if len(self._signals) != 0:
            return tstack([signal.range for signal in self._signals.values()])

    @range.setter
    def range(self, value):
        if value is not None:
            pass

    @property
    def signals(self):
        return self._signals

    @signals.setter
    def signals(self, value):
        if value is not None:
            self._signals = self.multi_signal_unpack_data(value)

    @property
    def labels(self):
        return list(self._signals.keys())

    @labels.setter
    def labels(self, value):
        if value is not None:
            assert len(value) == len(self._signals), (
                '"labels" length does not match "signals" length!')
            self._signals = OrderedDict(
                [(value[i], signal)
                 for i, (_key, signal) in enumerate(self._signals.items())])

    @staticmethod
    def signal_unpack_data(data=None, domain=None):
        return None, None

    @staticmethod
    def multi_signal_unpack_data(data=None, domain=None, labels=None):
        domain_upk, range_upk, signals = None, None, None
        signals = OrderedDict()
        if isinstance(data, MultiSignal):
            signals = data.signals
        elif (issubclass(type(data), Sequence) or
              isinstance(data, (tuple, list, np.ndarray, Iterator))):
            data = tsplit(list(data) if isinstance(data, Iterator) else data)
            assert data.ndim in (1, 2), (
                'User "data" must be a 1d or 2d array-like variable!')
            if data.ndim == 1:
                signals[0] = Signal(data, np.linspace(0, 1, data.size))
            else:
                domain_upk, range_upk = ((data[0], data[1:])
                                         if domain is None else (domain, data))
                for i, range_upk_c in enumerate(range_upk):
                    signals[i] = Signal(range_upk_c, domain_upk)
        elif (issubclass(type(data), Mapping) or
              isinstance(data, (dict, OrderedDict))):
            domain_upk, range_upk = tsplit(sorted(data.items()))
            for i, range_upk in enumerate(tsplit(range_upk)):
                signals[i] = Signal(range_upk, domain_upk)
        elif is_pandas_installed():
            if isinstance(data, Series):
                signals[0] = Signal(data)
            elif isinstance(data, DataFrame):
                # Check order consistency.
                signals = {label: Signal(data, name=label)
                           for label, data in data.to_dict().items()}

        if domain is not None and signals is not None:
            for signal in signals.values():
                assert len(domain) == len(signal.domain), (
                    'User "domain" is not compatible with unpacked signals!')
                signal.domain = domain

        if labels is not None and signals is not None:
            assert len(labels) == len(signals), (
                'User "labels" is not compatible with unpacked signals!')
            signals = OrderedDict(
                [(labels[i], signal)
                 for i, (_key, signal) in enumerate(signals.items())])

        return signals


def test_multi_signal_empty_object_initialisation():
    pass


def test_multi_signal_oject_initialisation():
    domain = np.arange(0, 1000, 100)
    range_1 = np.linspace(1, 10, domain.size)
    range_2 = np.linspace(1, 10, domain.size) + 1
    range_3 = np.linspace(1, 10, domain.size) + 2

    cms1 = MultiSignal(tstack((domain, range_1, range_2, range_3)))
    np.testing.assert_array_equal(cms1.range,
                                  tstack((range_1, range_2, range_3)))
    np.testing.assert_array_equal(cms1.domain, domain)

    cms2 = MultiSignal(range_1)
    np.testing.assert_array_equal(cms2.range, range_1[:, np.newaxis])
    np.testing.assert_array_equal(cms2.domain, np.linspace(0, 1, domain.size))

    cms3 = MultiSignal(range_1, labels=['My Label'])
    assert list(cms3.signals.keys()) == ['My Label']

    cms4 = MultiSignal(
        tstack((range_1, range_2, range_3)), domain / 1000, ('a', 'b', 'c'))
    np.testing.assert_array_equal(cms4.range,
                                  tstack((range_1, range_2, range_3)))
    np.testing.assert_array_equal(cms4.domain, domain / 1000)
    assert list(cms4.signals.keys()) == ['a', 'b', 'c']

    cms5 = MultiSignal(Series(range_1, domain))
    np.testing.assert_array_equal(cms5.range, range_1[:, np.newaxis])
    np.testing.assert_array_equal(cms5.domain, domain)

    dataframe = DataFrame({'aa': range_1, 'bb': range_2, 'cc': range_3})
    cms6 = MultiSignal(dataframe)
    np.testing.assert_array_equal(cms6.range, tstack((range_1, range_2, range_3)))


test_multi_signal_oject_initialisation()
