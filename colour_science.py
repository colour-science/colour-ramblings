#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import matplotlib.pyplot
import mpl_toolkits.mplot3d
import warnings
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.patches import FancyArrowPatch, Rectangle

import colour.plotting.common
from colour.plotting import *

colour.plotting.common.DEFAULT_FIGURE_WIDTH = 20
colour.plotting.common.DEFAULT_FIGURE_HEIGHT = (
    colour.plotting.common.DEFAULT_FIGURE_WIDTH * 27 / 64)
colour.plotting.common.DEFAULT_FIGURE_SIZE = (
    colour.plotting.common.DEFAULT_FIGURE_WIDTH,
    colour.plotting.common.DEFAULT_FIGURE_HEIGHT)

DEFAULT_FIGURE_SIZE = colour.plotting.common.DEFAULT_FIGURE_SIZE

warnings.simplefilter('ignore')


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = mpl_toolkits.mplot3d.proj3d.proj_transform(
            xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def trim(a, threshold=1):
    c = np.where(np.average(a, axis=0) != 1)[0]
    r = np.where(np.average(a, axis=1) != 1)[0]

    trim_window = min(r), max(r), min(c), max(c)

    return a[trim_window[0]:trim_window[1] + 1,
           trim_window[2]:trim_window[3] + 1,
           ...]


def electromagnetic_wave_plot(extent=np.pi,
                              period=1,
                              amplitude=1,
                              fields_colour=(
                                      (1.0, 0.5, 0.0),
                                      (0.01, 0.78, 1.0)),
                              zoom_factor=1.75,
                              legend_location=(0.385, 0.285),
                              **kwargs):
    figure = canvas(**kwargs)
    axes = figure.gca(projection='3d')

    u = np.linspace(-extent, extent, extent * period * 250)
    v, w = np.sin(u * period) * amplitude, u * 0
    axes.add_artist(
        Arrow3D((-extent * 1.25, extent * 1.25),
                (0, 0),
                (0, 0),
                mutation_scale=20,
                linewidth=2,
                arrowstyle='-|>',
                color='black',
                alpha=0.5))

    field_points = np.array((colour.tstack((-u, v, w)),
                             colour.tstack((u, w, v))))
    polygons = Poly3DCollection(field_points,
                                edgecolors='black',
                                facecolors=fields_colour)
    polygons.set_alpha(0.75)
    axes.add_collection3d(polygons)

    for i, axis in enumerate('xyz'):
        min_a = np.min(field_points[..., i]) / zoom_factor
        max_a = np.max(field_points[..., i]) / zoom_factor
        getattr(axes, 'set_{}lim'.format(axis))((min_a, max_a))

    u = np.linspace(-extent, extent, extent * period * 4)
    v, w = np.sin(u * period) * amplitude, u * 0
    line_points_a = np.dstack((colour.tstack((-u, w, w)),
                               colour.tstack((-u, v, w))))
    line_points_b = np.dstack((colour.tstack((u, w, w)),
                               colour.tstack((u, w, v))))

    for line_points, line_style in ((line_points_a, '-'),
                                    (line_points_b, '--')):
        line_points = np.rollaxis(line_points, 2, 1)
        lines = Line3DCollection(
            line_points, color='black', linestyles=line_style)
        lines.set_alpha(0.75)
        axes.add_collection3d(lines)

    for i, field in enumerate(('Magnetic', 'Electric')):
        matplotlib.pyplot.figtext(
            legend_location[0],
            legend_location[1] + i * 0.025,
            '[ ]',
            backgroundcolor=list(fields_colour[i]) + [0.75],
            color=(0, 0, 0, 0))
        matplotlib.pyplot.figtext(
            legend_location[0] + 0.01,
            legend_location[1] + i * 0.025,
            '{0} Field'.format(field),
            color='black')

    settings = {
        'camera_aspect': 'equal',
        'no_axes': True}
    settings.update(kwargs)

    camera(**settings)
    decorate(**settings)

    return display(**settings)


def electromagnetic_wave_image():
    filename = 'colour_science/Electromagnetic_Wave_001.png'

    electromagnetic_wave_plot(extent=np.pi * 2,
                              amplitude=3,
                              elevation=22.5,
                              azimuth=-45,
                              figure_size=(30, 30 * 27 / 64),
                              filename=filename)

    colour.write_image(trim(colour.read_image(filename))[..., 0:3], filename)

    return filename
