# TODO: this is rendering mostly ported or adapted from the original Minigrid. A bit dirty right now...
import functools
import math

import numpy as np

from ..core.constants import Colors, Tiles
from .utils import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_hexagon,
    point_in_rect,
    point_in_triangle,
    point_in_line,
    rotate_fn,
)

COLORS_MAP = {
    Colors.RED: np.array((255, 0, 0)),
    Colors.GREEN: np.array((0, 255, 0)),
    Colors.BLUE: np.array((0, 0, 255)),
    Colors.PURPLE: np.array((112, 39, 195)),
    Colors.YELLOW: np.array((255, 255, 0)),
    Colors.GREY: np.array((100, 100, 100)),
    Colors.BLACK: np.array((0, 0, 0)),
    Colors.ORANGE: np.array((255, 140, 0)),
    Colors.WHITE: np.array((255, 255, 255)),
    Colors.BROWN: np.array((160, 82, 45)),
    Colors.PINK: np.array((225, 20, 147)),
    Colors.CYAN: np.array((0, 255, 255)),
    Colors.MAGENTA: np.array((255, 0, 255)),
    Colors.LIME: np.array((191, 255, 0)),
    Colors.TEAL: np.array((0, 128, 128)),
    Colors.MAROON: np.array((128, 0, 0)),
    Colors.NAVY: np.array((0, 0, 128)),
    Colors.TURQUOISE: np.array((64, 224, 208)),
    Colors.CHARTREUSE: np.array((127, 255, 0)),
    Colors.OLIVE: np.array((128, 128, 0)),
    Colors.MINT: np.array((189, 252, 201)),
    Colors.CORAL: np.array((255, 127, 80)),
    Colors.CRIMSON: np.array((220, 20, 60)),
    Colors.INDIGO: np.array((75, 0, 130)),
    Colors.BEIGE: np.array((245, 245, 220)),
    Colors.LAVENDER: np.array((230, 230, 250)),
    Colors.KHAKI: np.array((240, 230, 140)),
    Colors.JADE: np.array((0, 168, 107)),
    Colors.FUCHSIA: np.array((255, 0, 255)),
    Colors.SCARLET: np.array((255, 36, 0)),
    Colors.AQUA: np.array((0, 255, 255)),
    Colors.MAUVE: np.array((224, 176, 255)),
    Colors.PEARL: np.array((234, 224, 200)),
    Colors.RUBY: np.array((224, 17, 95)),
    Colors.AMBER: np.array((255, 191, 0)),
    Colors.EMERALD: np.array((80, 200, 120)),
}


def _render_floor(img, color):
    # draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    # draw tile
    fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), COLORS_MAP[color] / 2)

    # # other grid lines (was used for paper visualizations)
    # fill_coords(img, point_in_rect(1 - 0.031, 1, 0, 1), (100, 100, 100))
    # fill_coords(img, point_in_rect(0, 1, 1 - 0.031, 1), (100, 100, 100))
    #


def _render_wall(img, color):
    fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS_MAP[color])


def _render_ball(img, color):
    _render_floor(img, Colors.BLACK)
    fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS_MAP[color])


def _render_square(img, color):
    _render_floor(img, Colors.BLACK)
    fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), COLORS_MAP[color])


def _render_pyramid(img, color):
    _render_floor(img, Colors.BLACK)
    tri_fn = point_in_triangle(
        (0.15, 0.8),
        (0.85, 0.8),
        (0.5, 0.2),
    )
    fill_coords(img, tri_fn, COLORS_MAP[color])


def _render_hex(img, color):
    _render_floor(img, Colors.BLACK)
    fill_coords(img, point_in_hexagon(0.35), COLORS_MAP[color])

def _render_diamond(img, color):
    _render_floor(img, Colors.BLACK)
    tri_fn1 = point_in_triangle((0.5, 0.2), (0.2, 0.5), (0.5, 0.8))
    tri_fn2 = point_in_triangle((0.5, 0.2), (0.8, 0.5), (0.5, 0.8))
    fill_coords(img, lambda x, y: tri_fn1(x, y) or tri_fn2(x, y), COLORS_MAP[color])

def _render_parallelogram(img, color):
    # Define the four corners of the parallelogram
    p1 = (0.3, 0.3)
    p2 = (0.7, 0.3)
    p3 = (0.8, 0.7)
    p4 = (0.4, 0.7)

    # Create two triangles to form the parallelogram
    tri_fn1 = point_in_triangle(p1, p2, p3)
    tri_fn2 = point_in_triangle(p1, p3, p4)
    _render_floor(img, Colors.BLACK)
    fill_coords(img, lambda x, y: tri_fn1(x, y) or tri_fn2(x, y), COLORS_MAP[color])

def _render_trapezoid(img, color):
    p1 = (0.3, 0.3)
    p2 = (0.7, 0.3)
    p3 = (0.8, 0.7)
    p4 = (0.2, 0.7)

    tri_fn1 = point_in_triangle(p1, p2, p3)
    tri_fn2 = point_in_triangle(p1, p3, p4)
    _render_floor(img, Colors.BLACK)
    fill_coords(img, lambda x, y: tri_fn1(x, y) or tri_fn2(x, y), COLORS_MAP[color])

def _render_pentagon(img, color):
    # Define the five corners of the pentagon
    p1 = (0.5, 0.2)
    p2 = (0.7, 0.4)
    p3 = (0.6, 0.7)
    p4 = (0.4, 0.7)
    p5 = (0.3, 0.4)

    # Create three triangles to form the pentagon
    tri_fn1 = point_in_triangle(p1, p2, p5)
    tri_fn2 = point_in_triangle(p2, p3, p4)
    tri_fn3 = point_in_triangle(p4, p5, p2)
    _render_floor(img, Colors.BLACK)
    fill_coords(img, lambda x, y: tri_fn1(x, y) or tri_fn2(x, y) or tri_fn3(x, y), COLORS_MAP[color])


def _render_cross(img, color):
    _render_floor(img, Colors.BLACK)
    # Define the two rectangles of the cross
    rect_fn1 = point_in_rect(0.45, 0.55, 0.2, 0.8)
    rect_fn2 = point_in_rect(0.2, 0.8, 0.45, 0.55)

    fill_coords(img, lambda x, y: rect_fn1(x, y) or rect_fn2(x, y), COLORS_MAP[color])

def _render_star(img, color):
    # yes, this is a hexagram not a star, but who cares...
    _render_floor(img, Colors.BLACK)
    tri_fn2 = point_in_triangle(
        (0.15, 0.75),
        (0.85, 0.75),
        (0.5, 0.15),
    )
    tri_fn1 = point_in_triangle(
        (0.15, 0.3),
        (0.85, 0.3),
        (0.5, 0.9),
    )
    fill_coords(img, tri_fn1, COLORS_MAP[color])
    fill_coords(img, tri_fn2, COLORS_MAP[color])


def _render_goal(img, color):
    # draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    # draw tile
    fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), COLORS_MAP[color])

    # # other grid lines (was used for paper visualizations)
    # fill_coords(img, point_in_rect(1 - 0.031, 1, 0, 1), (100, 100, 100))
    # fill_coords(img, point_in_rect(0, 1, 1 - 0.031, 1), (100, 100, 100))


def _render_key(img, color):
    _render_floor(img, Colors.BLACK)
    # Vertical quad
    fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), COLORS_MAP[color])
    # Teeth
    fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), COLORS_MAP[color])
    # Ring
    fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), COLORS_MAP[color])
    fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


def _render_door_locked(img, color):
    fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * COLORS_MAP[color])
    # Draw key slot
    fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), COLORS_MAP[color])


def _render_door_closed(img, color):
    fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
    fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))
    # Draw door handle
    fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), COLORS_MAP[color])


def _render_door_open(img, color):
    _render_floor(img, Colors.BLACK)
    # draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    # draw door
    fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), COLORS_MAP[color])
    fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))


def _render_player(img, direction):
    tri_fn = point_in_triangle(
        (0.12, 0.19),
        (0.87, 0.50),
        (0.12, 0.81),
    )
    # Rotate the agent based on its direction
    tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (direction - 1))
    fill_coords(img, tri_fn, COLORS_MAP[Colors.RED])


TILES_FN_MAP = {
    Tiles.FLOOR: _render_floor,
    Tiles.WALL: _render_wall,
    Tiles.BALL: _render_ball,
    Tiles.SQUARE: _render_square,
    Tiles.PYRAMID: _render_pyramid,
    Tiles.HEX: _render_hex,
    Tiles.STAR: _render_star,
    Tiles.GOAL: _render_goal,
    Tiles.KEY: _render_key,
    Tiles.DOOR_LOCKED: _render_door_locked,
    Tiles.DOOR_CLOSED: _render_door_closed,
    Tiles.DOOR_OPEN: _render_door_open,
    Tiles.EMPTY: lambda img, color: img,
    Tiles.DIAMOND: _render_diamond,
    Tiles.PARALLELOGRAM: _render_parallelogram,
    Tiles.TRAPEZOID: _render_trapezoid,
    Tiles.PENTAGON: _render_pentagon,
    Tiles.CROSS: _render_cross
,
}


# TODO: add highlight for can_see_through_walls=Fasle
def get_highlight_mask(grid, agent, view_size):
    mask = np.zeros((grid.shape[0] + 2 * view_size, grid.shape[1] + 2 * view_size), dtype=np.bool_)
    agent_y, agent_x = agent.position + view_size

    if agent.direction == 0:
        y, x = agent_y - view_size + 1, agent_x - (view_size // 2)
    elif agent.direction == 1:
        y, x = agent_y - (view_size // 2), agent_x
    elif agent.direction == 2:
        y, x = agent_y, agent_x - (view_size // 2)
    elif agent.direction == 3:
        y, x = agent_y - (view_size // 2), agent_x - view_size + 1
    else:
        raise RuntimeError("Unknown direction")

    mask[y : y + view_size, x : x + view_size] = True
    mask = mask[view_size:-view_size, view_size:-view_size]
    assert mask.shape == (grid.shape[0], grid.shape[1])

    return mask


@functools.cache
def render_tile(tile, agent_direction=None, highlight=False, tile_size=32, subdivs=3):
    img = np.full((tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8, fill_value=255)
    # draw tile
    TILES_FN_MAP[tile[0]](img, tile[1])
    # draw agent if on this tile
    if agent_direction is not None:
        _render_player(img, agent_direction)

    if highlight:
        highlight_img(img, alpha=0.2)

    # downsample the image to perform supersampling/anti-aliasing
    img = downsample(img, subdivs)

    return img


# WARN: will NOT work under jit and needed for debugging/presentation mainly.
def render(grid, agent=None, view_size=7, tile_size=32):
    grid = np.asarray(grid)

    # compute the total grid size
    height_px = grid.shape[0] * tile_size
    width_px = grid.shape[1] * tile_size

    img = np.full((height_px, width_px, 3), dtype=np.uint8, fill_value=-1)

    # compute agent fov highlighting
    if agent is not None:
        highlight_mask = get_highlight_mask(grid, agent, view_size)

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if agent is not None and np.array_equal((y, x), agent.position):
                agent_direction = int(agent.direction)
            else:
                agent_direction = None

            tile_img = render_tile(
                tile=tuple(grid[y, x]),
                agent_direction=agent_direction,
                highlight=False if agent is None else highlight_mask[y, x],
                tile_size=tile_size,
            )

            ymin = y * tile_size
            ymax = (y + 1) * tile_size
            xmin = x * tile_size
            xmax = (x + 1) * tile_size
            img[ymin:ymax, xmin:xmax, :] = tile_img

    return img
