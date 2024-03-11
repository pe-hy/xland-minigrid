from itertools import product

import jax.numpy as jnp
from flax import struct
import pickle


class Tiles(struct.PyTreeNode):
    EMPTY: int = struct.field(pytree_node=False, default=0)
    END_OF_MAP: int = struct.field(pytree_node=False, default=1)
    UNSEEN: int = struct.field(pytree_node=False, default=2)
    FLOOR: int = struct.field(pytree_node=False, default=3)
    WALL: int = struct.field(pytree_node=False, default=4)
    BALL: int = struct.field(pytree_node=False, default=5)
    SQUARE: int = struct.field(pytree_node=False, default=6)
    PYRAMID: int = struct.field(pytree_node=False, default=7)
    GOAL: int = struct.field(pytree_node=False, default=8)
    KEY: int = struct.field(pytree_node=False, default=9)
    DOOR_LOCKED: int = struct.field(pytree_node=False, default=10)
    DOOR_CLOSED: int = struct.field(pytree_node=False, default=11)
    DOOR_OPEN: int = struct.field(pytree_node=False, default=12)
    HEX: int = struct.field(pytree_node=False, default=13)
    STAR: int = struct.field(pytree_node=False, default=14)
    DIAMOND: int = struct.field(pytree_node=False, default=15)
    PARALLELOGRAM: int = struct.field(pytree_node=False, default=16)
    TRAPEZOID: int = struct.field(pytree_node=False, default=17)
    PENTAGON: int = struct.field(pytree_node=False, default=18)
    CROSS: int = struct.field(pytree_node=False, default=19)


class Colors(struct.PyTreeNode):
    EMPTY: int = struct.field(pytree_node=False, default=0)
    END_OF_MAP: int = struct.field(pytree_node=False, default=1)
    UNSEEN: int = struct.field(pytree_node=False, default=2)
    RED: int = struct.field(pytree_node=False, default=3)
    GREEN: int = struct.field(pytree_node=False, default=4)
    BLUE: int = struct.field(pytree_node=False, default=5)
    PURPLE: int = struct.field(pytree_node=False, default=6)
    YELLOW: int = struct.field(pytree_node=False, default=7)
    GREY: int = struct.field(pytree_node=False, default=8)
    BLACK: int = struct.field(pytree_node=False, default=9)
    ORANGE: int = struct.field(pytree_node=False, default=10)
    WHITE: int = struct.field(pytree_node=False, default=11)
    BROWN: int = struct.field(pytree_node=False, default=12)
    PINK: int = struct.field(pytree_node=False, default=13)
    CYAN: int = struct.field(pytree_node=False, default=14)
    MAGENTA: int = struct.field(pytree_node=False, default=15)
    LIME: int = struct.field(pytree_node=False, default=16)
    TEAL: int = struct.field(pytree_node=False, default=17)
    MAROON: int = struct.field(pytree_node=False, default=18)
    NAVY: int = struct.field(pytree_node=False, default=19)
    TURQUOISE: int = struct.field(pytree_node=False, default=20)
    CHARTREUSE: int = struct.field(pytree_node=False, default=21)
    OLIVE: int = struct.field(pytree_node=False, default=22)
    MINT: int = struct.field(pytree_node=False, default=23)
    CORAL: int = struct.field(pytree_node=False, default=24)
    CRIMSON: int = struct.field(pytree_node=False, default=25)
    INDIGO: int = struct.field(pytree_node=False, default=26)
    BEIGE: int = struct.field(pytree_node=False, default=27)
    LAVENDER: int = struct.field(pytree_node=False, default=28)
    KHAKI: int = struct.field(pytree_node=False, default=29)
    JADE: int = struct.field(pytree_node=False, default=30)
    FUCHSIA: int = struct.field(pytree_node=False, default=31)
    SCARLET: int = struct.field(pytree_node=False, default=32)
    AQUA: int = struct.field(pytree_node=False, default=33)
    MAUVE: int = struct.field(pytree_node=False, default=34)
    PEARL: int = struct.field(pytree_node=False, default=35)
    RUBY: int = struct.field(pytree_node=False, default=36)
    AMBER: int = struct.field(pytree_node=False, default=37)
    EMERALD: int = struct.field(pytree_node=False, default=38)


tiles = [attr for attr in Tiles.__dict__.values() if isinstance(attr, int)]
colors = [attr for attr in Colors.__dict__.values() if isinstance(attr, int)]

id_ranges = [(0, 21), (10000, 10100), (100000, 100100)]
obstacle_ranges = [(11001, 11037), (101001, 101037)]
# Exclude certain tiles and colors
excluded_tiles = [
    Tiles.EMPTY,
    Tiles.END_OF_MAP,
    Tiles.UNSEEN,
    Tiles.FLOOR,
    Tiles.WALL,
    Tiles.GOAL,
    Tiles.KEY,
    Tiles.DOOR_LOCKED,
    Tiles.DOOR_CLOSED,
    Tiles.DOOR_OPEN,
    Tiles.CROSS,
]
excluded_colors = [Colors.EMPTY, Colors.END_OF_MAP, Colors.UNSEEN]

tiles = [tile for tile in tiles if tile not in excluded_tiles]
colors = [color for color in colors if color not in excluded_colors]

# Generate unique combinations of tiles and colors
combinations = list(product(tiles, colors))

# Check if we have enough unique combinations for all IDs
total_ids = sum(end - start for start, end in id_ranges)
if total_ids > len(combinations):
    raise ValueError("Not enough unique combinations for all IDs")

# Assign unique combinations to IDs
id_to_combination = {}
for start, end in id_ranges:
    for id in range(start, end):
        id_to_combination[id] = combinations.pop(0)

cross_products_1 = list(product([11, 10], colors))  # obstacle will only be closed/locked door and some color
cross_products = cross_products_1
print(len(cross_products))

for start, end in obstacle_ranges:
    for id in range(start, end):
        id_to_combination[id] = cross_products.pop(0)

id_to_combination[99999] = (12, 4)

with open("pkls/id_to_combination.pkl", "wb") as f:
    pickle.dump(id_to_combination, f)

print(id_to_combination)
print(len(id_to_combination))
