import jax.numpy as jnp
from flax import struct

# GRID: [tile, color]
NUM_LAYERS = 2
NUM_TILES = 20
NUM_COLORS = 39
NUM_ACTIONS = 6


# TODO: do we really need END_OF_MAP? seem like unseen can be used instead...
# enums, kinda...
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



# Only ~100 combinations so far, better to preallocate them
TILES_REGISTRY = []
for tile_id in range(NUM_TILES):
    for color_id in range(NUM_COLORS):
        TILES_REGISTRY.append((tile_id, color_id))
TILES_REGISTRY = jnp.array(TILES_REGISTRY, dtype=jnp.uint8).reshape(NUM_TILES, NUM_COLORS, -1)


DIRECTIONS = jnp.array(
    (
        (-1, 0),  # Up
        (0, 1),  # Right
        (1, 0),  # Down
        (0, -1),  # Left
    )
)


WALKABLE = jnp.array(
    (
        Tiles.EMPTY,
        Tiles.FLOOR,
        Tiles.GOAL,
        Tiles.DOOR_OPEN,
    )
)

PICKABLE = jnp.array(
    (
        Tiles.BALL,
        Tiles.SQUARE,
        Tiles.PYRAMID,
        Tiles.KEY,
        Tiles.HEX,
        Tiles.STAR,
        Tiles.DIAMOND,
        Tiles.PARALLELOGRAM,
        Tiles.PENTAGON,
        Tiles.CROSS,
    )
)

FREE_TO_PUT_DOWN = jnp.array(
    (
        Tiles.EMPTY,
        Tiles.FLOOR,
    )
)

LOS_BLOCKING = jnp.array(
    (
        Tiles.WALL,
        Tiles.DOOR_LOCKED,
        Tiles.DOOR_CLOSED,
    )
)
