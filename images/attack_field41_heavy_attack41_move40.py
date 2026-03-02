import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import math


def idx_to_rc(idx: int, n: int) -> tuple[int, int]:
    return idx // n, idx % n


def idx_to_center(idx: int, n: int) -> tuple[float, float]:
    r, c = idx_to_rc(idx, n)
    return c + 0.5, (n - 1 - r) + 0.5


def point_from_center_towards(
    center: tuple[float, float],
    target: tuple[float, float],
    offset: float,
) -> tuple[float, float]:
    dx = target[0] - center[0]
    dy = target[1] - center[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return center
    return center[0] + dx / length * offset, center[1] + dy / length * offset


n = 7

# Middle blue unit
attacker_idx = 24
attacker_radius = 0.35

# Heavy target and movement
heavy_idx = 41
heavy_radius = 0.35
move_to_idx = 40
attack_target_idx = 41

fig, ax = plt.subplots(figsize=(7, 7), dpi=200)

# grid
for r in range(n):
    for c in range(n):
        ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, fill=False, linewidth=1))

# labels for indices only (skip occupied cells)
skip = {attacker_idx, heavy_idx}
for r in range(n):
    for c in range(n):
        idx = r * n + c
        if idx in skip:
            continue
        x = c + 0.5
        y = (n - 1 - r) + 0.5
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=9)

# blue unit in the middle with red/magenta frame
attacker_center = idx_to_center(attacker_idx, n)
attacker = Circle(
    attacker_center,
    attacker_radius,
    facecolor="blue",
    edgecolor="#E83E8C",
    linewidth=2,
)
ax.add_patch(attacker)

# heavy unit on field 41
heavy_center = idx_to_center(heavy_idx, n)
heavy = Circle(
    heavy_center,
    heavy_radius,
    facecolor="#E8E12A",
    edgecolor="#0077FF",
    linewidth=2,
)
ax.add_patch(heavy)

# movement indication: heavy moves to center of cell 40
move_target_center = idx_to_center(move_to_idx, n)
move_start = point_from_center_towards(heavy_center, move_target_center, heavy_radius)
move_end = move_target_center
ax.plot(
    [move_start[0], move_end[0]],
    [move_start[1], move_end[1]],
    color="gray",
    linewidth=1.2,
    solid_capstyle="round",
)

# attack indication: middle unit attacks center of heavy cell 41
attack_target_center = idx_to_center(attack_target_idx, n)
attack_start = point_from_center_towards(attacker_center, attack_target_center, attacker_radius)
attack_end = attack_target_center
ax.plot(
    [attack_start[0], attack_end[0]],
    [attack_start[1], attack_end[1]],
    color="#E83E8C",
    linewidth=2.2,
    solid_capstyle="round",
)

ax.set_xlim(0, n)
ax.set_ylim(0, n)
ax.set_aspect("equal")
ax.axis("off")

png_path = "images/data/attack_field41_heavy_attack41_move40.png"
pdf_path = "images/data/attack_field41_heavy_attack41_move40.pdf"
svg_path = "images/data/attack_field41_heavy_attack41_move40.svg"
plt.savefig(png_path, bbox_inches="tight", pad_inches=0.25)
plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.25)
plt.savefig(svg_path, bbox_inches="tight", pad_inches=0.25, format="svg")
plt.close(fig)

print(png_path, pdf_path, svg_path)
