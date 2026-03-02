from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
 
n = 7
center = (3,3)
 
fig, ax = plt.subplots(figsize=(7,7), dpi=200)
 
# grid
for r in range(n):
    for c in range(n):
        ax.add_patch(Rectangle((c, n-1-r), 1, 1, fill=False, linewidth=1))
 
# draw circle for ranged unit (no label text)
cr, cc = center
circle = Circle((cc+0.5, (n-1-cr)+0.5), 0.35, facecolor="blue", edgecolor="blue", linewidth=2)
ax.add_patch(circle)
 
# labels for indices only (skip center)
for r in range(n):
    for c in range(n):
        idx = r*n + c
        x = c + 0.5
        y = (n-1-r) + 0.5
        if (r,c) != center:
            ax.text(x, y, str(idx), ha="center", va="center", fontsize=9)
 
ax.set_xlim(0,n)
ax.set_ylim(0,n)
ax.set_aspect("equal")
ax.axis("off")
 
png_path = "images/data/relative_attack_position_v3.png"
pdf_path = "images/data/relative_attack_position_v3.pdf"
svg_path = "images/data/relative_attack_position_v3.svg"
plt.savefig(png_path, bbox_inches="tight", pad_inches=0.25)
plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.25)
plt.savefig(svg_path, bbox_inches="tight", pad_inches=0.25, format="svg")
plt.close(fig)
 
png_path, pdf_path, svg_path
