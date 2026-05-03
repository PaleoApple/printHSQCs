import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.ndimage import zoom

# ----------------------------
# 1. Load and Crop Data
# ----------------------------
dic, data = ng.pipe.read_2D("newHisVl.ft2")
uc_H = ng.pipe.make_uc(dic, data, dim=1)
uc_N = ng.pipe.make_uc(dic, data, dim=0)

h_start, h_end = 9.0, 6.5
n_start, n_end = 130.0, 105.0

x0, x1 = uc_H.i(h_start, "ppm"), uc_H.i(h_end, "ppm")
y0, y1 = uc_N.i(n_start, "ppm"), uc_N.i(n_end, "ppm")

z_raw = np.abs(data[min(y0, y1):max(y0, y1), min(x0, x1):max(x0, x1)])
z_proc = z_raw - z_raw.min()
max_height = 20.0
z_final = (z_proc / z_proc.max()) * max_height

# ----------------------------
# 2. Geometry Setup (The "Secret Sauce")
# ----------------------------
# We define a 10% padding so labels don't get cut off or sit on the very edge
h_range = h_start - h_end
n_range = n_start - n_end
pad_h = h_range * 0.15 
pad_n = n_range * 0.15

# These limits MUST be identical for both SVG and Grid/STL logic
limits = {
    'xlim': (h_start + pad_h, h_end - pad_h),
    'ylim': (n_start + pad_n, n_end - pad_n)
}

# ----------------------------
# 3. Generate the SVG (Text Only)
# ----------------------------
plt.rcParams['svg.fonttype'] = 'path'
fig_svg, ax_svg = plt.subplots(figsize=(10, 10))
ax_svg.set_xlim(*limits['xlim'])
ax_svg.set_ylim(*limits['ylim'])
ax_svg.set_axis_off()

label_color = 'black'

# X-Axis (1H) - Placed slightly above the data boundary
for x_val in np.arange(h_end, h_start + 0.1, 0.5):
    ax_svg.text(x_val, n_start + (pad_n * 0.2), f"{x_val:.1f}", 
                color=label_color, fontsize=16, ha='center', weight='bold')

# Y-Axis (15N) - Placed to the left of the data boundary
for y_val in np.arange(n_end, n_start + 1, 5):
    ax_svg.text(h_start + (pad_h * 0.2), y_val, f"{int(y_val)} ", 
                color=label_color, fontsize=16, va='center', ha='right', weight='bold')

# Legend/Nuclei Label
ax_svg.text(h_start, n_start + (pad_n * 0.6), r"$^{15}$N / $^1$H (ppm)", 
            color=label_color, fontsize=18, ha='right', weight='bold', style='italic')

# CRITICAL: No bbox_inches='tight'. This keeps the 10x10 aspect ratio stable.
fig_svg.savefig('nmr_labels_only.svg', format='svg', transparent=True, pad_inches=0)
plt.close(fig_svg)

# ----------------------------
# 4. Create Grid Mask & Merge with STL Data
# ----------------------------
grid_height = 0.5 # 0.5mm physical height for grid lines
fig_grid, ax_grid = plt.subplots(figsize=(10, 10), dpi=150)
fig_grid.patch.set_facecolor('black')
ax_grid.set_facecolor('black')
ax_grid.set_xlim(*limits['xlim'])
ax_grid.set_ylim(*limits['ylim'])
ax_grid.set_axis_off()

# Draw the white grid on black background
ax_grid.grid(True, color='white', linewidth=1.5)
ax_grid.set_xticks(np.arange(h_end, h_start + 0.1, 0.5))
ax_grid.set_yticks(np.arange(n_end, n_start + 1, 5))

fig_grid.canvas.draw()
grid_mask = np.frombuffer(fig_grid.canvas.tostring_rgb(), dtype=np.uint8)
grid_mask = grid_mask.reshape(fig_grid.canvas.get_width_height()[::-1] + (3,))
grid_mask = np.dot(grid_mask[...,:3], [0.2989, 0.5870, 0.1140]) / 255.0
plt.close(fig_grid)

# Resize grid to match data and apply
grid_resized = zoom(np.flipud(grid_mask), [z_final.shape[0]/grid_mask.shape[0], z_final.shape[1]/grid_mask.shape[1]])
z_final = np.maximum(z_final, grid_resized * grid_height)

# ----------------------------
# 5. Generate STL (100mm x 100mm base)
# ----------------------------
rows, cols = z_final.shape
x = np.linspace(0, 100, cols) 
y = np.linspace(0, 100, rows)
X, Y = np.meshgrid(x, y)

num_tri = (rows - 1) * (cols - 1) * 2
hsqc_mesh = mesh.Mesh(np.zeros(num_tri, dtype=mesh.Mesh.dtype))

counter = 0
for i in range(rows - 1):
    for j in range(cols - 1):
        # Triangle 1
        hsqc_mesh.v0[counter] = [X[i, j], Y[i, j], z_final[i, j]]
        hsqc_mesh.v1[counter] = [X[i+1, j], Y[i+1, j], z_final[i+1, j]]
        hsqc_mesh.v2[counter] = [X[i, j+1], Y[i, j+1], z_final[i, j+1]]
        counter += 1
        # Triangle 2
        hsqc_mesh.v0[counter] = [X[i+1, j], Y[i+1, j], z_final[i+1, j]]
        hsqc_mesh.v1[counter] = [X[i+1, j+1], Y[i+1, j+1], z_final[i+1, j+1]]
        hsqc_mesh.v2[counter] = [X[i, j+1], Y[i, j+1], z_final[i, j+1]]
        counter += 1

hsqc_mesh.save('nmr_peaks_and_grid.stl')
print("Refactored STL and SVG saved. Import both into Bambu Studio for a perfect overlay.")
