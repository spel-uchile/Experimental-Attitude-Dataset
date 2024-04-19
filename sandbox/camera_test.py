import numpy as np
import time
import pyvista as pv
from pyvista import examples

pv.set_plot_theme("document")

# download mesh
mesh = examples.download_cow()

decimated = mesh.decimate_boundary(target_reduction=0.75)

p = pv.Plotter(shape=(1, 2), border=False)
p.subplot(0, 0)
p.add_text("Original mesh", font_size=24)
p.add_mesh(mesh, show_edges=False, color=True)
p.subplot(0, 1)
p.add_text("Decimated version", font_size=24)
p.add_mesh(decimated, color=True, show_edges=False)
p.add_axes()
p.show_grid()
p.link_views()  # link all the views
# Set a camera position to all linked views
p.camera_position = [(15, 5, 0), (0, 0, 0), (0, 1, 0)]

p.open_gif("linked.gif")
# Update camera and write a frame for each updated position
nframe = 450
for i in range(nframe):
    print(i)
    p.camera_position = [
        (15, 0, 0), # position
#         (15 * np.cos(i * np.pi / 45.0), 5.0, 15 * np.sin(i * np.pi / 45.0)),
        (0, 0, -180 + 360 * i / 150), # focal_point
        (0, 1, 0), # view_up
    ]
    p.write_frame()
    time.sleep(0.3)

# Close movie and delete object
p.close()