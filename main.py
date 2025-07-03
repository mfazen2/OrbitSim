import os
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

#
#
#
#need to implement UI still, would definitely help with timescale issues/visualization of key moments in\
#simulation
from PyQt6.QtWidgets import QSlider, QDoubleSpinBox, QPushButton
#
#
#

import cupy as cp
from numba import cuda
from vispy import app, scene

#helper function
def get_orthonorm(n):
    n = n / cp.linalg.norm(n) #force unit vector
    ref = cp.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else cp.array([0.0, 1.0, 0.0])
    tang_x = cp.cross(ref, n)
    if cp.linalg.norm(tang_x) < 1e-8:
        #fallback... idk, suggested by stack overflow
        ref = cp.array([0.0, 0.0, 1.0])
        tang_x = cp.cross(ref, n)
    tang_x /= cp.linalg.norm(tang_x)
    tang_y = -cp.cross(n, tang_x)
    return tang_x, tang_y, n


# init vispy canvas
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'
scatter = scene.visuals.Markers(parent=view.scene, scaling='scene')

#
#
#
# physics params
n_bodies = 6400
n_alive = cp.array([n_bodies],dtype=cp.int32)
soft = 1e-3
timestep = 3000000
#
#
#
#
positions = (cp.random.randn(n_bodies, 3).astype(cp.float32)*4500)
velocities = cp.random.randn(n_bodies, 3).astype(cp.float32)*.0025


alive = cp.ones(n_bodies, dtype=cp.float32)
G = 6.674e-11

#apply large central object
positions[0] = cp.array([0, 0, 0])
velocities[0] = cp.array([0, 0, 0])
masses = cp.random.rand(n_bodies).astype(cp.float32)*5
masses[0] = 256

radii = masses * 4
radii[1:] = cp.log1p(masses[1:] * 100) * 7.5
radii[0] = radii[0]*0.5

def initialize_orbits(positions, velocities, masses, central_index=0):
    central_mass = masses[central_index]
    for i in range(positions.shape[0]):
        if i == central_index:
            continue

        r = positions[i] - positions[central_index]
        r_mag = cp.linalg.norm(r)

        if r_mag == 0:
            continue  # prevent division by zero

        orbital_speed = cp.sqrt(G * central_mass / r_mag)

        # Use a helper to get orthogonal vector
        tang_x, tang_y, _ = get_orthonorm(r)

        velocities[i] = orbital_speed * tang_x



if cp.any(cp.isnan(velocities)):
    print("NaNs detected in velocities!")
if cp.any(cp.isnan(positions)):
    print("NaNs detected in velocities!")


#correct distances prior to simulation to avoid instant collisions and/or physics issues on faster timescales
@cuda.jit
def corrector(coords, rads):
    print("corrector running")
    i = cuda.grid(1)
    n = coords.shape[0]
    if i < n and rads[i] != 0:
        xi, yi, zi = coords[i, 0], coords[i, 1], coords[i, 2]
        radius_i = rads[i]
        for j in range(n):
            if i != j:
                dx = coords[j, 0] - xi
                dy = coords[j, 1] - yi
                dz = coords[j, 2] - zi
                sqr_dist= dx * dx + dy * dy + dz * dz + soft
                radius_j = rads[j]
                min_dist_sqr = (radius_i + radius_j) ** 2
                if sqr_dist < min_dist_sqr:
                    scale_factor =  2*(min_dist_sqr/sqr_dist)
                    coords[j, 0] += dx * scale_factor
                    coords[j, 1] += dy * scale_factor
                    coords[j, 2] += dz * scale_factor





@cuda.jit
def n_body(coords, masses, accs, rads, live,n_live):
    i = cuda.grid(1)
    if i >= n_live[0]:
        return
    idx = live[i]

    ax, ay, az = 0.0, 0.0, 0.0
    x_i,y_i,z_i = coords[idx,0], coords[idx,1], coords[idx,2]
    m_i = masses[idx]
    radius_i = rads[idx]

    for j in range(live.shape[0]):
        jdx = live[j]
        if idx == jdx:
            continue
        #compute dist
        x_j,y_j,z_j = coords[jdx,0], coords[jdx,1], coords[jdx,2]
        m_j = masses[jdx]
        dx = x_j - x_i
        dy = y_j - y_i
        dz = z_j - z_i

        sqr_dist = dx**2 + dy**2 + dz**2 + soft
        if sqr_dist > 1e9: continue
        radius_j = rads[jdx]
        min_dist_sqr = (radius_i + radius_j) ** 2
        if sqr_dist < min_dist_sqr:
            sqr_dist = min_dist_sqr

        inv_d = sqr_dist**(-0.5)
        inv_d3 = inv_d**3

        s = G * m_j * inv_d3

        ax += dx * s
        ay += dy * s
        az += dz * s

    accs[idx,0] = ax
    accs[idx,1] = ay
    accs[idx,2] = az

@cuda.jit
def integrate_kernel(coords, vels, accels, live, dt,n_live):
    i = cuda.grid(1)
    if i >= n_live[0]:
        return
    idx = live[i]
    vels[idx, 0] += accels[idx, 0] * dt
    vels[idx, 1] += accels[idx, 1] * dt
    vels[idx, 2] += accels[idx, 2] * dt
    coords[idx, 0] += vels[idx, 0] * dt
    coords[idx, 1] += vels[idx, 1] * dt
    coords[idx, 2] += vels[idx, 2] * dt

@cuda.jit
def collision_handler(coords, vels, accs, masses, rads, live,n_live,alive):
    i = cuda.grid(1)
    if i >= n_live[0]: return
    idx = live[i]
    for j in range(i + 1, live.shape[0]):
        jdx = live[j]
        if jdx != 0 and idx != jdx:
            dx = coords[jdx, 0] - coords[idx, 0]
            dy = coords[jdx, 1] - coords[idx, 1]
            dz = coords[jdx, 2] - coords[idx, 2]
            r = (dx * dx + dy * dy + dz * dz) ** 0.5

            if r < (rads[idx] + rads[jdx])/2.2:
                # Inelastic collision - choose body #1 to merge
                total_mass = masses[idx] + masses[jdx]


                vels[idx, 0] = (masses[idx] * vels[idx, 0] + masses[jdx] * vels[jdx, 0]) / total_mass
                vels[idx, 1] = (masses[idx] * vels[idx, 1] + masses[jdx] * vels[jdx, 1]) / total_mass
                vels[idx, 2] = (masses[idx] * vels[idx, 2] + masses[jdx] * vels[jdx, 2]) / total_mass
                rads[idx] = (rads[jdx]**3 + rads[idx]**3)**(1/3)
                #remove body j
                if alive[jdx] == 1:
                    masses[jdx] = 0
                    rads[jdx] = 0
                    coords[jdx][0] = 1e9 #A long time ago in a galaxy far, far away
                    alive[jdx] = 0
                    cuda.atomic.sub(n_live, 0, 1)
                    print('should update')
                break



import cupy as cp
from cupy.cuda import pinned_memory
import numpy as np
from vispy import app,scene
threads_per_block = 64
blocks_per_grid = (n_bodies + threads_per_block - 1) // threads_per_block
corrector[blocks_per_grid, threads_per_block](positions,radii)

initialize_orbits(positions, velocities, masses, central_index=0)
color = cp.random.rand(n_bodies,4).astype(cp.float32)
for i in range(n_bodies):
    color[i][3] = 1

color[0][0] = 1
color[0][1] = 1
color[0][2] = 0

view.camera.fov = 85
view.camera.distance = 2000


#prealloc memory to increase efficiency
from cupy.cuda import pinned_memory

host_pos = pinned_memory.alloc_pinned_memory(positions.nbytes)
host_color = pinned_memory.alloc_pinned_memory(color.nbytes)
host_radii = pinned_memory.alloc_pinned_memory(radii.nbytes)

# # Create numpy arrays from pinned memory
host_pos_arr = np.ndarray(positions.shape, dtype=positions.dtype, buffer=host_pos)
host_color_arr = np.ndarray(color.shape, dtype=color.dtype, buffer=host_color)
host_radii_arr = np.ndarray(radii.shape, dtype=radii.dtype, buffer=host_radii)

# Create streams
compute_stream = cp.cuda.Stream()
transfer_stream = cp.cuda.Stream(non_blocking=True)

accelerations = cp.zeros((n_bodies, 3), dtype=cp.float32)

def update(event):
    view.camera.center = (positions[0, 0], positions[0, 1], positions[0, 2])
    print(n_alive)
    #CUDA config
    live_index = cp.where(alive == 1)[0]
    if (len(live_index) == 0):
        return
    threads_per_block = 128
    blocks_per_grid = (len(live_index) + threads_per_block - 1) // threads_per_block

    with compute_stream:
        accelerations.fill(0)
        n_body[blocks_per_grid, threads_per_block](positions, masses, accelerations, radii, live_index, n_alive)
        integrate_kernel[blocks_per_grid, threads_per_block](positions, velocities, accelerations, live_index, timestep,
                                                             n_alive)
        collision_handler[blocks_per_grid, threads_per_block](positions, velocities, accelerations, masses, radii,
                                                              live_index, n_alive, alive)

    #async optimizations
    compute_stream.synchronize()
    with transfer_stream:
        positions.get(out=host_pos_arr, stream=transfer_stream)
        color.get(out=host_color_arr, stream=transfer_stream)
        radii.get(out=host_radii_arr, stream=transfer_stream)
    transfer_stream.synchronize()

    scatter.set_data(
        host_pos_arr,
        edge_color=None,
        face_color=host_color_arr,
        size=host_radii_arr,
    )
    canvas.update()


#anim setup
timer = app.Timer(interval=0.016, connect=update, start=True)
canvas.show()
app.run()


