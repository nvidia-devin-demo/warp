# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Particle Bounce
#
# Shows how to simulate particles falling under gravity and bouncing off a
# floor plane. Each substep integrates velocities and positions with
# semi-implicit Euler, then projects penetrating particles back onto the
# floor and reflects their normal velocity using a restitution coefficient.
#
# FMA contraction is disabled for this module so the kernel produces
# bitwise-identical results on CPU and CUDA devices.
#
###########################################################################

import numpy as np

import warp as wp
import warp.render

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def integrate_particles(
    gravity: wp.vec3,
    floor_height: float,
    radius: float,
    restitution: float,
    friction: float,
    dt: float,
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
):
    tid = wp.tid()

    v_new = v[tid] + gravity * dt
    x_new = x[tid] + v_new * dt

    # contact with the floor plane y = floor_height
    min_height = floor_height + radius
    if x_new[1] < min_height:
        x_new = wp.vec3(x_new[0], min_height, x_new[2])

        if v_new[1] < 0.0:
            v_new = wp.vec3(v_new[0] * friction, -v_new[1] * restitution, v_new[2] * friction)

    x[tid] = x_new
    v[tid] = v_new


class Example:
    def __init__(self, stage_path="example_particle_bounce.usd", num_particles=1024, seed=42):
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.gravity = wp.vec3(0.0, -9.81, 0.0)
        self.floor_height = 0.0
        self.radius = 0.05
        self.restitution = 0.8
        self.friction = 0.98

        # initial state is generated on the host so every device starts from identical data
        rng = np.random.default_rng(seed)
        positions = rng.uniform(low=(-2.0, 0.5, -2.0), high=(2.0, 4.0, 2.0), size=(num_particles, 3))
        velocities = rng.uniform(low=(-1.0, -2.0, -1.0), high=(1.0, 2.0, 1.0), size=(num_particles, 3))

        self.x = wp.array(positions, dtype=wp.vec3)
        self.v = wp.array(velocities, dtype=wp.vec3)

        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = None

        # capture the substep loop into a CUDA graph to reduce launch overhead
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                integrate_particles,
                dim=self.x.shape[0],
                inputs=[
                    self.gravity,
                    self.floor_height,
                    self.radius,
                    self.restitution,
                    self.friction,
                    self.sim_dt,
                ],
                outputs=[self.x, self.v],
            )

    def step(self):
        with wp.ScopedTimer("step", active=False):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_ground()
            self.renderer.render_points(
                points=self.x.numpy(), radius=self.radius, name="particles", colors=(0.8, 0.3, 0.2)
            )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_particle_bounce.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--num-particles", type=int, default=1024, help="Total number of particles.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_particles=args.num_particles)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
