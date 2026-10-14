# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.examples.core.example_particle_bounce import Example
from warp.tests.unittest_utils import add_function_test, get_test_devices

NUM_PARTICLES = 512
NUM_FRAMES = 120


def run_simulation(device, num_frames=NUM_FRAMES):
    with wp.ScopedDevice(device):
        example = Example(stage_path=None, num_particles=NUM_PARTICLES)
        min_height = np.inf
        for _ in range(num_frames):
            example.step()
            min_height = min(min_height, float(example.x.numpy()[:, 1].min()))
        return example, min_height


def test_particles_stay_above_floor(test, device):
    example, min_height = run_simulation(device)

    contact_height = example.floor_height + example.radius
    test.assertGreaterEqual(min_height, contact_height)

    # after two seconds of simulated time, the particles should have reached the floor and bounced
    v = example.v.numpy()
    test.assertTrue(np.all(np.isfinite(example.x.numpy())))
    test.assertTrue(np.all(np.isfinite(v)))
    test.assertTrue(np.any(v[:, 1] > 0.0))


def test_gpu_matches_cpu(test, device):
    if not wp.get_device(device).is_cuda:
        test.skipTest("Requires a CUDA device to compare against the CPU")

    cpu_example, _ = run_simulation("cpu")
    gpu_example, _ = run_simulation(device)

    np.testing.assert_allclose(gpu_example.x.numpy(), cpu_example.x.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gpu_example.v.numpy(), cpu_example.v.numpy(), rtol=1e-6, atol=1e-6)


devices = get_test_devices()


class TestParticleBounce(unittest.TestCase):
    pass


add_function_test(
    TestParticleBounce, "test_particles_stay_above_floor", test_particles_stay_above_floor, devices=devices
)
add_function_test(TestParticleBounce, "test_gpu_matches_cpu", test_gpu_matches_cpu, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
