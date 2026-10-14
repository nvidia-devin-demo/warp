# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.examples.core.example_particle_bounce import Example
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

NUM_PARTICLES = 512
NUM_FRAMES = 120


def run_example(device):
    """Run the example on ``device`` and return it with the per-frame minimum particle height."""
    with wp.ScopedDevice(device):
        example = Example(stage_path=None, num_particles=NUM_PARTICLES)
        min_heights = np.empty(NUM_FRAMES)
        for i in range(NUM_FRAMES):
            example.step()
            min_heights[i] = example.x.numpy()[:, 1].min()
    return example, min_heights


def test_particles_stay_above_floor(test, device):
    example, min_heights = run_example(device)

    x = example.x.numpy()
    v = example.v.numpy()
    test.assertTrue(np.all(np.isfinite(x)))
    test.assertTrue(np.all(np.isfinite(v)))

    # compare in single precision, matching the kernel arithmetic
    min_height = np.float32(example.floor_height) + np.float32(example.radius)
    test.assertGreaterEqual(min_heights.min(), min_height)

    # two seconds of simulated time is enough for every particle to reach the floor at least once
    test.assertTrue(np.any(min_heights == min_height))
    test.assertTrue(np.any(v[:, 1] > 0.0), "Expected some particles to be moving upward after bouncing")


def test_gpu_matches_cpu(test, device):
    cpu_example, cpu_min_heights = run_example("cpu")
    gpu_example, gpu_min_heights = run_example(device)

    np.testing.assert_allclose(gpu_min_heights, cpu_min_heights, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gpu_example.x.numpy(), cpu_example.x.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gpu_example.v.numpy(), cpu_example.v.numpy(), rtol=1e-6, atol=1e-6)


class TestParticleBounce(unittest.TestCase):
    pass


add_function_test(
    TestParticleBounce, "test_particles_stay_above_floor", test_particles_stay_above_floor, devices=get_test_devices()
)
add_function_test(TestParticleBounce, "test_gpu_matches_cpu", test_gpu_matches_cpu, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
