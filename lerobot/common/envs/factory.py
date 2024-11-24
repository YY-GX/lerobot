#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib

import gymnasium as gym
from omegaconf import DictConfig


def make_env(cfg: DictConfig, n_envs: int | None = None) -> gym.vector.VectorEnv | None:
    """Makes a gym vector environment according to the evaluation config.

    n_envs can be used to override eval.batch_size in the configuration. Must be at least 1.
    """
    if n_envs is not None and n_envs < 1:
        raise ValueError("`n_envs must be at least 1")

    if cfg.env.name == "real_world":
        return

    package_name = f"gym_{cfg.env.name}"

    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(
            f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.env.name}]'`"
        )
        raise e

    gym_handle = f"{package_name}/{cfg.env.task}"
    gym_kwgs = dict(cfg.env.get("gym", {}))

    if cfg.env.get("episode_length"):
        gym_kwgs["max_episode_steps"] = cfg.env.episode_length

    # batched version of the env that returns an observation of shape (b, c)
    env_cls = gym.vector.AsyncVectorEnv if cfg.eval.use_async_envs else gym.vector.SyncVectorEnv
    env = env_cls(
        [
            lambda: gym.make(gym_handle, disable_env_checker=True, **gym_kwgs)
            for _ in range(n_envs if n_envs is not None else cfg.eval.batch_size)
        ]
    )

    return env












"""Utils for evaluating policies in LIBERO simulation environments."""

import os

current_working_directory = os.getcwd()
os.chdir(os.environ['PYTHONPATH'])
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero import benchmark
os.chdir(current_working_directory)



def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    # env_args = {"metadata": None, "observation_space": None, "action_space": None}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_env_factory(task, resolution=256):
    """Returns a callable that initializes the LIBERO environment."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}

    def create_env():
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
        return env

    return create_env, task_description




def make_libero_env(cfg: DictConfig, task_suite_name, task_id: int, resolution=256, n_envs: int | None = None) -> gym.vector.VectorEnv | None:
    """Makes a gym vector environment according to the evaluation config.

    n_envs can be used to override eval.batch_size in the configuration. Must be at least 1.
    """

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)

    # env, task_description = get_libero_env(task, resolution=resolution)
    env, task_description = get_libero_env_factory(task, resolution=resolution)
    print(f"Current task description: {task_description}")

    initial_states = task_suite.get_task_init_states(task_id)
    # env.reset()
    # env.set_init_state(initial_states[0 % initial_states.shape[0]])




    if n_envs is not None and n_envs < 1:
        raise ValueError("`n_envs must be at least 1")

    # batched version of the env that returns an observation of shape (b, c)
    # env_cls = gym.vector.AsyncVectorEnv if cfg.eval.use_async_envs else gym.vector.SyncVectorEnv
    # env = env_cls(
    #     [
    #         lambda: env
    #         for _ in range(n_envs if n_envs is not None else cfg.eval.batch_size)
    #     ]
    # )

    env = SubprocVectorEnv(
        [
            lambda: env
            for _ in range(n_envs if n_envs is not None else cfg.eval.batch_size)
        ], num_envs=n_envs
    )

    # env = SubprocVectorEnv(
    #     [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
    # )

    return env