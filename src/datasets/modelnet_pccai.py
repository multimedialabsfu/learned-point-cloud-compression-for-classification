# NOTE: This module has been adapted from https://github.com/InterDigitalInc/PccAI
#
# See https://github.com/InterDigitalInc/PccAI/blob/main/LICENSE.
# LICENSE is also reproduced below:
#
#
# The copyright in this software is being made available under the BSD License,
# included below. This software may be subject to InterDigital and other third
# party and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2022, InterDigital
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#   * Neither the name of InterDigital nor the names of the Project where this
#   contribution had been made may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS
# LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import os
import os.path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets.modelnet import ModelNet
from torch_geometric.transforms.sample_points import SamplePoints

from compressai.registry import register_dataset

# from pccai.utils.convert_octree import OctreeOrganizer
# import pccai.utils.logger as logger


class ModelNetBase(Dataset):
    """A base ModelNet data loader."""

    def __init__(
        self,
        root,
        split="train",
        name="40",
        num_points=None,
        coord_minmax=None,
        centralize=True,
        random_rotation=False,
        random_permutation=True,
        normalize=True,
        voxelize=False,
        sparse_collate=False,
    ):
        self.split = split
        self.num_points = num_points
        self.preprocessor = PointCloudPreprocessor(
            num_points=num_points,
            coord_minmax=coord_minmax,
            centralize=centralize,
            random_rotation=random_rotation,
            random_permutation=random_permutation,
            normalize=normalize,
            voxelize=voxelize,
            sparse_collate=sparse_collate,
        )
        self.point_cloud_dataset = ModelNet(
            root=root,
            name=name,
            train=self.split == "train",
            transform=SamplePoints(
                num=self.num_points,
                remove_faces=True,
                include_normals=False,
            ),
        )

    def __len__(self):
        return len(self.point_cloud_dataset)


class PointCloudPreprocessor:
    def __init__(
        self,
        num_points=None,
        coord_minmax=None,
        centralize=True,
        random_rotation=False,
        random_permutation=True,
        normalize=True,
        voxelize=False,
        sparse_collate=False,
    ):
        self.num_points = num_points
        self.coord_minmax = coord_minmax
        self.centralize = centralize
        self.random_rotation = random_rotation
        self.random_permutation = random_permutation
        self.normalize = normalize
        self.voxelize = voxelize
        self.sparse_collate = sparse_collate

    def preprocess(self, pc):
        """Perform different types of pre-processings to the ModelNet point clouds."""
        pc = np.asanyarray(pc).astype("float32")

        if self.centralize:
            pc = pc - np.mean(pc, axis=0)

        if self.random_rotation:
            pc = np.dot(pc, random_rotation_3x3_matrix())

        if self.normalize:
            pc = normalize_unit_ball(pc)

        if self.coord_minmax is not None:
            coord_min, coord_max = self.coord_minmax
            pc_min, pc_max = np.min(pc), np.max(pc)
            pc = (pc - pc_min) / (pc_max - pc_min) * (coord_max - coord_min) + coord_min

        if self.random_permutation:
            idx = np.arange(pc.shape[0])
            np.random.shuffle(idx)
            pc = pc[idx, :]

        if self.voxelize:
            # Discretize.
            pc = np.round(pc).astype("int32")

            # Keep only unique points following discretization.
            pc = np.unique(pc, axis=0)

            # Pad with -1 rows until pc.shape[0] == num_points.
            # rows = np.full((self.num_points - pc.shape[0], 3), -1, dtype=pc.dtype)
            # pc = np.vstack((pc, rows))

            # NOTE: No voxelization has yet occurred here.

        # This is to facilitate the sparse tensor construction with Minkowski Engine
        if self.sparse_collate:
            assert self.voxelize

            column = np.zeros((pc.shape[0], 1), dtype=pc.dtype)
            pc = np.hstack((column, pc))

            # NOTE: Should this not be moved earlier? Why -1? Why is the first column not 0?
            # Pad with -1 rows until pc.shape[0] == num_points.
            # rows = np.full((self.num_points - pc.shape[0], 4), -1, dtype=pc.dtype)
            # pc = np.vstack((pc, rows))

            # NOTE: No idea why this line of code exists.
            pc[0][0] = 1

        return pc

    def preprocess_always(self, pc):
        if self.random_permutation:
            idx = np.arange(pc.shape[0])
            np.random.shuffle(idx)
            pc = pc[idx, :]

        return pc


@register_dataset("ModelNetSimple")
class ModelNetSimple(ModelNetBase):
    """A simple ModelNet data loader where point clouds are directly represented as 3D points."""

    def __init__(self, cache_path="", transform=None, **kwargs):
        super().__init__(**kwargs)
        self.cache = None
        if cache_path:
            self._load_cache(cache_path)
        # self.transform = transform  # TODO

    def __getitem__(self, index):
        if self.cache:
            raise NotImplementedError
            item = self.cache[index]
            item = self.preprocessor.preprocess_always(item)
            item = item.astype(np.float32)
            return item
        else:
            item = self.point_cloud_dataset[index]
            points = self.preprocessor.preprocess_always(
                self.preprocessor.preprocess(item.pos.numpy())
            )
            return {
                "index": index,
                "points": torch.tensor(points),
                "labels": item.y.item(),
            }

    def _load_cache(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.cache = pickle.load(f)
            return

        self._generate_cache(path)

    def _generate_cache(self, path):
        raise NotImplementedError

        self.cache = []

        for i in range(len(self.point_cloud_dataset)):
            # labels = self.point_cloud_dataset[i].y.numpy()
            points = self.point_cloud_dataset[i].pos.numpy()
            points = self.preprocessor.preprocess(points)  # .astype(np.uint8)
            self.cache.append(points)
            # self.cache.append((points, labels))

        with open(path, "wb") as f:
            pickle.dump(self.cache, f)


# class ModelNetOctree(ModelNetBase):
#     """ModelNet data loader with uniform sampling and octree partitioning."""
#
#     def __init__(self, data_config, sele_config, **kwargs):
#
#         data_config['voxelize'] = True
#         data_config['sparse_collate'] = False
#         super().__init__(data_config, sele_config)
#
#         self.rw_octree = data_config.get('rw_octree', False)
#         if self.rw_octree:
#             self.rw_partition_scheme = data_config.get('rw_partition_scheme', 'default')
#         self.octree_cache_folder = 'octree_cache'
#
#         # Create an octree formatter to organize octrees into arrays
#         self.octree_organizer = OctreeOrganizer(
#             data_config['octree_cfg'],
#             data_config[sele_config].get('max_num_points', data_config['num_points']),
#             kwargs['syntax'].syntax_gt,
#             self.rw_octree,
#             data_config[sele_config].get('shuffle_blocks', False),
#         )
#
#     def __len__(self):
#         return len(self.point_cloud_dataset)
#
#     def __getitem__(self, index):
#
#         while True:
#             if self.rw_octree:
#                 file_name = os.path.join(dataset_path_default, self.octree_cache_folder, self.rw_partition_scheme, str(index)) + '.pkl'
#             else: file_name = None
#
#             # perform octree partitioning and organize the data
#             pc = self.pc_preprocess(self.point_cloud_dataset[index].pos.numpy())
#             pc_formatted, _, _, _, all_skip = self.octree_organizer.organize_data(pc, file_name=file_name)
#             if all_skip:
#                 index += 1
#                 if index >= len(self.point_cloud_dataset): index = 0
#             else: break
#
#         return pc_formatted


def random_rotation_3x3_matrix():
    rot = np.eye(3, dtype="float32")
    rot[0, 0] *= np.random.randint(0, 2) * 2 - 1
    rot = np.dot(rot, np.linalg.qr(np.random.randn(3, 3))[0])
    return rot


def normalize_unit_ball(points):
    dist = np.sqrt(np.sum(points**2, axis=1))
    return points / dist.max()
