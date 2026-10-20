
import os
import torch
import numpy as np


class Graph:
    """The Graph to model the skeletons extracted by the Alpha-Pose.
    Args:
        - strategy: (string) must be one of the follow candidates
            - uniform: Uniform Labeling,
            - distance: Distance Partitioning,
            - spatial: Spatial Configuration,
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        - layout: (string) must be one of the follow candidates
            - coco_cut: Is COCO format but cut 4 joints (L-R ears, L-R eyes) out.
        - max_hop: (int) the maximal distance between two connected nodes.
        - dilation: (int) controls the spacing between the kernel points.
    """
    def __init__(self,
                 layout='coco_cut',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop)
        self.get_adjacency(strategy)

    def get_edge(self, layout):
        if layout == 'coco_cut':
            # Modified to use 18 joints instead of 14
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]

            # This neighbor_link defines the skeleton structure with 18 joints
            # Based on standard COCO keypoints:
            # 0=nose, 1=neck, 2=right shoulder, 3=right elbow, 4=right wrist,
            # 5=left shoulder, 6=left elbow, 7=left wrist, 8=right hip,
            # 9=right knee, 10=right ankle, 11=left hip, 12=left knee,
            # 13=left ankle, 14=right eye, 15=left eye, 16=right ear, 17=left ear
            neighbor_link = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6),
                            (8, 1), (9, 8), (10, 9), (11, 1), (12, 11), (13, 12),
                            (14, 0), (15, 0), (16, 14), (17, 15)]

            self.edge = self_link + neighbor_link
            self.center = 1  # Center is the neck/mid-body
        else:
            raise ValueError('This layout is not supported!')

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("This strategy is not supported!")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD