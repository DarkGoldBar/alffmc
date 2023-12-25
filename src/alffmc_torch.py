# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import Tensor, BoolTensor, IntTensor


def to_lower_triangle(matrix: Tensor) -> Tensor:
    Q, R = torch.linalg.qr(matrix.T)
    return R.T * torch.sign(torch.diag(R.T))


def periodic_limit(fcoords: Tensor, matrix: Tensor, r_intra: float) -> tuple[Tensor, IntTensor]:
    matinv = torch.linalg.inv(matrix)
    unitcube = torch.Tensor([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]).to(matrix.device)
    coords = fcoords.inner(matrix)
    c_max = coords.max(axis=0).values
    c_min = coords.min(axis=0).values
    c_span = c_max - c_min
    cube = unitcube.repeat(2, 1, 1)
    cube[0] = unitcube * c_span + c_min
    cube[1] = cube[0] + r_intra * unitcube + r_intra * (unitcube - 1)
    cube_outter = cube[1].inner(matinv)
    period_range = torch.vstack([
        (cube_outter.min(axis=0).values).floor().int(),
        (cube_outter.max(axis=0).values).ceil().int()
    ])
    return cube, period_range


def expand_symmerty(fcoords: Tensor, affine_matrix: Tensor) -> Tensor:
    S = len(affine_matrix)
    fc4 = torch.ones(len(fcoords), 4,
                    dtype=fcoords.dtype,
                    device=fcoords.device)
    fc4[:, :3] = fcoords
    fc = fc4.repeat(S, 1, 1)
    for i in range(S):
        torch.inner(fc[i], affine_matrix[i], out=fc[i])
    fc = fc[:, :, :3]
    fc -= fc.floor()
    return fc


def expand_period(fcoords: Tensor, period_range: IntTensor) -> Tensor:
    cells = torch.cartesian_prod(
        torch.arange(period_range[0, 0], period_range[1, 0]),
        torch.arange(period_range[0, 1], period_range[1, 1]),
        torch.arange(period_range[0, 2], period_range[1, 2]),
    ).to(fcoords.device)
    return fcoords.repeat(len(cells), 1, 1) + cells.reshape(len(cells), 1, 3)


def ddosap(
        fcoords: Tensor,
        matrix: Tensor,
        affine_matrix: Tensor,
        symprec=1e-4,
        r_inner=2.,
        r_intra=3.,
        as_vector_jj=False,
    ) -> tuple[IntTensor, IntTensor, Tensor, IntTensor]:
    """distance detector of symmetric and periodic system
    对称和周期系统距离探测器

    Args:
        fcoords: (N, 3)  非对称单元的原子的分数坐标
        matrix:  (3, 3)  周期性边界框
        affine_matrix: (S, 4, 4) 对称操作的仿射变换矩阵
        symmprec: (float) 重叠原子判断阈值
        r_inner: (float) 非对称单元内连接距离
        r_intra: (float) 非对称单元外连接距离
        as_vector_jj: 使输出 vector_ij 改变为 j->j'

    Return:
        index_i: (E)
        index_j: (E)
        vector_ij: (E, 3) 向量 i -> j'
        conn_type: (E)
    """
    N = len(fcoords)
    S = len(affine_matrix)
    matrix = to_lower_triangle(matrix)
    cube, period_range = periodic_limit(fcoords, matrix, r_intra)
    fc = expand_symmerty(fcoords, affine_matrix).reshape(-1, 3)
    fc = expand_period(fc, period_range).reshape(-1, 3)

    index1 = origin_symmop_index(affine_matrix)
    mask1 = torch.arange(N*index1, N*(index1+1))
    index2 = origin_period_index(period_range)
    mask2 = torch.arange(N*S*index2, N*S*(index2+1))
    mask = mask2[mask1]
    c = fc.inner(matrix)

    # coords3: atoms out cube, in cube+r_max
    mask3 = np.all((cube_out[0] <= fc), axis=1) * np.all((coords3 < cube_out[-1]), axis=1)
    coords3 = coords3[mask3]
    index3 = np.tile(np.arange(N), S * C)[mask3]
    radius3 = r_inner

    # coords2: atoms in cube
    mask2 = np.all((cube_in[0] <= coords3), axis=1) * np.all((coords3 < cube_in[-1]), axis=1)
    coords2 = coords3[mask2]
    index2 = index3[mask2]
    radius2 = r_intra
    assert coords2.shape[0] == S * N

    # collision test
    idx, vec = detectDistance(coords2, coords3, radius2, radius3, symprec=symprec)
    idx[:, 0] = index2[idx[:, 0]]
    idx[:, 1] = index3[idx[:, 1]]
    return idx, vec

