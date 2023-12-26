# -*- coding: utf-8 -*-
import torch
from itertools import product
from torch import Tensor, BoolTensor, IntTensor


def alffmc_torch(
        fcoords: Tensor,
        radius: Tensor,
        lattice: Tensor,
        affine_matrices: Tensor,
        symprec=1e-4):
    """
    Atom linkage finder for molecule crystal using PyTorch.

    Args:
        fcoords (torch.Tensor): Fractional coordinates of atoms in the asymmetric unit.
        radius (torch.Tensor): Radii of atoms in the asymmetric unit.
        lattice (torch.Tensor): Lattice vectors.
        affine_matrices (torch.Tensor): Affine matrices of symmetry operations.
        symprec (float, optional): Threshold for overlap atoms. Defaults to 1e-4.

    Returns:
        tuple: A tuple containing the indices and vectors of the colliding atoms.
    """
    natom = radius.size(0)
    nsym = affine_matrices.size(0)
    assert radius.size(0) == fcoords.size(0)
    assert lattice.size() == (3, 3)
    assert affine_matrices.size() == (nsym, 4, 4)

    # QR decomposition in PyTorch
    q, r = torch.linalg.qr(lattice.T)
    mat = r.T
    matinv = torch.linalg.inv(mat)
    matdiag = torch.diag(mat)

    # unitcell
    fc4 = torch.ones((natom, 4), dtype=fcoords.dtype, device=fcoords.device)
    fc4[:, :3] = fcoords
    unit_fcoords = torch.cat([torch.matmul(fc4, a)[:, :3] for a in affine_matrices])
    unit_fcoords -= torch.floor(unit_fcoords)
    unit_coords = torch.matmul(unit_fcoords, mat)

    # rectangular box
    r_max = torch.max(radius) * 2
    box = torch.tensor(list(product([0, 1], repeat=3)), dtype=fcoords.dtype, device=fcoords.device)
    ibox = box * matdiag
    obox = ibox + r_max * box + r_max * (box - 1)
    obox_frac = torch.matmul(obox, matinv)

    # smallest supercell containing all atoms in a rectangular box
    pmin = torch.floor(torch.min(obox_frac, dim=0).values).to(torch.int)
    pmax = torch.ceil(torch.max(obox_frac, dim=0).values).to(torch.int) + 1
    ncell = torch.prod(pmax - pmin)
    frac_offset = torch.stack(torch.meshgrid([torch.arange(a, b) for a, b in zip(pmin, pmax)]), dim=-1).view(-1, 3)
    cart_offset = torch.matmul(frac_offset, mat)
    super_coords = unit_coords.repeat(ncell, 1)
    super_coords += cart_offset.repeat_interleave(nsym * natom, dim=0)

    mask1 = torch.all((obox[0] <= super_coords), dim=1) & torch.all((super_coords < obox[-1]), dim=1)
    coords1 = super_coords[mask1]
    index1 = torch.arange(natom).repeat(nsym * ncell)[mask1]
    radius1 = radius[index1]

    mask2 = torch.all((ibox[0] <= coords1), dim=1) & torch.all((coords1 < ibox[-1]), dim=1)
    coords2 = coords1[mask2]
    index2 = index1[mask2]
    radius2 = radius1[mask2]

    assert coords2.size(0) == nsym * natom, f'{coords2.size(0)} != {nsym * natom}'

    index, vector = collisionDetect_torch(coords1, coords2, radius1, radius2, symprec=symprec)
    index[:, 0] = index1[index[:, 0]]
    index[:, 1] = index2[index[:, 1]]
    return index, vector


def collisionDetect_torch(coords1: Tensor, coords2: Tensor, radius1: Tensor, radius2: Tensor, symprec=1e-4) -> tuple[Tensor, Tensor]:
    r2 = (radius1.view(-1, 1) + radius2)**2
    d2 = -2 * torch.matmul(coords1, coords2.t()) \
        + torch.sum(coords1**2, dim=1).view(-1, 1) \
        + torch.sum(coords2**2, dim=1)
    mask = (symprec**2 < d2) * (d2 < r2)
    index = torch.nonzero(mask)
    vector = coords2[index[:, 1]] - coords1[index[:, 0]]
    return index, vector
