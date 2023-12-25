import numpy as np
from itertools import product


def alffmc_np(
        fcoords: np.ndarray,
        radius: np.ndarray,
        lattice: np.ndarray,
        affine_matrices: np.ndarray,
        symprec=1e-4):
    """
    Atom linkage finder for molecule crystal.

    Args:
        fcoords (np.ndarray): Fractional coordinates of atoms in the asymmetric unit.
        radius (np.ndarray): Radii of atoms in the asymmetric unit.
        lattice (np.ndarray): Lattice vectors.
        affine_matrices (np.ndarray): Affine matrices of symmetry operations.
        symprec (float, optional): Threshold for overlap atoms. Defaults to 1e-4.

    Returns:
        tuple: A tuple containing the indices and vectors of the colliding atoms.
    """
    natom = len(radius)
    nsym = len(affine_matrices)
    assert len(radius) == len(fcoords)
    assert lattice.shape == (3, 3)
    assert affine_matrices.shape == (nsym, 4, 4)

    mat = np.linalg.qr(lattice.T).R.T
    matinv = np.linalg.inv(mat)
    matdiag = np.diag(mat)

    # unitcell
    fc4 = np.ones((natom, 4))
    fc4[:, :3] = fcoords
    unit_fcoords = np.vstack([np.inner(fc4, a)[:, :3] for a in affine_matrices])
    unit_fcoords -= np.floor(unit_fcoords)
    unit_coords = np.dot(unit_fcoords, mat)

    # rectangular box
    r_max = np.max(radius) * 2
    box = np.array(sorted(product([0, 1], repeat=3)))
    ibox = box * matdiag
    obox = ibox + r_max * box + r_max * (box - 1)
    obox_frac = np.dot(obox, matinv)

    # smallest supercell containing all atoms in a rectangular box
    pmin = np.floor(np.min(obox_frac, axis=0)).astype(int)
    pmax = np.ceil(np.max(obox_frac, axis=0)).astype(int) + 1
    ncell = pmax - pmin
    ncell = ncell[0] * ncell[1] * ncell[2]
    frac_offset = np.array(np.meshgrid(*(np.arange(a, b) for a, b in zip(pmin, pmax)))).T.reshape(-1, 3)
    cart_offset = np.dot(frac_offset, mat)
    super_coords = np.tile(unit_coords, (ncell, 1))
    super_coords += np.tile(cart_offset, nsym * natom).reshape((-1, 3))

    mask1 = np.all((obox[0] <= super_coords), axis=1) * np.all((super_coords < obox[-1]), axis=1)
    coords1 = super_coords[mask1]
    index1 = np.tile(np.arange(natom), nsym * ncell)[mask1]
    radius1 = radius[index1]

    mask2 = np.all((ibox[0] <= coords1), axis=1) * np.all((coords1 < ibox[-1]), axis=1)
    coords2 = coords1[mask2]
    index2 = coords1[mask2]
    radius2 = coords1[mask2]

    assert coords2.shape[0] == nsym * natom, f'{coords2.shape[0]} != {nsym * natom}'

    index, vector = collisionDetect(coords1, coords2, radius1, radius2, symprec=symprec)
    index[:, 0] = index1[index[:, 0]]
    index[:, 1] = index2[index[:, 1]]
    return index, vector


def collisionDetect(coords1, coords2, radius1, radius2, symprec=1e-4):
    r2 = (radius1.reshape((-1, 1)) + radius2)**2
    d2 = -2 * np.dot(coords1, coords2.T) \
        + np.sum(coords1**2, axis=1).reshape((-1, 1)) \
        + np.sum(coords2**2, axis=1)
    mask = (symprec**2 < d2) * (d2 < r2)
    index = np.vstack(np.where(mask)).T
    vector = coords2[index[:, 1]] - coords1[index[:, 0]]
    return index, vector

