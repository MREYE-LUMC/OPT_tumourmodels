import warnings

import numpy as np
import trimesh
from scipy.spatial.distance import pdist, squareform


def calc_Prom_Centre(
    tumour: trimesh.Trimesh,
    eye_cc: trimesh.Trimesh,
    sclera_tumour: trimesh.Trimesh | None = None,
    include_sclera: bool = True,
    return_all_prominences: bool = False,
) -> (
    tuple[float, np.ndarray, np.ndarray]
    | tuple[float, np.ndarray, np.ndarray, list[tuple[float, np.ndarray]]]
):
    """Calculate the prominence (=thickness) through centre of the eye and apex of tumour (apex defined by closest point to eye centre)

    Parameters
    ----------
    tumour : trimesh.Trimesh
        Trimesh object of tumour
    eye_cc : trimesh.Trimesh
        Trimesh object of eye contour through ciliary body
    sclera_tumour : trimesh.Trimesh | None, optional
        Trimesh object of tumour including sclera, by default None
    include_sclera : bool, optional
        Boolean, set to True if prominence should be measured including sclera, by default True
    return_all_prominences : bool, optional
        Boolean, set to True if all prominences and their corresponding coordinates should be returned, by default False

    Returns
    -------
    float
        Prominence
    np.ndarray
        Base coordinate
    np.ndarray
        Top coordinate
    list[tuple[float, np.ndarray]]
        List of tuples containing prominences and their corresponding coordinates. Only returned if return_all_prominences is set to True.
    """

    mmp = eye_cc.center_mass

    apex_coor, (dist_to_top, *_), *_ = trimesh.proximity.closest_point(
        tumour, np.reshape(eye_cc.center_mass, (1, 3))
    )

    if dist_to_top < 0.4:
        warnings.warn(
            "Warning: distance to top is smaller than 0.4 mm, manually check result"
        )

    # Finding intersection with sclera at tumour base
    origins = np.reshape(mmp, (1, 3))

    if not tumour.contains(origins):
        directions = np.reshape(apex_coor - mmp, (1, 3))
    elif tumour.contains(origins):
        directions = np.reshape(mmp - apex_coor, (1, 3))
        warnings.warn("Center of mass is inside tumour, manually check result")

    intersector_tumour = (
        trimesh.ray.ray_triangle.RayMeshIntersector(sclera_tumour)
        if include_sclera
        else trimesh.ray.ray_triangle.RayMeshIntersector(tumour)
    )

    *_, intersect_coordinates = intersector_tumour.intersects_id(
        origins, directions, return_locations=True, multiple_hits=True
    )

    # Calculating prominence
    top_coor = apex_coor[0]
    prominences: list[tuple[float, np.ndarray]] = []
    max_prominence = 0.0
    base_coor = None

    for coord in intersect_coordinates:
        if np.array_equal(coord, top_coor):
            continue

        prominence = float(np.linalg.norm(coord - top_coor))
        prominences.append((prominence, coord))

        if prominence > max_prominence:
            max_prominence = prominence
            base_coor = coord

    if base_coor is None:
        raise ValueError(
            "No valid base intersection found. Please check the input meshes."
        )

    if return_all_prominences:
        return max_prominence, base_coor, top_coor, prominences

    return max_prominence, base_coor, top_coor


def calc_LBD(tumour, eye):
    """
    Calculation of largest basal diameter
    Inputs: trimesh object of tumour and eye
    Outputs: LBD, lbd_coor1, lbd_coor2
    """

    # Create base using a shrunk eye contour that does not reach the choroid
    cog = eye.center_mass
    shrink_factor = 0.90
    shrink_matrix = [
        [shrink_factor, 0, 0, 0],
        [0, shrink_factor, 0, 0],
        [0, 0, shrink_factor, 0],
        [0, 0, 0, 1],
    ]

    eye_shrunk = eye.apply_transform(shrink_matrix)

    transl = (
        cog - eye_shrunk.center_mass
    )  # The scaling shifts center of mass, translating it back to original place again
    eye_shrunk.apply_translation(transl)
    base = trimesh.boolean.difference([tumour, eye_shrunk], engine="manifold")

    # find LBD
    lbd_dist = squareform(pdist(base.vertices, "euclidean"))
    lbd = np.max(lbd_dist)
    idx_lbd = np.unravel_index(lbd_dist.argmax(), lbd_dist.shape)
    lbd_coor1 = base.vertices[idx_lbd[0]]
    lbd_coor2 = base.vertices[idx_lbd[1]]

    return lbd, lbd_coor1, lbd_coor2, base


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    Inputs:
    - vec1: A 3d "source" vector
    - vec2: A 3d "destination" vector

    Outputs:
    - rotation_matrix: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (
        (vec1 / np.linalg.norm(vec1)).reshape(3),
        (vec2 / np.linalg.norm(vec2)).reshape(3),
    )
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix
