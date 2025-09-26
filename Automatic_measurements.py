
def Prom_Centre(tumour, eye_cc, sclera_tumour='None', include_sclera = True):
    """
    Prominence (=thickness) through centre of the eye and apex of tumour (apex defined by closest point to eye centre)
    Inputs:
        tumour: trimesh object of tumour
        eye_cc: trimesh object of eye contour through ciliary body
        sclera_tumour: trimesh object of tumour including sclera
        include_sclera: boolean, set to True if prominence should be measured including sclera
    Outputs: prom_centre, top_coor, base_coor
    """
    import warnings

    import numpy as np
    import trimesh

    mmp = eye_cc.center_mass

    apex_coor = trimesh.proximity.closest_point(tumour, np.reshape(eye_cc.center_mass, [1,3]))[0]

    dist_to_top = trimesh.proximity.closest_point(tumour, np.reshape(eye_cc.center_mass, [1,3]))[1]
    if dist_to_top < 0.4:
        warnings.warn('Warning: distance to top is smaller than 0.4 mm, manually check result')

    # Finding intersection with sclera at tumour base
    origins = np.reshape([mmp],[1,3])

    if not tumour.contains(np.reshape(mmp, [1,3])):
        directions = np.reshape([apex_coor-mmp], [1,3])
    if tumour.contains(np.reshape(mmp, [1,3])):
        directions = np.reshape([mmp-apex_coor], [1,3])
        warnings.warn('Center of mass is inside tumour, manually check result')

    if include_sclera == True:
        intersector_tumour = trimesh.ray.ray_triangle.RayMeshIntersector(sclera_tumour)
    if include_sclera == False:
        intersector_tumour = trimesh.ray.ray_triangle.RayMeshIntersector(tumour)

    intersect_coordinates = intersector_tumour.intersects_id(origins, directions, return_locations=True, multiple_hits=True)

    # Calculating prominence
    top_coor = apex_coor[0]
    prom_centre1 = np.sqrt((intersect_coordinates[2][0][0]-top_coor[0])**2 + (intersect_coordinates[2][0][1]-top_coor[1])**2 + (intersect_coordinates[2][0][2]-top_coor[2])**2)

    # These if statements are needed because the multiple hits from intersects_id are not in a logical order
    if len(intersect_coordinates[2]) > 1:
        prom_centre2 = np.sqrt((intersect_coordinates[2][1][0]-top_coor[0])**2 + (intersect_coordinates[2][1][1]-top_coor[1])**2 + (intersect_coordinates[2][1][2]-top_coor[2])**2)

        if prom_centre2 > prom_centre1:
            prom_centre = prom_centre2
            base_coor = intersect_coordinates[2][1]
        else:
            prom_centre = prom_centre1
            base_coor = intersect_coordinates[2][0]

    else:
        prom_centre = prom_centre1
        base_coor = intersect_coordinates[2][0]

    return prom_centre,  base_coor, top_coor

def LBD(tumour,eye):
    """
    Calculation of largest basal diameter
    Inputs: trimesh object of tumour and eye
    Outputs: LBD, lbd_coor1, lbd_coor2
    """
    import numpy as np
    import trimesh
    from scipy.spatial.distance import pdist, squareform

    # Create base using a shrunk eye contour that does not reach the choroid
    cog = eye.center_mass
    shrink_factor = 0.90
    shrink_matrix = [[shrink_factor, 0, 0,0],[0, shrink_factor,0,0], [0,0,shrink_factor,0], [0,0,0,1]]

    eye_shrunk = eye.apply_transform(shrink_matrix)

    transl = cog - eye_shrunk.center_mass #The scaling shifts center of mass, translating it back to original place again
    eye_shrunk.apply_translation(transl)
    base = trimesh.boolean.difference([tumour,eye_shrunk], engine = 'manifold')

    # find LBD
    lbd_dist = squareform(pdist(base.vertices, 'euclidean'))
    lbd = np.max(lbd_dist)
    idx_lbd = np.unravel_index(lbd_dist.argmax(), lbd_dist.shape)
    lbd_coor1 = base.vertices[idx_lbd[0]]
    lbd_coor2 = base.vertices[idx_lbd[1]]

    return lbd, lbd_coor1, lbd_coor2, base

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    Inputs:
    - vec1: A 3d "source" vector
    - vec2: A 3d "destination" vector
    
    Outputs:
    - rotation_matrix: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    import numpy as np
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

