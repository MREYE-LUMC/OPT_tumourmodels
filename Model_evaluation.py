import numpy as np
import trimesh
from scipy.spatial import KDTree

from Automatic_measurements import calc_Prom_Centre


def volume_analysis(tumour, model):
    """ Determines several volume comparison metrics for two meshes with a Manifold3D backend.
    Inputs:
    - tumour: trimesh object of tumour
    - model: trimesh object of tumour model

    Outputs:
    - results: gives absolute [mm^3] and relative [%] overlap between tumour and model, overestimation of the tumour by the model, underestimation of the            tumour by the model, and intersetion over union (IoU)

    """


    union = trimesh.boolean.union([tumour, model], engine = 'manifold') # ALWAYS USE MANIFOLD3D BECAUSE BLENDER BACKEND GIVES WONKY BOOLEANS
    intersection = trimesh.boolean.intersection([tumour, model], engine = 'manifold') # ALWAYS USE MANIFOLD3D BECAUSE BLENDER BACKEND GIVES WONKY BOOLEANS

    overlap_abs = intersection.volume # in mm3
    overlap_rel = intersection.volume/tumour.volume *100 # in %

    overestimation_abs = model.volume - intersection.volume # in mm3
    overestimation_rel = overestimation_abs/tumour.volume *100# in %

    underestimation_abs = union.volume - model.volume # in mm3
    underestimation_rel = underestimation_abs/tumour.volume *100 # in %

    IoU = intersection.volume/union.volume

    results = {'overlap_abs': overlap_abs, 'overlap_rel': overlap_rel, 'overestimation_abs': overestimation_abs,
                   'overestimation_rel': overestimation_rel, 'underestimation_abs': underestimation_abs,
                   'underestimation_rel': underestimation_rel, 'IoU': IoU}

    return results


def compute_signed_distances(from_points, to_points, target_mesh):
    """ This function is used in signed_surface_dist and computes signed distances between two point clouds using KDTree.
    The sign is dependent on target_mesh: if the point is inside target_mesh, distance will be negative. If not, distance will be positive.
    Inputs:
    - from_points: points from where to determine distance
    - to_points: points to which to determine distance
    - target_mesh: trimesh object with target mesh (in our case: tumour)

    Output:
    - signed_dists: list of signed distances (only in direction from "from_points" to "to_points" and not the other way around)
    """
    tree = KDTree(to_points)
    dists, idx = tree.query(from_points)
    nearest_points = to_points[idx]

    # Check if the target_mesh contains the points
    signed_dists = []
    for point, dist in zip(nearest_points, dists):
        if target_mesh.contains(point.reshape(1, 3)):
            signed_dists.append(-dist)
        else:
            signed_dists.append(dist)
    return signed_dists

def signed_surface_dist(tumour, model, eye):
    """"  Determines signed surface distance metrics for two meshes. Only looks at tumour and model tops, not bases. Except for thickness <2 mm: then entire        tumours. Model OUTSIDE tumour is positive distance, model INSIDE tumour is negative distance
    
    Inputs:
    - tumour: trimesh object of tumour
    - model: trimesh object of tumour model
    - eye: trimesh object of eye

    Outputs:
    - alldists: All distances between tumour and model
    - dist_metrics: Median absolute, minimum, 0.5th, 1st, 2nd, 5th, 25th, 50th, 75th, 95th percentile and max of alldists.
     """

    thickness, thickness_base, thickness_top = calc_Prom_Centre(tumour, eye, include_sclera=False)

    if thickness > 2: # For tumours larger than 2 mm thickness, only use the tumour top. All points inside the shrunk eye mesh are considered tumour top.
        cog = eye.center_mass
        shrink_factor = 0.9
        shrink_matrix = [[shrink_factor, 0, 0,0],[0, shrink_factor,0,0], [0,0,shrink_factor,0], [0,0,0,1]]
        eye_shrunk = eye.copy()
        eye_shrunk = eye_shrunk.apply_transform(shrink_matrix)
        transl = cog - eye_shrunk.center_mass #The scaling shifts center of mass, translating it back to original place again
        eye_shrunk.apply_translation(transl)

        points_tumour = tumour.sample(10000)
        inside_mask = eye_shrunk.contains(points_tumour)
        tumour_top = points_tumour[inside_mask]

        points_model = model.sample(10000)
        inside_mask2 = eye_shrunk.contains(points_model)
        model_top = points_model[inside_mask2]

        tumour_points = tumour_top
        model_points = model_top
    else:
        tumour_points = tumour.vertices
        model_points = model.vertices

    # Compute signed distances in both directions
    dists = compute_signed_distances(tumour_points, model_points, tumour)
    dists2 = compute_signed_distances(model_points, tumour_points, tumour)

    alldists = dists + dists2

    surf_dist_median_abs = np.median(abs(np.array(alldists)))
    surf_dist_min = np.min(np.array(alldists))
    surf_dist_perc_05 = np.percentile(np.array(alldists), 0.5)
    surf_dist_perc_1 = np.percentile(np.array(alldists), 1)
    surf_dist_perc_2 = np.percentile(np.array(alldists), 2)
    surf_dist_perc_5 = np.percentile(np.array(alldists), 5)
    surf_dist_perc_25 = np.percentile(np.array(alldists), 25)
    surf_dist_perc_50 = np.percentile(np.array(alldists), 50)
    surf_dist_perc_75 = np.percentile(np.array(alldists), 75)
    surf_dist_perc_95 = np.percentile(np.array(alldists), 95)
    surf_dist_max = np.max(np.array(alldists))

    dist_metrics  = {'surf_dist_median_abs': surf_dist_median_abs, 'surf_dist_min': surf_dist_min, 'surf_dist_perc_0.5': surf_dist_perc_05,'surf_dist_perc_1': surf_dist_perc_1,'surf_dist_perc_2': surf_dist_perc_2,'surf_dist_perc_5': surf_dist_perc_5, 'surf_dist_perc_25': surf_dist_perc_25,
                'surf_dist_perc_50': surf_dist_perc_50, 'surf_dist_perc_75': surf_dist_perc_75, 'surf_dist_perc_95': surf_dist_perc_95, 'surf_dist_max': surf_dist_max}

    return alldists, dist_metrics
