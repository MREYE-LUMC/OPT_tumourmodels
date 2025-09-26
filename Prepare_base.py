def correct_base(tumour,eye, threshold_angle = 45):
    """ This function generates a tumour base, based on a tumour delineation. All points with an angle between normal of  point and normal of corresponding eye          point below the threshold angle are considered tumour base
    Inputs:
    - tumour: trimesh object of tumour
    - eye: trimesh object of eye
    - threshold_angle: angle in degrees between normal of the point and normal of the closest point in the eye in degrees

    Output:
    - list of points in new base
     """

    import numpy as np
    import trimesh

    from Automatic_measurements import LBD

    lbd, lbd_coor1, lbd_coor2, base = LBD(tumour,eye)

    angles = []
    indices = []

    for i in range(len(base.vertices)):
        # For each point in base, find closest point on eye
        point = np.reshape(base.vertices[i], (1,3))
        closest, distance, triangle_id = trimesh.proximity.closest_point(eye, point)

        # Extract normals for base point and corresponding eye point
        base_normal = base.vertex_normals[i]
        eye_normal = eye.face_normals[triangle_id]

        # determine angle between vectors
        vectors = np.ones([1,2,3])
        vectors[0,0,:] = base_normal
        vectors[0,1,:] = eye_normal

        angle = np.rad2deg(trimesh.geometry.vector_angle(vectors)) #trimesh function gives angle in radians
        angles.append(angle[0])

        # if angle between vectors is smaller than 45 degrees, add point to new (corrected) base
        if angle <threshold_angle:
            indices.append(i)

    corrected_base = base.vertices[indices]
    corrected_base_normals = base.vertex_normals[indices]

    # Uncomment this section for a figure of the tumour with tumour base
    #fig = plt.figure()
    #axes = fig.add_subplot(projection='3d')
    #axes.plot(eye.vertices[:,0], eye.vertices[:,1], eye.vertices[:,2], c='w', alpha = 0.8)
    #axes.scatter(tumour.vertices[:,0], tumour.vertices[:,1], tumour.vertices[:,2], c='g', alpha = 0.4, label = 'delineated tumour')
    #axes.scatter(corrected_base[:,0], corrected_base[:,1], corrected_base[:,2], c='b', alpha = 0.4, label = 'corrected tumour base')
    #plt.title('Corrected tumour base with original delinated tumour')
    return corrected_base, corrected_base_normals

def redefine_prom(tumour,eye, corrected_base, prom_base_orig):
    """ For tumours with center of eye within tumour (or very close, <0.4 mm), automatic prominence determination is less reliable. 
    This function replaces the base coordinate with middle of base for these tumours 
    Inputs:
    - tumour: trimesh object of tumour 
    - eye: trimesh object of eye
    - corrected_base: list of points in base
    - prom_base_orig: original base point of thickness vector """

    import numpy as np
    import trimesh

    mmp = eye.center_mass
    apex_coor = trimesh.proximity.closest_point(tumour, np.reshape(eye.center_mass, [1,3]))[0]
    dist_to_top = trimesh.proximity.closest_point(tumour, np.reshape(eye.center_mass, [1,3]))[1]

    if dist_to_top <0.4 or tumour.contains(np.reshape(mmp, [1,3])):
        redefined = True
        mean_base_location = np.array([np.mean(corrected_base[:,0]), np.mean(corrected_base[:,1]), np.mean(corrected_base[:,2])])
        prom_base = trimesh.proximity.closest_point(tumour, np.reshape(mean_base_location, [1,3]))[0]
        prom_base = np.reshape(prom_base, np.shape(prom_base_orig))
    else:
        redefined = False
        prom_base = np.array(prom_base_orig)
    return prom_base, redefined

def fit_sphere_to_points(points):
    """ This function fits a sphere to a 3D point cloud and is needed for the expand_base function. 
    Inputs: 
    - points: numpy array of points in shape [n,3]

    Returns:
    - numpy array with center location of sphere
    - radius of the sphere

    """
    import numpy as np
    from scipy.optimize import leastsq


    # Define the function to minimize
    def residuals(params, x, y, z):
        xc, yc, zc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2) - r

    # Initial guess: center at mean, radius as mean distance to mean
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_m, y_m, z_m = x.mean(), y.mean(), z.mean()
    r0 = np.mean(np.linalg.norm(points - np.array([x_m, y_m, z_m]), axis=1))
    params0 = [x_m, y_m, z_m, r0]

    # Fit
    result, _ = leastsq(residuals, params0, args=(x, y, z))
    xc, yc, zc, r = result
    return np.array([xc, yc, zc]), r

def generate_sphere_points(center, radius, num_points=10000):
    """ This function generates points on a sphere and is needed for the expand_base function. 
    Inputs:
    - center: numpy array with center location of sphere
    - radius: radius of the sphere
    - num_points: number of points to be generated

    Returns:
    - numpy array of points with length num_points

    """
    import numpy as np

    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2*np.pi, num_points)
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    return np.vstack((x, y, z)).T

def filter_close_sphere_points(sphere_points, original_points, max_distance=0.001):
    """ This function filters a numpy array of points on a sphere and lets through points with a very small distance (max_distance) to the original points.
    It's needed for the expand_base function. 
    Inputs: 
    - sphere_points: numpy array of points on the sphere
    - original_points: numpy array of original points to compare the sphere points to
    - max_distance: allowed distance between points

    Returns: 
    - sphere_points which are closer than max_distance to original points

    """
    import numpy as np
    from scipy.spatial import distance_matrix
    dists = distance_matrix(sphere_points, original_points)
    min_dists = np.min(dists, axis=1)
    return sphere_points[min_dists < max_distance]

def expand_base(corrected_base, max_distance = 1.0):
    """ This function expands the tumour base with max_distance. Needs the functions fit_sphere_to_points, generate_sphere_points, and                filter_close_sphere_points. 
    Inputs:
    - corrected_base: points in the tumour base
    - max_distance: distance with which the base needs to be expanded
    """
    import numpy as np
    center, radius = fit_sphere_to_points(corrected_base)
    sphere_points = generate_sphere_points(center, radius, 50000)
    close_points = filter_close_sphere_points(sphere_points, corrected_base, max_distance)
    expanded_base = np.vstack((corrected_base,close_points))

    return expanded_base, close_points
