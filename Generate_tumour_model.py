def model_height(thickness,d,d_max,sf):
    """ This function determines the height of the tumour model for each point in the tumour base
    Inputs: 
    - thickness: maximum thickness of the tumour [mm]
    - d: distance from center of base to the point in the tumour base
    - d_max: distance from center of base to edge of the base, in the same direction as d
    - sf: chosen shape factor (degree of the polynomial, typically 1-10 with 1 being like a pyramid and 10 being like a cube)
    
    Outputs:
    - z: height of the tumour model at the specified point in the tumour base

    """
    x = d/d_max
    z = thickness-abs(thickness*x**sf)
    return z
    
def directed_angle_between_vectors(v1,v2):
    """ This function determines the angle in degrees between vector 1 and vector 2. Gives values between 0 and 360 degrees (clockwise), with x-axis being 0        degrees.
    Inputs: 
    - v1: first vector, shape [2,]
    - v2: second vector, shape [2,]

    Outputs:
    - angle in degrees 
    """
    import numpy as np
    x1,y1 = v1
    x2,y2 = v2
    
    dot = x1*x2 + y1*y2      # Dot product between [x1, y1] and [x2, y2]
    det = x1*y2 - y1*x2      # Determinant
    angle = np.degrees(np.arctan2(det, dot))  

    if angle <0: 
        angle =angle+360 # To make sure angles are between 0 and 360, with xaxis being 0 degrees 
    return angle
    
def generate_tumour_model(tumour,eye, corrected_base, sf): 
    """ This function creates a tumour model based on a delineated tumour
    Inputs:
    - tumour: trimesh object of tumour delineation
    - eye: trimesh object of eye
    - corrected_base: list of points in tumour base with shape [n,3]. Can be generated with def correct_base
    - sf: chosen shape factor (degree of the polynomial, typically 1-10 with 1 being like a pyramid and 10 being like a cube)
    
    Outputs:
    - tumour_model_rot_transl: list of points for generated tumour model
    
    """ 
    from Automatic_measurements import Prom_Centre, LBD, rotation_matrix_from_vectors
    from Prepare_base import redefine_prom
    import numpy as np
    import trimesh
    import alphashape
    from shapely.geometry import Polygon, Point
    import scipy
    import matplotlib.pyplot as plt
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
    prom,  prom_base_orig, prom_top = Prom_Centre(tumour,eye, include_sclera = False)
    lbd, lbd_coor1, lbd_coor2, base = LBD(tumour,eye)
    
    # Step 1: For tumours with center of eye within tumour, automatic prominence determination is less reliable. 
    # Replace base coordinate with middle of base for these tumours
    prom_base, redefined = redefine_prom(tumour,eye, corrected_base, prom_base_orig)

    prom = np.linalg.norm(prom_top-prom_base)
    
    # Step 2: rotate and translate to make eye_center = (0,0,0) and prominence along z-axis
    # Rotate and translate to make base_coor (0,0,0) and prominence = z-axis
    points_base = corrected_base #corrected base, produced by def correct_base
    points_eye = eye.vertices #array of all points in eye
    cog_eye = eye.center_mass

    #Define transformation matrices
    vec_x = [-prom,0,0]
    vec_prom = prom_top-prom_base
    
    rot_mat = rotation_matrix_from_vectors(vec_prom,vec_x) #Aligns vec_prom with vec_x 
    rot_mat2 = rotation_matrix_from_vectors(vec_x,vec_prom) #For late, to rotate back to original position
    transl = np.reshape(-cog_eye, [3,1]) # translate to make eye_center (0,0,0)

    # Perform translation and rotation for eye, base, LBD and prom
    points_base = points_base + np.repeat(np.reshape(transl,[1,3]), len(points_base), axis =0) #Base: translate first
    points_base = np.array([np.dot(rot_mat, points_base[i,:]) for i in range(len(points_base))]) #Base: then rotate with rot_mat

    points_eye = points_eye + np.repeat(np.reshape(transl,[1,3]), len(points_eye), axis =0) #eye: translate first
    points_eye = np.array([np.dot(rot_mat, points_eye[i,:]) for i in range(len(points_eye))]) #eye: then rotate with rot_mat

    lbd_coor1_tform = np.dot(rot_mat,np.reshape(lbd_coor1, [3,1])+transl)
    lbd_coor2_tform = np.dot(rot_mat,np.reshape(lbd_coor2, [3,1])+transl)
    prom_base_tform = np.dot(rot_mat,np.reshape(prom_base, [3,1])+transl)
    prom_top_tform = np.dot(rot_mat,np.reshape(prom_top, [3,1])+transl)
    cog_eye_tform = np.dot(rot_mat, np.reshape(cog_eye,[3,1])+transl)

    # Step 2b: rotate around x-axis to make LBD parallel (not equal to) with Z-axis
    # Determine angle between LBD and z-axis
    vec_lbd = [lbd_coor1_tform[1]-lbd_coor2_tform[1], lbd_coor1_tform[2]-lbd_coor2_tform[2]]
    vec_axis = [0,10]
    theta = directed_angle_between_vectors(np.reshape(vec_lbd, [2,]),vec_axis) 

    # Define rotation matrix
    rot_mat_xaxis = [[1,0,0],[0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))], [0, np.sin(np.radians(theta)), np.cos(np.radians(theta))]]

    # Perform translation and rotation for eye, base, LBD and prom
    points_base2 = np.array([np.dot(rot_mat_xaxis, points_base[i,:]) for i in range(len(points_base))]) #Base: rotate around x-axis
    points_eye2 = np.array([np.dot(rot_mat_xaxis, points_eye[i,:]) for i in range(len(points_eye))]) #Eye: rotate around x-axis

    lbd_coor1_tform2 = np.dot(rot_mat_xaxis,np.reshape(lbd_coor1_tform, [3,1]))
    lbd_coor2_tform2 = np.dot(rot_mat_xaxis,np.reshape(lbd_coor2_tform, [3,1]))
    prom_base_tform2 = np.dot(rot_mat_xaxis,np.reshape(prom_base_tform, [3,1]))
    prom_top_tform2 = np.dot(rot_mat_xaxis,np.reshape(prom_top_tform, [3,1]))
    cog_eye_tform2 = np.dot(rot_mat_xaxis, np.reshape(cog_eye_tform,[3,1]))

    #Step 3: define fundus view
    alphas =[]
    betas = []
    for point in points_base2: 
        # Determine angle alpha
        alpha = np.arctan2(point[2], point[0]) *180/np.pi #in degrees
        alphas.append(alpha)
    
        # Determine angle beta
        beta = np.arctan2(point[1], point[0]) *180/np.pi #in degrees
        betas.append(beta)
    
    # Determine which alpha/beta combination corresponds with edge 
    angles = np.array([[alphas[i], betas[i]] for i in range(len(alphas))])
    points = angles # points in fundus view 
    
    # Compute the concave hull
    shape_alpha = 0.05  # It is possible to adjust alpha for desired "tightness" of the boundary: 0.05 seems to work best
    concave_hull = alphashape.alphashape(points, shape_alpha)

    # Extract boundary points from the concave hull
    if isinstance(concave_hull, Polygon):
        boundary_points = np.array(concave_hull.exterior.coords)
    else:
        print('Try another alpha')
    
    # For every point P_edge in edge, angle gamma can be defined
    ref_vec = [1,0] # Reference vector to determine angle gamma 
    gamma_angles = []
    for i in range(len(boundary_points)):
        angle = directed_angle_between_vectors(ref_vec,boundary_points[i])
        gamma_angles.append(angle) 
        
    # Interpolator to determine new boundary coordinates based on new angle gamma  
    boundary_interp = scipy.interpolate.interp1d(gamma_angles, boundary_points, axis =0)

    # For each point in base P_base: determine gamma. With interpolator, find point on edge P_boundary with this gamma.
    base_gammas=[]
    corr_boundary_points = []
    for i in range(len(points)):
        gamma = directed_angle_between_vectors(ref_vec,points[i]) 
        base_gammas.append(gamma)
        try:
            newpoint_boundary = boundary_interp(gamma)
            corr_boundary_points.append(newpoint_boundary)
        except:
            corr_boundary_points.append([np.nan, np.nan])

    # Uncomment here if you want to print a figure of the point cloud with boundary points    
    #fig = plt.figure(figsize=(8, 6))
    #plt.scatter(points[:, 0], points[:, 1], label='Point Cloud', color='blue', s=10)
    #plt.scatter(boundary_points[:, 0], boundary_points[:, 1], label='Boundary points', c=gamma_angles, cmap ='Reds')
    #plt.title('Point cloud with boundary')
    #plt.xlabel('X-axis')
    #plt.ylabel('Y-axis')
    #plt.legend()
    #plt.grid()
    #plt.show()
    #plt.colorbar(label = 'Angle gamma [deg]')
    
    # Determine distances d and dmax
    ref_point=np.array([0,0])
    error_idx = []
    point_heights = []
    
    for i in range(len(points)):
        d = np.linalg.norm(points[i]-ref_point)
        dmax = np.linalg.norm(corr_boundary_points[i]-ref_point)
    
        if d>dmax: 
            error_idx.append(i)
            point_heights.append(np.nan)
        
        else: 
            point_heights.append(model_height(prom,d,dmax,sf))
    
    # Determine tumour top
    tops = []
    for i in range(len(points_base2)):
        point = points_base2[i] # Points in tumour base 
        vec_top = np.array([-1,0,0]) # Direction is parallel to thickness vector, thickness vector lies on x-axis
        top_coor = point + vec_top*point_heights[i]
        tops.append(top_coor)
    tops = np.array(tops)

    #Rotate entire tumour back
    #First, put all points back in one array
    tumour_model_points = np.concatenate((points_base2,tops))

    # Rotate back to original position
    tumour_model_rotated = [] 

    # Rotation matrix x-as (step 2b) 
    rot_mat_xaxis_back = [[1,0,0],[0, np.cos(np.radians(-theta)), -np.sin(np.radians(-theta))], [0, np.sin(np.radians(-theta)), np.cos(np.radians(-theta))]]
    tumour_model_rotated_2b = np.array([np.dot(rot_mat_xaxis_back, tumour_model_points[i,:]) for i in range(len(tumour_model_points))])

    # Rotation and translation as performed in step 2a
    rot_mat2 = rotation_matrix_from_vectors(vec_x,vec_prom) 
    transl_back = np.reshape(cog_eye, [3,1]) 

    tumour_model_rotated_2a = np.array([np.dot(rot_mat2, tumour_model_rotated_2b[i,:]) for i in range(len(tumour_model_points))])
    tumour_model_translated_2a = tumour_model_rotated_2a + np.repeat(np.reshape(transl_back,[1,3]), len(tumour_model_points), axis =0) 

    tumour_model_rot_transl = tumour_model_translated_2a[~np.isnan(tumour_model_translated_2a).any(axis=1)] # rotated and translated back to original position 

    return tumour_model_rot_transl

def generate_tumour_model_extrathickness(tumour,eye, corrected_base, sf, addedthickness): 
    """ This function creates a tumour model based on a delineated tumour, with added thickness 
    Inputs:
    - tumour: trimesh object of tumour delineation
    - eye: trimesh object of eye
    - corrected_base: list of points in tumour base with shape [n,3]. Can be generated with def correct_base
    - sf: chosen shape factor (degree of the polynomial, typically 1-10 with 1 being like a pyramid and 10 being like a cube)
    - addedthickness: amount of added thickness [mm] 
    
    Outputs:
    - tumour_model_rot_transl: list of points for generated tumour model """
    
    
    from Automatic_measurements import Prom_Centre, LBD, rotation_matrix_from_vectors
    from Prepare_base import redefine_prom
    import numpy as np
    import trimesh
    import alphashape
    from shapely.geometry import Polygon, Point
    import scipy
    import matplotlib.pyplot as plt
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
    prom,  prom_base_orig, prom_top = Prom_Centre(tumour,eye, include_sclera = False)
    lbd, lbd_coor1, lbd_coor2, base = LBD(tumour,eye)
    
    # Step 1: For tumours with center of eye within tumour, automatic prominence determination is less reliable. 
    # Replace base coordinate with middle of base for these tumours
    prom_base, redefined = redefine_prom(tumour,eye, corrected_base, prom_base_orig)

    prom = np.linalg.norm(prom_top-prom_base)
    
    # Step 2: rotate and translate to make eye_center = (0,0,0) and prominence along z-axis
    # Rotate and translate to make base_coor (0,0,0) and prominence = z-axis
    points_base = corrected_base #corrected base, produced by def correct_base
    points_eye = eye.vertices #array of all points in eye
    cog_eye = eye.center_mass

    #Define transformation matrices
    vec_x = [-prom,0,0]
    vec_prom = prom_top-prom_base
    
    rot_mat = rotation_matrix_from_vectors(vec_prom,vec_x) #Aligns vec_prom with vec_x 
    rot_mat2 = rotation_matrix_from_vectors(vec_x,vec_prom) #For late, to rotate back to original position
    transl = np.reshape(-cog_eye, [3,1]) # translate to make eye_center (0,0,0)

    # Perform translation and rotation for eye, base, LBD and prom
    points_base = points_base + np.repeat(np.reshape(transl,[1,3]), len(points_base), axis =0) #Base: translate first
    points_base = np.array([np.dot(rot_mat, points_base[i,:]) for i in range(len(points_base))]) #Base: then rotate with rot_mat

    points_eye = points_eye + np.repeat(np.reshape(transl,[1,3]), len(points_eye), axis =0) #eye: translate first
    points_eye = np.array([np.dot(rot_mat, points_eye[i,:]) for i in range(len(points_eye))]) #eye: then rotate with rot_mat

    lbd_coor1_tform = np.dot(rot_mat,np.reshape(lbd_coor1, [3,1])+transl)
    lbd_coor2_tform = np.dot(rot_mat,np.reshape(lbd_coor2, [3,1])+transl)
    prom_base_tform = np.dot(rot_mat,np.reshape(prom_base, [3,1])+transl)
    prom_top_tform = np.dot(rot_mat,np.reshape(prom_top, [3,1])+transl)
    cog_eye_tform = np.dot(rot_mat, np.reshape(cog_eye,[3,1])+transl)

    # Step 2b: rotate around x-axis to make LBD parallel (not equal to) with Z-axis
    # Determine angle between LBD and z-axis
    vec_lbd = [lbd_coor1_tform[1]-lbd_coor2_tform[1], lbd_coor1_tform[2]-lbd_coor2_tform[2]]
    vec_axis = [0,10]
    theta = directed_angle_between_vectors(np.reshape(vec_lbd, [2,]),vec_axis) 

    # Define rotation matrix
    rot_mat_xaxis = [[1,0,0],[0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))], [0, np.sin(np.radians(theta)), np.cos(np.radians(theta))]]

    # Perform translation and rotation for eye, base, LBD and prom
    points_base2 = np.array([np.dot(rot_mat_xaxis, points_base[i,:]) for i in range(len(points_base))]) #Base: rotate around x-axis
    points_eye2 = np.array([np.dot(rot_mat_xaxis, points_eye[i,:]) for i in range(len(points_eye))]) #Eye: rotate around x-axis

    lbd_coor1_tform2 = np.dot(rot_mat_xaxis,np.reshape(lbd_coor1_tform, [3,1]))
    lbd_coor2_tform2 = np.dot(rot_mat_xaxis,np.reshape(lbd_coor2_tform, [3,1]))
    prom_base_tform2 = np.dot(rot_mat_xaxis,np.reshape(prom_base_tform, [3,1]))
    prom_top_tform2 = np.dot(rot_mat_xaxis,np.reshape(prom_top_tform, [3,1]))
    cog_eye_tform2 = np.dot(rot_mat_xaxis, np.reshape(cog_eye_tform,[3,1]))

    #Step 3: define fundus view
    alphas =[]
    betas = []
    for point in points_base2: 
        # Determine angle alpha
        alpha = np.arctan2(point[2], point[0]) *180/np.pi #in degrees
        alphas.append(alpha)
    
        # Determine angle beta
        beta = np.arctan2(point[1], point[0]) *180/np.pi #in degrees
        betas.append(beta)
    
    # Determine which alpha/beta combination corresponds with edge 
    angles = np.array([[alphas[i], betas[i]] for i in range(len(alphas))])
    points = angles # points in fundus view 
    
    # Compute the concave hull
    shape_alpha = 0.05  # It is possible to adjust alpha for desired "tightness" of the boundary: 0.05 seems to work best
    concave_hull = alphashape.alphashape(points, shape_alpha)

    # Extract boundary points from the concave hull
    if isinstance(concave_hull, Polygon):
        boundary_points = np.array(concave_hull.exterior.coords)
    else:
        print('Try another alpha')
    
    # For every point P_edge in edge, angle gamma can be defined
    ref_vec = [1,0] # Reference vector to determine angle gamma 
    gamma_angles = []
    for i in range(len(boundary_points)):
        angle = directed_angle_between_vectors(ref_vec,boundary_points[i])
        gamma_angles.append(angle) 
        
    # Interpolator to determine new boundary coordinates based on new angle gamma  
    boundary_interp = scipy.interpolate.interp1d(gamma_angles, boundary_points, axis =0)

    # For each point in base P_base: determine gamma. With interpolator, find point on edge P_boundary with this gamma.
    base_gammas=[]
    corr_boundary_points = []
    for i in range(len(points)):
        gamma = directed_angle_between_vectors(ref_vec,points[i]) 
        base_gammas.append(gamma)
        try:
            newpoint_boundary = boundary_interp(gamma)
            corr_boundary_points.append(newpoint_boundary)
        except:
            corr_boundary_points.append([np.nan, np.nan])

    # Uncomment here if you want to print a figure of the point cloud with boundary points    
    #fig = plt.figure(figsize=(8, 6))
    #plt.scatter(points[:, 0], points[:, 1], label='Point Cloud', color='blue', s=10)
    #plt.scatter(boundary_points[:, 0], boundary_points[:, 1], label='Boundary points', c=gamma_angles, cmap ='Reds')
    #plt.title('Point cloud with boundary')
    #plt.xlabel('X-axis')
    #plt.ylabel('Y-axis')
    #plt.legend()
    #plt.grid()
    #plt.show()
    #plt.colorbar(label = 'Angle gamma [deg]')
    
    # Determine distances d and dmax
    ref_point=np.array([0,0])
    error_idx = []
    point_heights = []
    
    for i in range(len(points)):
        d = np.linalg.norm(points[i]-ref_point)
        dmax = np.linalg.norm(corr_boundary_points[i]-ref_point)
    
        if d>dmax: 
            error_idx.append(i)
            point_heights.append(np.nan)
        
        else: 
            point_heights.append(model_height(prom+addedthickness,d,dmax,sf))
    
    # Determine tumour top
    tops = []
    for i in range(len(points_base2)):
        point = points_base2[i] # Points in tumour base 
        vec_top = np.array([-1,0,0]) # Direction is parallel to thickness vector, thickness vector lies on x-axis
        top_coor = point + vec_top*point_heights[i]
        tops.append(top_coor)
    tops = np.array(tops)

    #Rotate entire tumour back
    #First, put all points back in one array
    tumour_model_points = np.concatenate((points_base2,tops))

    # Rotate back to original position
    tumour_model_rotated = [] 

    # Rotation matrix x-as (step 2b) 
    rot_mat_xaxis_back = [[1,0,0],[0, np.cos(np.radians(-theta)), -np.sin(np.radians(-theta))], [0, np.sin(np.radians(-theta)), np.cos(np.radians(-theta))]]
    tumour_model_rotated_2b = np.array([np.dot(rot_mat_xaxis_back, tumour_model_points[i,:]) for i in range(len(tumour_model_points))])

    # Rotation and translation as performed in step 2a
    rot_mat2 = rotation_matrix_from_vectors(vec_x,vec_prom) 
    transl_back = np.reshape(cog_eye, [3,1]) 

    tumour_model_rotated_2a = np.array([np.dot(rot_mat2, tumour_model_rotated_2b[i,:]) for i in range(len(tumour_model_points))])
    tumour_model_translated_2a = tumour_model_rotated_2a + np.repeat(np.reshape(transl_back,[1,3]), len(tumour_model_points), axis =0) 

    tumour_model_rot_transl = tumour_model_translated_2a[~np.isnan(tumour_model_translated_2a).any(axis=1)] # rotated and translated back to original position 

    return tumour_model_rot_transl
    



def upsample_point_cloud(points, factor=10, k=10):
    """
    Upsample a 3D point cloud by interpolating between nearest neighbors.

    Inputs:
    - points: (n, 3) NumPy array of XYZ coordinates.
    - factor: Multiplication factor for upsampling.
    - k: Number of nearest neighbors to consider.

    Returns
    - upsampled_points: (m, 3) NumPy array of upsampled points.
    - new_points: points that were added, (m-n,3) NumPy array
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    num_new_points = int((factor - 1) * len(points))
    
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Generate new points
    new_points = []
    for _ in range(num_new_points):
        base_idx = np.random.randint(0, len(points))  # Select a base point
        neighbor_idx = np.random.choice(indices[base_idx][1:])  # Choose a neighbor (excluding itself)
        
        alpha = np.random.rand(1)  # Random interpolation weight
        new_point = points[base_idx] * alpha + points[neighbor_idx] * (1 - alpha)
        new_points.append(new_point)
    
    new_points = np.vstack(new_points)

    # Combine original and new points
    upsampled_points = np.vstack((points, new_points))

    return upsampled_points, new_points

def save_point_cloud_as_stl(points, filename, alpha = 0.15, max_attempts=5):
    """
    Tries to generate and save an STL file from a point cloud.
    If the generated mesh is not a volume, it upsamples the points.
    
    Inputs:
    - points: list of points that need to be included
    - filename: name of the location + file that the STL needs to be saved at
    - alpha: alpha with which the concave hull is computed
    - max_attemps: how many times the point cloud can be upsampled. If max_attemps is reached, saving the file was unsuccesful.

    Outputs:
    - None
    
    """
    import alphashape
    import pymeshfix
    import trimesh
    from tqdm.auto import tqdm 

    attempts = 0
    
    while attempts < max_attempts:
        # Generate mesh using alphashape
        mesh = alphashape.alphashape(points, alpha)  
        
        meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces) # Convert to pymeshfix format
        meshfix.repair() # Fill holes using pymeshfix
        fixed = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f) #Convert back to trimesh
        
        # Check if the mesh is a valid volume
        if fixed.is_volume:
            fixed.export(filename)
            tqdm.write(f"   Saved STL file: {filename}")
            return True
        
        print(f"   Mesh not a volume, upsampling points (Attempt {attempts + 1})...")
        points, new_points = upsample_point_cloud(points, factor=1.5, k=10)  # Upsample the point cloud
        attempts += 1
    
    tqdm.write("Failed to generate a valid volume mesh after upsampling.")
    return False

