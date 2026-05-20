import time
import math
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# 1. Setup Connection
client = RemoteAPIClient()
sim = client.require('sim')

# 2. Start Simulation
sim.startSimulation()
print("Simulation Started")

# Helper function for transformation matrix
def transformMat(alpha, beta, gamma, tx, ty, tz):
    rotx = np.array([
        [1, 0, 0],
        [0, math.cos(alpha), -math.sin(alpha)],
        [0, math.sin(alpha),  math.cos(alpha)]
    ])
    roty = np.array([
        [math.cos(beta), 0, math.sin(beta)],
        [0, 1, 0],
        [-math.sin(beta), 0, math.cos(beta)]
    ])
    rotz = np.array([
        [math.cos(gamma), -math.sin(gamma), 0],
        [math.sin(gamma),  math.cos(gamma), 0],
        [0, 0, 1]
    ])
    rot_total = np.matmul(rotx, roty)
    rot_total = np.matmul(rot_total, rotz)
    
    trans_vector = np.array([[tx], [ty], [tz]])
    R_t_3x4 = np.hstack((rot_total, trans_vector))
    homogeneous_row = np.array([[0, 0, 0, 1]])
    transform_matrix_4x4 = np.vstack((R_t_3x4, homogeneous_row))
    return transform_matrix_4x4

# 3. Object Handles
sim.addLog(1, "Initializing Path Tracking and Mapping...")
p3dx = sim.getObject("/PioneerP3DX")
p3dx_rw = sim.getObject("/PioneerP3DX/rightMotor")
p3dx_lw = sim.getObject("/PioneerP3DX/leftMotor")
LH_Handle = sim.getObject("/LH")
perp_Handle = sim.getObject("/Perp")

# Path handles
path_Handle = []
for i in range(0, 47):
    path_Handle.append(sim.getObject(f"/p{i}"))

# Sensor handles (Picking 4 front-facing/angled sensors)
# Note: Pioneer P3DX usually has ultrasonicSensor1 through 16.
sensor_handles = []
for i in [0, 3, 4, 7]: 
    sensor_handles.append(sim.getObject(f"/PioneerP3DX/visible/ultrasonicSensor[{i}]"))

# Robot Parameters
rw = 0.195 / 2
rb = 0.318 / 2
LH_distance = 0.8

# Data Arrays for Mapping
x_odom = []
y_odom = []
map_x = []
map_y = []

try:
    # 4. Main Loop (Run for 90 seconds)
    start_time = time.time()
    elapsed_prev = 0.0
    
    while (time.time() - start_time) < 75:
        elapsed = time.time() - start_time
        dt = elapsed - elapsed_prev
        elapsed_prev = elapsed

        # Get Pose of p3dx
        p3dx_position = sim.getObjectPosition(p3dx, sim.handle_world)
        p3dx_orientation = sim.getObjectOrientation(p3dx, sim.handle_world)
        
        # Save robot trajectory
        x_odom.append(p3dx_position[0])
        y_odom.append(p3dx_position[1])

        # --- SENSOR MAPPING LOGIC ---
        for sensor in sensor_handles:
            res, dist, pt, obj, norm = sim.readProximitySensor(sensor)
            if res > 0: # If obstacle detected
                # Get sensor transformation matrix relative to the world
                sensor_matrix = sim.getObjectMatrix(sensor, sim.handle_world)
                # Transform detected point (relative to sensor) to absolute world coordinates
                pt_world = sim.multiplyVector(sensor_matrix, pt)
                map_x.append(pt_world[0])
                map_y.append(pt_world[1])

        # --- PATH TRACKING LOGIC ---
        # Calculate LH position wrt the world
        T_mat = transformMat(0, 0, p3dx_orientation[2], p3dx_position[0], p3dx_position[1], p3dx_position[2])
        LH_position_to_world = T_mat @ np.array([[LH_distance], [0], [0], [1]])
        LH_position_to_world = LH_position_to_world[:3, :]

        # Get path points positions
        path_points = []
        for i in range(len(path_Handle)):
            path_points.append(sim.getObjectPosition(path_Handle[i], sim.handle_world))
        
        # Create list of A->B vectors (Path segments)
        vec_AB = []
        for i in range(len(path_points)-1):
            A = np.array(path_points[i]).reshape(3,1)
            B = np.array(path_points[i+1]).reshape(3,1)
            vec_AB.append(B - A)
        
        # Close the loop (Connect last point to first point)
        A_last = np.array(path_points[-1]).reshape(3,1)
        B_first = np.array(path_points[0]).reshape(3,1)
        vec_AB.append(B_first - A_last) 
        
        # Create list of A->LH vectors
        vec_ALH = []
        for i in range(len(path_points)):
            A = np.array(path_points[i]).reshape(3,1)
            vec_ALH.append(LH_position_to_world - A)
            
        # Project ALH on AB to find scalar projection point
        scalar_proj_points = []
        for i in range(len(vec_AB)):
            ALH = vec_ALH[i]
            AB = vec_AB[i]
            
            # Dot product (ALH • AB) / (AB • AB)
            ab_squared = float(np.dot(AB.T, AB))
            if ab_squared == 0:
                scalar_proj = 0.0
            else:
                scalar_proj = float(np.dot(ALH.T, AB) / ab_squared)
            
            # Clamp projection between 0 and 1 (keep it on the segment)
            if scalar_proj < 0:
                scalar_proj = 0.0
            elif scalar_proj > 1:
                scalar_proj = 1.0
                
            A = np.array(path_points[i]).reshape(3,1)
            scalar_proj_point = A + (scalar_proj * AB)
            scalar_proj_points.append(scalar_proj_point)
                    
        # Find closest scalar projection point to LH
        closest_index = 0
        min_distance = np.linalg.norm(scalar_proj_points[0] - LH_position_to_world)
        for i in range(1, len(scalar_proj_points)):
            distance = np.linalg.norm(scalar_proj_points[i] - LH_position_to_world)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        desired_position = scalar_proj_points[closest_index]

        # Transformation matrix robot w.r.t world 
        T_world_robot = transformMat(0, 0, p3dx_orientation[2], p3dx_position[0], p3dx_position[1], p3dx_position[2])
        
        # Desired position wrt robot
        desired_position_wrt_robot = np.linalg.inv(T_world_robot) @ np.append(desired_position, np.array([[1]]), axis=0)
        desired_position_wrt_robot = desired_position_wrt_robot[:3, :]

        # Error calculation
        ed = math.sqrt(desired_position_wrt_robot[0]**2 + desired_position_wrt_robot[1]**2)
        eh = math.atan2(desired_position_wrt_robot[1], desired_position_wrt_robot[0])

        # Calc body speed (Tune these gains if the robot oscillates)
        vx = 0.5 * ed
        wx = 1.2 * eh

        # Calc wheel speeds
        wr_vel = (vx + (rb*wx)/2)/rw   
        wl_vel = (vx - (rb*wx)/2)/rw

        # Actuate wheel speeds
        sim.setJointTargetVelocity(p3dx_rw, wr_vel)
        sim.setJointTargetVelocity(p3dx_lw, wl_vel)

        # Update visuals for LH and Perp dummies in simulation
        sim.setObjectPosition(LH_Handle, sim.handle_world, LH_position_to_world.flatten().tolist())
        sim.setObjectPosition(perp_Handle, sim.handle_world, desired_position.flatten().tolist())

finally:
    # 5. Stop Simulation safely
    sim.setJointTargetVelocity(p3dx_rw, 0)
    sim.setJointTargetVelocity(p3dx_lw, 0)
    sim.stopSimulation()
    print("\nSimulation Stopped")

    # 6. Plot the Extracted Map and Trajectory
    print("Generating Map...")
    plt.figure(figsize=(10, 8))
    plt.scatter(map_x, map_y, c='black', s=5, label='Mapped Obstacles (Sensors)')
    plt.plot(x_odom, y_odom, c='red', linewidth=2, label='Robot Trajectory')
    plt.title('Pioneer P3DX Exploration Map')
    plt.xlabel('X Coordinate (World)')
    plt.ylabel('Y Coordinate (World)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()