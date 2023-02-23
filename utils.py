import cv2
import argparse
import glob
import numpy as np
import torch

from PIL import Image
import open3d as o3d
import config

class Open3dVisualizer():
    def __init__(self, K):
        self.point_cloud = o3d.geometry.PointCloud()
        self.o3d_started = False
        self.K = K

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
    
    def __call__(self, rgb_image, depth_map, max_dist=20):
        self.update(rgb_image, depth_map, max_dist)

    def update(self, rgb_image, depth_map, max_dist=20):
        # Prepare the rgb image
        rgb_image_resize = cv2.resize(rgb_image, (depth_map.shape[1],depth_map.shape[0]))
        rgb_image_resize = cv2.cvtColor(rgb_image_resize, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_image_resize), 
                                                                   o3d.geometry.Image(depth_map),
                                                                   1, depth_trunc=max_dist*1000, 
                                                                   convert_rgb_to_intensity = False)
        temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.K)
        temp_pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        # Add values to vectors
        self.point_cloud.points = temp_pcd.points
        self.point_cloud.colors = temp_pcd.colors

        # Add geometries if it is the first time
        if not self.o3d_started:
            self.vis.add_geometry(self.point_cloud)
            self.o3d_started = True

            # Set camera view
            ctr = self.vis.get_view_control()
            ctr.set_front(np.array([ -0.0053112027751292369, 0.28799919460714768, 0.95761592250270977 ]))
            ctr.set_lookat(np.array([-78.783105080589237, -1856.8182240774879, -10539.634663481682]))
            ctr.set_up(np.array([-0.029561736688513099, 0.95716567219818627, -0.28802774118017438]))
            ctr.set_zoom(0.31999999999999978)

        else:
            self.vis.update_geometry(self.point_cloud)

        self.vis.poll_events()
        self.vis.update_renderer()

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(config.DEVICE)

def get_calibration_parameters(file):
    parameters = []
    with open(file, 'r') as f:
        fin = f.readlines()
        for line in fin:
            #print("Line: {}".format(line))
            if line[:4] == 'K_02':
                parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1))
            elif line[:4] == 'T_02':
                parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1))
            elif line[:9] == 'P_rect_02':
                parameters.append(np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)) 
            elif line[:4] == 'K_03':
                parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1)) 
            elif line[:4] == 'T_03':
                parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1))
            elif line[:9] == 'P_rect_03':
                parameters.append(np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)) 
    return parameters

def find_distances(depth_map, pred_bboxes, img, method="center"):
    """
    Go through each bounding box and take a point in the corresponding depth map. 
    It can be:
    * The Center of the box
    * The average value
    * The minimum value (closest point)
    * The median of the values
    """
    depth_list = []
    h, w, _ = img.shape
    for box in pred_bboxes:
        x1 = int(box[0]*w - box[2]*w*0.5) # center_x - width /2
        y1 = int(box[1]*h-box[3]*h*0.5) # center_y - height /2
        x2 = int(box[0]*w + box[2]*w*0.5) # center_x + width/2
        y2 = int(box[1]*h+box[3]*h*0.5) # center_y + height/2
        #print(np.array([x1, y1, x2, y2]))
        obstacle_depth = depth_map[y1:y2, x1:x2]
        if method=="closest":
            depth_list.append(obstacle_depth.min()) # take the closest point in the box
        elif method=="average":
            depth_list.append(np.mean(obstacle_depth)) # take the average
        elif method=="median":
            depth_list.append(np.median(obstacle_depth)) # take the median
        else:
            depth_list.append(depth_map[int(box[1]*h)][int(box[0]*w)]) # take the center
    return depth_list

def calc_depth_map(disp_left, k_left, t_left, t_right):
    # Get the focal length from the K matrix
    f = k_left[0, 0]
    # Get the distance between the cameras from the t matrices (baseline)
    b = abs(t_left[0] - t_right[0]) #On the setup page, you can see 0.54 as the distance between the two color cameras (http://www.cvlibs.net/datasets/kitti/setup.php)
    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1
    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disp_left.shape, np.single)
    # Calculate the depths 
    depth_map[:] = f * b / disp_left[:]
    return depth_map

def draw_depth(depth_map, img_width, img_height, max_dist=10):
		
	return util_draw_depth(depth_map, (img_width, img_height), max_dist)

def draw_disparity(disparity_map, img_width, img_height):

    disparity_map =  cv2.resize(disparity_map,  (img_width, img_height))
    norm_disparity_map = 255*((disparity_map-np.min(disparity_map))/
                              (np.max(disparity_map)-np.min(disparity_map)))

    #return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, alpha=0.01),cv2.COLORMAP_JET)
    return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)

def add_depth(depth_list, result, pred_bboxes):
    h, w, _ = result.shape
    res = result.copy()
    for i, distance in enumerate(depth_list):
        #cv2.line(res,(int(pred_bboxes[i][0]*w - pred_bboxes[i][2]*w/2),int(pred_bboxes[i][1]*h - pred_bboxes[i][3]*h*0.5)),(int(pred_bboxes[i][0]*w + pred_bboxes[i][2]*w*0.5),int(pred_bboxes[i][1]*h + pred_bboxes[i][3]*h*0.5)),(255,255,255),2)
        #cv2.line(res,(int(pred_bboxes[i][0]*w + pred_bboxes[i][2]*w/2),int(pred_bboxes[i][1]*h - pred_bboxes[i][3]*h*0.5)),(int(pred_bboxes[i][0]*w - pred_bboxes[i][2]*w*0.5),int(pred_bboxes[i][1]*h + pred_bboxes[i][3]*h*0.5)),(255,255,255),2)
        cv2.putText(res, 'z={0:.2f} m'.format(distance), (int(pred_bboxes[i][0]*w - pred_bboxes[i][2]*w*0.5),int(pred_bboxes[i][1]*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)    
    return res

def util_draw_depth(depth_map, img_shape, max_dist):

	norm_depth_map = 255*(1-depth_map/max_dist)
	norm_depth_map[norm_depth_map < 0] = 0
	norm_depth_map[norm_depth_map >= 255] = 0
	norm_depth_map =  cv2.resize(norm_depth_map, img_shape)
	return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)

def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = colors.copy()
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts, out_colors])
    with open(fn, 'ab') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
