import numpy as np
import cv2
import warp as wp
import trimesh as tm
import copy

@wp.kernel
def draw(mesh: wp.uint64, width: wp.int32, height: wp.int32, focal_pixel: wp.float32, depth_max: wp.float32, pixels: wp.array(dtype=wp.float32)):
    """Warp kernel function to generate depth image from warp meshes.
    img             -- id of mesh.
    width           -- width of depth image.
    height          -- height of depth image.
    focal_pixel     -- pixel focal length.
    depth_max       -- max frustrum depth.
    pixels          -- output depth image.
    """
    tid = wp.tid()

    px = wp.float32(tid%width)
    py = wp.float32(tid//width)

    x = (px-float(width)/2.)/focal_pixel
    y = (py-float(height)/2.)/focal_pixel

    # compute view ray https://nvidia.github.io/warp/_build/html/modules/functions.html#warp.mesh_query_ray
    ro = wp.vec3(0.,0.,0.)
    rd = wp.normalize(wp.vec3(x, y, 1.0))

    t = float(0.0)  # length of ray
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    if wp.mesh_query_ray(mesh, ro, rd, 50., t, u, v, sign, n, f):
        pixels[tid] = t*rd[2]
    else:
        pixels[tid] = depth_max


def depth_to_pointcloud(depth_img, hfov, depth_max):
    """Converts a depth image to a point cloud."""
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    f = width/2/np.tan(hfov)  # pixel focal length

    ## creates a (x,y,z) meshgrid of image plane z=1.
    x = np.arange(0, width, dtype=np.float32)
    y = np.arange(0, height, dtype=np.float32)
    x,y = np.meshgrid(x,y)

    ## get 3D coordinate of pixel and scale by real z
    x = (x - width/2) / f * depth_img * depth_max
    y = (y - height/2) / f * depth_img * depth_max
    z = depth_img * depth_max

    return np.stack([x, y, z], axis=0)



def edge_detection(image, th1=30, th2=50):
    """Wrapper around Canny edge detector.
    The edges are shifted toward the min pixel among 4-neighbors to be inside the closest objects.
    """
    edge_image = cv2.Canny(image, th1, th2)
    edges = np.where(edge_image > 0)
    edges = np.array(list(zip(edges[0], edges[1])))  # zip into more convenient format

    for i in range(edges.shape[0]):
        edge = edges[i]
        ## check min among 4-neighbors (up to dist of 2)
        neighbor_list = [
            (max(edge[0] - 1, 0), edge[1]),  # up
            (min(edge[0] + 1, image.shape[0]-1), edge[1]),  # down
            (edge[0], max(0, edge[1] - 1)),  # left
            (edge[0], min(image.shape[1]-1, edge[1] + 1)),  #right
            (max(edge[0] - 2, 0), edge[1]),  # 2nd to up
            (min(edge[0] + 2, image.shape[0]-1), edge[1]),  # 2nd to down
            (edge[0], max(0, edge[1] - 2)),  # # 2nd to left
            (edge[0], min(image.shape[1]-1, edge[1] + 2)),  # 2d to right
        ]
        min_depth = image[edge[0], edge[1]]
        min_neighbor = (0,0)
        ## check min among neighbor_list
        for neighbor in neighbor_list:
            if 0.1 < image[neighbor[0], neighbor[1]] < min_depth:
                min_depth = image[neighbor[0], neighbor[1]]
                min_neighbor = neighbor
        ## change min edge value to neighbor
        if 0.1 < min_depth < image[edge[0], edge[1]]:
            edges[i] = min_neighbor

    return edges, edge_image


def inflate_edges(depth_img, mesh, drone_size, depth_max, hfov, device='cuda:0'):
    """Inflate the edges of objects in depth image by a given mesh.
    depth_img   -- image to process.
    mesh        -- trimesh used for inflation.
    drone_size  -- size of drone for shifting (m).
    depth_max   -- depth max in frustrum.
    hfov        -- horizontal fov.
    """
    wp.init()

    width = depth_img.shape[1]
    height = depth_img.shape[0]

    ## offset all pixels in image by drone_size
    offset_depth = depth_img - drone_size/depth_max

    ## get edgespoints, faces
    edges, edge_img = edge_detection((depth_img*255.0).astype(np.uint8), 15, 40)

    ## pointcloud
    pc = depth_to_pointcloud(depth_img, hfov, depth_max)

    ## meshes
    mesh_list = []

    # pc_list = pc.reshape((3,-1))[:,::2]
    # num_points = pc_list.shape[1]
    # for i in range(num_points):
    #     mesh_list.append(mesh.copy())
    #     mesh_list[-1].apply_translation(pc_list[:, i])

    if edges.shape[0] > 10:
        pc_edges = pc[:, edges[:,0], edges[:,1]]
        num_points = pc_edges.shape[1]
        for i in range(num_points):
            mesh_list.append(copy.deepcopy(mesh))
            mesh_list[-1].apply_translation(pc_edges[:, i])

        mesh_aggregated = tm.util.concatenate(mesh_list)

        points = wp.array(np.array(mesh_aggregated.vertices), dtype=wp.vec3, device=device)
        faces = wp.array(np.array(mesh_aggregated.faces.flatten()), dtype=wp.int32, device=device)
        wp_mesh = wp.Mesh(points, faces)
        pixels = wp.zeros(height*width, dtype=wp.float32, device=device)
        wp.launch(
            kernel=draw,
            dim=height*width,
            inputs=[wp_mesh.id, width, height, width/2/np.tan(hfov), depth_max, pixels],
            device=device,
        )
        edges_inflated = pixels.numpy().reshape((height, width))/depth_max
        edges_inflated = np.minimum(offset_depth, edges_inflated)
    else:
        edges_inflated = offset_depth

    return edges_inflated
