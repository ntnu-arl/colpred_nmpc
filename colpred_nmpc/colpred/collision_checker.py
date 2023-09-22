import numpy as np
import warp as wp


class ColChecker:
    """Parallelized collision checker in depth images."""
    def __init__(self, depth_max, hfov, vfov, safe_ball_size, margin=20, seed=None, use_cpu=False):
        self.depth_max = depth_max
        self.hfov = hfov
        self.vfov = vfov
        self.safe_ball_size = safe_ball_size
        self.margin = margin
        self.rng = np.random.default_rng(seed)
        wp.init()
        self.device = 'cpu' if use_cpu else wp.get_device()


    def sample_pos_in_safeball(self, nb_points, add_margin=False):
        """Samples nb_points states (x, y, z) into the a ball centered on origin.
        Return both the points tensor and the normalized points tensor (wrt depth_max, hfov and vfov).
        The uniform spherical sampling code snippet is taken from https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
        The add_margin param allows to enlarge the sampling bounds outside the strict limits of the ball.
        """
        safe_ball_size = self.safe_ball_size*(100+self.margin)/100 if add_margin else self.safe_ball_size*0.99

        phi = self.rng.uniform(0, 2*np.pi, size=(nb_points,1))
        costheta = self.rng.uniform(-1, 1, size=(nb_points,1))
        u = self.rng.uniform(0, 1, size=(nb_points,1))
        theta = np.arccos(costheta)
        r = safe_ball_size * u**(1/3)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        points = np.concatenate([x, y, z], axis=1).astype('float32')
        points_normalized = np.concatenate([x/self.depth_max, y/(self.depth_max*np.tan(self.hfov)), z/(self.depth_max*np.tan(self.vfov))], axis=1).astype('float32')

        return points, points_normalized


    def sample_pos_in_frustrum(self, nb_points, add_margin=False):
        """Samples nb_points states (x, y, z) that fall into the camera frustrum.
        The depth sampling follows a beta distrib in order to have less points right in front of the camera.
        The bearing sampling is uniform.
        Return both the points tensor and the normalized points tensor (wrt depth_max, hfov and vfov).
        The add_margin param allows to enlarge the sampling bounds outside the strict limits of the camera frustrum.
        """
        if add_margin:
            drange = (self.depth_max)*(100+self.margin)/100
            dinf = - (drange - self.depth_max)/2
            hlim = self.hfov*(100+self.margin)/100
            vlim = self.vfov*(100+self.margin)/100
        else:
            drange = self.depth_max*0.99
            dinf = 0.01
            hlim = self.hfov*0.99
            vlim = self.vfov*0.99

        x = self.rng.beta(3, 2, size=(nb_points,1)) * drange + dinf  # (2,1) or (4,2) gives good sampling behavior, not really motivated
        # dsup = dinf + drange
        # x = self.rng.uniform(dinf, dsup, size=(nb_points,1))
        bearing_h = self.rng.uniform(-hlim, hlim, size=(nb_points,1))
        bearing_v = self.rng.uniform(-vlim, vlim, size=(nb_points,1))

        y = np.tan(bearing_h) * x
        z = np.tan(bearing_v) * x

        points = np.concatenate([x, y, z], axis=1).astype('float32')
        points_normalized = np.concatenate([x/self.depth_max, y/(self.depth_max*np.tan(self.hfov)), z/(self.depth_max*np.tan(self.vfov))], axis=1).astype('float32')

        return points, points_normalized


    def grid_frustrum_slice(self, nb_points, bearing_v_deg, add_margin=False):
        """Generates a grid of ~nb_points of uniformly spaced points over a camera frustrum "slice", e.g. for a given vertical bearing.
        The grid has floor(nb_points**(1/2)) points in each direction, so if nb_points is not a perfect square, it does not correspond to the output shape.
        Returns the point tensor as well as its normalized version.
        The add_margin param allows to enlarge the sampling bounds outside the strict limits of the camera frustrum.
        """
        if add_margin:
            drange = (self.depth_max)*(100+self.margin)/100
            dinf = - (drange - self.depth_max)/2
            dsup = dinf + drange
            hlim = self.hfov*(100+self.margin)/100
        else:
            dinf = 0.01
            dsup = self.depth_max*0.99
            hlim = self.hfov*0.99
        grid_size = int(nb_points**(1/2))

        x = np.linspace(dinf, dsup, grid_size)
        bearing_h = np.linspace(-hlim, hlim, grid_size)
        bearing_v = bearing_v_deg * np.pi / 180

        points = np.zeros((grid_size**2, 3), dtype='float32')
        points[:,0] = np.repeat(x, grid_size)
        points[:,1] = np.matmul(np.tan(bearing_h.reshape((grid_size, 1))), x.reshape((1,grid_size))).reshape(grid_size*grid_size, order='F')
        points[:,2] = np.matmul(np.tan(bearing_v * np.ones((grid_size, 1))), x.reshape((1,grid_size))).reshape(grid_size*grid_size, order='F')

        points_normalized = np.zeros((grid_size**2, 3), dtype='float32')
        points_normalized[:,0] = points[:,0] / self.depth_max
        points_normalized[:,1] = points[:,1] / (self.depth_max*np.tan(self.hfov))
        points_normalized[:,2] = points[:,2] / (self.depth_max*np.tan(self.vfov))

        return points, points_normalized


    def grid_frustrum(self, nb_points, depth_min, depth_max, hfov, vfov, add_margin=False, margin=20):
        """Generates a grid of ~nb_points of uniformly spaced points over a camera frustrum.
        The grid has floor(nb_points**(1/3)) points in each direction, so if nb_points is not a perfect cube, it does not correspond to the output shape.
        Returns the point tensor as well as its normalized version.
        The add_margin param allows to enlarge the sampling bounds outside the strict limits of the camera frustrum.
        """
        if add_margin:
            drange = (self.depth_max)*(100+self.margin)/100
            dinf = - (drange - self.depth_max)/2
            dsup = dinf + drange
            hlim = self.hfov*(100+self.margin)/100
            vlim = self.vfov*(100+self.margin)/100
        else:
            dinf = 0.01
            dsup = self.depth_max*0.99
            hlim = self.hfov*0.99
            vlim = self.vfov*0.99
        grid_size = int(nb_points**(1/3))

        x = np.linspace(dinf, dsup, grid_size)
        bearing_h = np.linspace(-hlim, hlim, grid_size)
        bearing_v = np.linspace(-vlim, vlim, grid_size)

        y = np.matmul(np.tan(bearing_h.reshape((grid_size, 1))), x.reshape((1,grid_size)))
        z = np.matmul(np.tan(bearing_v.reshape((grid_size, 1))), x.reshape((1,grid_size)))

        points = np.zeros((grid_size**3, 3), dtype='float32')
        points[:,0] = np.repeat(x, grid_size**2)
        points[:,1] = np.repeat(y.reshape(grid_size**2, order='F'), grid_size)
        points[:,2] = np.tile(z, (grid_size,1)).reshape(grid_size**3, order='F')

        points_normalized = np.zeros((grid_size**3, 3), dtype='float32')
        points_normalized[:,0] = points[:,0] / self.depth_max
        points_normalized[:,1] = points[:,1] / (depth_max*np.tan(self.hfov))
        points_normalized[:,2] = points[:,2] / (depth_max*np.tan(self.vfov))
        return points, points_normalized


    @staticmethod
    @wp.kernel
    def wp_col_check(img:wp.array3d(dtype=wp.float32), points:wp.array(dtype=wp.vec3), points_per_img: float, safe_ball: float,  depth_max: float, hfov: float, vfov: float, collisions: wp.array(dtype=wp.float32)):
        """Warp kernel function to parallelly check the collision ground truth for some point given a depth image.
        img             -- batch of N depth images, array of size (nb_batch, H, W).
        points          -- array of M points [x, y, z] to be checked in the images.
        points_per_img  -- number of points to check per image, such that M = N x points_per_img. Used to get the index of image corresponding to a tid.
        safe_ball       -- size of safe ball around the origin, within which the collision label is always 0.
        depth_max       -- max frustrum depth.
        hfov            -- horizon half fov.
        vfov            -- vertical half fov.
        collisions      -- output array of size M collision labels.
        """
        ## thread index
        tid = wp.tid()
        p = points[tid]

        if wp.length(p) > safe_ball:
            bearing_h = wp.atan2(p[1],p[0])
            bearing_v = wp.atan2(p[2],p[0])
            if p[0] <= 0 or p[0] >= depth_max or abs(bearing_h) >= hfov or abs(bearing_v) >= vfov:
                collisions[tid] = 1.
            else:
                ## compute image index
                img_idx = int(wp.floor(float(tid)/points_per_img))

                ## pixel coordinates
                dim_h = float(img.shape[2])/2.
                dim_v = float(img.shape[1])/2.
                pixel_h = int(-bearing_h/hfov * dim_h + dim_h)
                pixel_v = int(-bearing_v/vfov * dim_v + dim_v)

                if p[0] >= img[img_idx, pixel_v, pixel_h]:
                    collisions[tid] = 1.


    def check_image_points(self, depth_imgs, points):
        """Wrapper around the warp function to instantiate arrays and call the kernel.
        Size of depth images array is either (nb_batch, H, W) or (H, W).
        Depth images are assumed normalized wrt depth_max.
        Points are unormalized Cartesian coordinates.
        """
        if len(depth_imgs.shape) == 2:
            depth = wp.array(np.expand_dims(depth_imgs * self.depth_max, 0), dtype=wp.float32, device=self.device)
        elif len(depth_imgs.shape) == 3:
            depth = wp.array(depth_imgs * self.depth_max, dtype=wp.float32, device=self.device)
        else:
            raise AssertionError('input image must have size (nb_batch, H, W) or (H, W)')

        points_per_img = points.shape[0]/depth.shape[0]
        points = wp.array(points, dtype=wp.vec3, device=self.device)
        col_labels = wp.zeros((points.shape[0],), dtype=wp.float32, device=self.device)

        wp.launch(
            kernel=self.wp_col_check,
            dim=points.shape[0],
            inputs=[depth, points, points_per_img, self.safe_ball_size, self.depth_max, self.hfov, self.vfov],
            outputs=[col_labels],
            device=self.device
        )

        return col_labels.numpy().astype('float32')


    def check_image_points_np(self, depth_imgs, points):
        """Numpy version of check_image_points for easier debugging."""
        if len(depth_imgs.shape) == 2:
            img = np.array(np.expand_dims(depth_imgs * self.depth_max, 0))
        elif len(depth_imgs.shape) == 3:
            img = np.array(depth_imgs * self.depth_max)
        else:
            raise AssertionError('input image must have size (nb_batch, H, W) or (H, W)')
        points_per_img = points.shape[0]/img.shape[0]

        ## check all points
        collisions = np.zeros((points.shape[0],))
        for tid, p in enumerate(points):
            if np.linalg.norm(p) > safe_ball:
                bearing_h = np.arctan2(p[1],p[0])
                bearing_v = np.arctan2(p[2],p[0])
                if p[0] <= 0 or p[0] >= self.depth_max or abs(bearing_h) > self.hfov or abs(bearing_v) > self.vfov:
                    collisions[tid] = 1.
                else:
                    ## compute image index
                    img_idx = int(np.floor(float(tid)/points_per_img))

                    ## pixel coordinates
                    dim_h = float(img.shape[2])/2.
                    dim_v = float(img.shape[1])/2.
                    pixel_h = int(-bearing_h/self.hfov * dim_h + dim_h)
                    pixel_v = int(-bearing_v/self.vfov * dim_v + dim_v)

                    if p[0] >= img[img_idx, pixel_v, pixel_h]:
                        collisions[tid] = 1.

        return collisions.astype('float32')
