import torch
import numpy as np
from enum import Enum
import trimesh
import sys
sys.path.append("..") 
from tiny_nerf.utils.ray_utils import get_ray_directions, get_rays
import numpy as np

from torch.utils.data import Dataset

class Directions(Enum):
    FRONT = 0,      np.array([255, 0, 0, 255]),   'front'      # phi \in [-front/2, front/2)
    SIDE_LEFT = 1,  np.array([0, 255, 0, 255]),   'side'       # phi \in [front/2, 180-front/2)
    BACK = 2,       np.array([0, 0, 255, 255]),   'back'       # phi \in [180-front/2, 180+front/2)
    SIDE_RIGHT = 3, np.array([255, 255, 0, 255]), 'side'     # phi \in [180+front/2, 360-front/2)
    TOP = 4,        np.array([255, 0, 255, 255]), 'overhead' # theta \in [0, overhead]
    BOTTOM = 5,     np.array([0, 255, 255, 255]), 'bottom'  # theta \in [180-overhead, 180]
    def __int__(self):
        return self.value[0]
    
    def get_color(self):
        return self.value[1]
    
    def __str__(self):
        return self.value[2]

def normalize(x):
    return x / torch.sqrt(torch.sum(x**2, keepdim=True, dim=-1))


# taken from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/provider.py
def get_directions_from_spherical(thetas, phis, overhead_angle, front_angle):
    dirs = np.empty(thetas.shape[0], dtype=Directions)
    # first determine by phis
    #phis = phis % (2 * np.pi)
    dirs[(phis < front_angle / 2) | (phis >= 2 * np.pi - front_angle / 2)] = Directions.FRONT
    dirs[(phis >= front_angle / 2) & (phis < np.pi - front_angle / 2)] = Directions.SIDE_LEFT
    dirs[(phis >= np.pi - front_angle / 2) & (phis < np.pi + front_angle / 2)] = Directions.BACK
    dirs[(phis >= np.pi + front_angle / 2) & (phis < 2 * np.pi - front_angle / 2)] = Directions.SIDE_RIGHT
    # override by thetas
    dirs[thetas <= overhead_angle] = Directions.TOP
    dirs[thetas >= (np.pi - overhead_angle)] = Directions.BOTTOM
    return dirs


def generate_random_poses(size, theta_range = np.array([0, 180]),
                          phi_range = np.array([0, 360]),
                          radius_range = np.array([1, 1.5]),
                          front_angle = 60,
                          overhead_angle = 30):
    # convert theta range and phi range to radians
    # as sin, cos function expect radians
    theta_range = theta_range * (np.pi / 180)
    phi_range = phi_range * (np.pi / 180)
    front_angle = front_angle * (np.pi / 180)
    overhead_angle = overhead_angle * (np.pi / 180)

    # generate spherical coordinates
    thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis =  torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]
    radius = torch.rand(size) * (radius_range[1] - radius_range[0]) + radius_range[0]
    #phis[phis < 0] += 2 * np.pi
    # transform spehrical coordinates to the point on the sphere in cartesian coordinates
    x = radius * torch.sin(thetas) * torch.cos(phis)
    y = radius * torch.sin(thetas) * torch.sin(phis)
    z = radius * torch.cos(thetas)

    camera_position = torch.stack((x, y, z), dim=-1)

    # transform cartesian coordinates to the camera coordinates
    # initial camera UP trajectory
    up = torch.Tensor([0, 1, 0]).repeat(size, 1)

    forward = camera_position
    right = normalize(torch.cross(forward, up, dim=-1))
    up = normalize(torch.cross(forward, right, dim=-1))

    # create translation matrix
    poses = torch.empty((3, 4), dtype=torch.float).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right, up, forward), dim=-1)
    poses[:, :3, 3] = camera_position

    directions = get_directions_from_spherical(thetas, phis, overhead_angle, front_angle)

    return poses, directions

# taken from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/provider.py
def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]
    #print(poses.shape)
    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos.numpy(), a.numpy()],
                         [pos.numpy(), b.numpy()],
                         [pos.numpy(), c.numpy()],
                         [pos.numpy(), d.numpy()],
                         [a.numpy(), b.numpy()],
                         [b.numpy(), c.numpy()],
                         [c.numpy(), d.numpy()],
                         [d.numpy(), a.numpy()]])
        
        #print(segs)
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = np.array([dir.get_color()]).repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show(viewer='gl')

def visualize_rays(origins, directions, size = 0.1):
    axes = trimesh.creation.axis(axis_length=4)
    objects = [axes]

    for origin, direction in zip(origins, directions):
        end = origin + direction
        segs = np.array([origin.numpy(), end.numpy()], dtype='object')
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show(viewer='gl')

def generate_all_rays(size, H, W, focal):
    poses, dirs = generate_random_poses(size)
    directions = get_ray_directions(H, W, focal)
    all_rays = []

    for pose in poses:
        origin, direction = get_rays(directions, pose)
        all_rays += [torch.cat([origin, direction], 1)]
    all_rays = torch.cat(all_rays, 0)

    return all_rays, dirs

class CameraDataset(Dataset):
    def __init__(self, label="default", H=64, W=64, samples=0):
        self.label = label
        self.samples = samples
        self.H = H
        self.W = W
        self.default_fov = 20
        self.focal = self.H / (2 * np.tan(np.deg2rad(self.default_fov) / 2))

        self.allrays, self.directions = generate_all_rays(self.samples, self.H, self.W, self.focal)

        self.allrays = self.allrays.reshape(samples, H * W, 6)
    def __len__(self):
        return self.samples
    def __getitem__(self, idx):
        return {
            'rays': self.allrays[idx],
            'dirs': int(self.directions[idx])
        }
