import argparse, os, glob
import numpy as np

from tqdm import tqdm
from PIL import Image
from scipy import ndimage, interpolate, spatial
from sklearn.neighbors import KDTree

def interpolate_with_plane(i_NN, d_NN, i_interp):
    xyz_NN = np.concatenate((i_NN, np.expand_dims(d_NN, axis=2)), axis=2)
    normal_vect = np.cross((xyz_NN[:,1,:] - xyz_NN[:,0,:]), (xyz_NN[:,2,:] - xyz_NN[:,0,:]), axisa=1, axisb=1)
    c_value = np.sum(normal_vect * xyz_NN[:,0,:], axis=1)
    return (c_value - np.sum(normal_vect[:,:-1] * i_interp, axis=1)) / normal_vect[:,2]

def weights_by_dist(i_NN, i_interp):
    dist1 = np.sqrt(np.square(i_NN[:,0,0] - i_interp[:,0]) + np.square(i_NN[:,0,1] - i_interp[:,1]))
    dist2 = np.sqrt(np.square(i_NN[:,1,0] - i_interp[:,0]) + np.square(i_NN[:,1,1] - i_interp[:,1]))
    dist3 = np.sqrt(np.square(i_NN[:,2,0] - i_interp[:,0]) + np.square(i_NN[:,2,1] - i_interp[:,1]))
    total = dist3 + dist1 + dist2
    return np.column_stack((dist1/total, dist2/total, dist3/total))

def barycentric_interpolate(i_NN, d_NN, i_interp):
    # Compute denominator and numerators for calculating coords/weights of two nearest points
    denom = ((i_NN[:,1,1] - i_NN[:,2,1]) * (i_NN[:,0,0] - i_NN[:,2,0]) + (i_NN[:,2,0] - i_NN[:,1,0]) * (i_NN[:,0,1] - i_NN[:,2,1]))
    w1 = (i_NN[:,1,1] - i_NN[:,2,1]) * (i_interp[:,0] - i_NN[:,2,0]) + (i_NN[:,2,0] - i_NN[:,1,0]) * (i_interp[:,1] - i_NN[:,2,1])
    w2 = (i_NN[:,2,1] - i_NN[:,0,1]) * (i_interp[:,0] - i_NN[:,2,0]) + (i_NN[:,0,0] - i_NN[:,2,0]) * (i_interp[:,1] - i_NN[:,2,1])

    # Determine if NN points form triangle or line and computes barymetric coords/weights
    is_triangle = denom != 0.0
    denom[~is_triangle] = 1.0 # temporary set to avoid div by 0
    w1 = w1 / denom
    w2 = w2 / denom
    # 100% weight for nearest point if three points are in line
    w1[~is_triangle] = 1.0
    w2[~is_triangle] = 0.0
    # all weights sum up to 1
    w3 = 1 - w1 - w2

    # Don't interpolate points that are outside of triangle formed by 3 NN points
    barycentric_coords = np.column_stack((w1,w2,w3))
    outside_triangle = np.min(barycentric_coords, axis=1) < 0
    barycentric_coords[outside_triangle] = weights_by_dist(i_NN[outside_triangle], i_interp[outside_triangle])

    # Nearest Neighbor Method
    # barycentric_coords[outside_triangle] = np.array([1.0, 0.0, 0.0])

    # Calculate the new interpolated depth value
    return np.sum(barycentric_coords * d_NN, axis=1)

def depth_complete(img, dilation_iters=10):
    has_depth = img != 0
    to_interp = ndimage.binary_dilation(has_depth, iterations=dilation_iters)
    # points without depth that need to be interpolated
    to_interp = np.logical_xor(to_interp, has_depth)
    indices_with_depth = np.transpose(np.asarray(has_depth.nonzero()))
    indices_to_interp = np.transpose(np.asarray(to_interp.nonzero()))
    depths = img[indices_with_depth[:,0], indices_with_depth[:,1]]
    # print('# of potential undefined pixels to interpolate: {0}'.format(np.sum(to_interp)))

    # KDTree for fast kNN search
    # find 3 NN (nearest points with defined depth) for each point to be interpolated (no defined depth)
    tree = KDTree(indices_with_depth)
    indexes_NN = tree.query(indices_to_interp, k=3, return_distance=False, breadth_first=True)
    # get actual indices of each NN point and corresponding 16bit depth value
    indices_NN = indices_with_depth[indexes_NN] # n x 3 x 2
    depth_NN = img[indices_NN[:,:,0], indices_NN[:,:,1]] # n x 3

    # filter for edge values (large discrepancy in depth)
    non_edge = (np.amax(depth_NN, axis=1) - np.amin(depth_NN, axis=1) < 1000) # mask: n x 1
    # print('# of edge points filtered out: {0}'.format(np.sum(~non_edge)))

    # Interpolate depth values
    indices_to_interp = indices_to_interp[non_edge]
    new_depth = barycentric_interpolate(indices_NN[non_edge], depth_NN[non_edge], indices_to_interp)
    img[indices_to_interp[:,0], indices_to_interp[:,1]] = new_depth
    # print('# of total points interpolated: {0}'.format(new_depth.shape[0]))

    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='depth_16bit/')
    parser.add_argument('--dest', type=str, default='depth_16bit_comp_dist/')
    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    depth_maps = glob.glob(os.path.join(args.src, '*.png'))
    for i in tqdm(range(len(depth_maps))):
        d = np.array(Image.open(depth_maps[i]))
        name = depth_maps[i].split(args.src)[1].lstrip('/')
        d = Image.fromarray(depth_complete(d))
        d.save(os.path.join(args.dest, name))

    # img = Image.open(os.path.join(args.src, '82.png'))
    # img = np.array(img)
    # img = depth_complete(img)
    # img = Image.fromarray(img)
    # img.save('test.png')
