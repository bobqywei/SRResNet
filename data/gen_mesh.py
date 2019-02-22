import numpy as np
import argparse, os, glob

from PIL import Image

# def create_faces(img):
#     i_memo = {}
#     for i in range(len(x.shape)):
#         i_memo[(x[i], y[i])] = i + 1

def get_coords_3d(img, focal_len):
    has_depth = img != 0
    indices = np.asarray(has_depth.nonzero())
    x, y = indices[1], indices[0]
    z = img[y, x]
    x, y, z = x.astype(np.float), y.astype(np.float), z.astype(np.float)

    center_x, center_y = (img.shape[1] - 1.0) / 2.0, (img.shape[0] - 1.0) / 2.0
    x = ((x - center_x) * z) / focal_len
    y = ((center_y - y) * z) / focal_len # y coords are flipped

    return x, y, z

def generate_xyz(x, y, z, file_name):
    if not file_name.endswith('.xyz'):
        file_name += '.xyz'
    with open(file_name, 'w+') as xyz:
        for i in range(x.shape[0]):
            xyz.write("{0} {1} {2}\n".format(x[i], y[i], z[i]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='depth_16bit_comp/')
    parser.add_argument('--dest', type=str, default='meshlab/')
    parser.add_argument('--focal', type=float, default=721.5377)
    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    img = np.array(Image.open(os.path.join(args.src, '82.png')))
    x, y, z = get_coords_3d(img, args.focal)
    generate_xyz(x,y,z, os.path.join(args.dest, '82.xyz'))
