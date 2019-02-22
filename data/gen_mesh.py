import numpy as np
import argparse, os, glob

from PIL import Image

# def create_faces(img):
#     i_memo = {}
#     for i in range(len(x.shape)):
#         i_memo[(x[i], y[i])] = i + 1

def get_coords_3d_and_rgb(img, focal_len, rgb):
    has_depth = img != 0
    indices = np.asarray(has_depth.nonzero())
    x, y = indices[1], indices[0]
    z = img[y, x]
    rgb = rgb[y, x, :]
    x, y, z = x.astype(np.float), y.astype(np.float), z.astype(np.float)

    center_x, center_y = (img.shape[1] - 1.0) / 2.0, (img.shape[0] - 1.0) / 2.0
    x = ((x - center_x) * z) / focal_len
    y = ((center_y - y) * z) / focal_len # y coords are flipped

    return np.column_stack((x,y,z)), rgb

def generate_ply(xyz, rgb, file_name):
    if not file_name.endswith('.ply'):
        file_name += '.ply'

    header = ["ply", "format ascii 1.0", "element vertex {0}".format(xyz.shape[0]), "property float x", "property float y", 
    "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header"]

    with open(file_name, 'w+') as ply:
        for item in header:
            ply.write("{0}\n".format(item))

        for i in range(xyz.shape[0]):
            ply.write("{0} {1} {2} {3} {4} {5}\n".format(xyz[i,0], xyz[i,1], xyz[i,2], rgb[i,0], rgb[i,1], rgb[i,2]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='test/')
    parser.add_argument('--dest', type=str, default='meshlab/')
    parser.add_argument('--focal', type=float, default=721.5377)
    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    img = np.array(Image.open(os.path.join(args.src, '82.png')))
    rgb = np.array(Image.open(os.path.join(args.src, '82_rgb.png')).convert('RGB').resize((1244, 376)))
    xyz, rgb = get_coords_3d_and_rgb(img, args.focal, rgb)
    generate_ply(xyz, rgb, os.path.join(args.dest, '82.ply'))
