import sys
import numpy as np
import cv2
import imageio
import argparse


def threshold_triangle(image, nbins=256):
    counts, bin_centers = np.histogram(image, bins=range(nbins))
    nbins = len(counts)
    arg_peak_height = np.argmax(counts)
    peak_height = counts[arg_peak_height]
    arg_low_level, arg_high_level = np.where(counts > 0)[0][[0, -1]]
    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        counts = counts[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1
    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = counts[x1 + arg_low_level]
    norm = np.sqrt(peak_height ** 2 + width ** 2)
    peak_height /= norm
    width /= norm
    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level
    if flip:
        arg_level = nbins - arg_level - 1

    return bin_centers[arg_level]


def crop_paths(paths):
    in_paths = paths[0].split(',')
    out_paths = paths[1].split(',')
    new_paths = []
    new_paths.extend(in_paths)
    new_paths.extend(out_paths)
    return new_paths


def main():
    f = open('log_af.txt', 'w')
    f.write('path1 = ' + sys.argv[-2] + '\npath2 = ' + sys.argv[-1] + '\n')

    parser = argparse.ArgumentParser(description='Trashold3d triangle parametrs.')
    parser.add_argument('paths', type=str, nargs='+',
                        help='Input and output file paths')
    parser.add_argument("--nbins", default=256, type=int,
                        help="The number of bins with which to compute the histogram.")
    args = parser.parse_args()
    paths = args.paths
    paths = crop_paths(paths)
    input_path_0 = paths[0]
    output_path_0 = paths[1]

    f.write(input_path_0 + '\n'  + output_path_0 +  '\n')

    ret, images_0_ch = cv2.imreadmulti(input_path_0)
    f.write('readmulti ready')
    img_count = len(images_0_ch)
    ch1_result = []
    for index in range(img_count):
        print(index, end=' ')
        f.write(str(index) + '\n')
        ch1 = images_0_ch[index]
        ch1_result.append(threshold_triangle(ch1))
    np_results_ch1 = np.array(ch1_result, np.uint8)
    imageio.mimwrite(output_path_0, np_results_ch1)


if __name__ == "__main__":
    main()