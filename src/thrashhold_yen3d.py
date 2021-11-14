import sys
import numpy as np
import cv2
import imageio
import argparse


def threshold_yen(image, nbins=256):
    """Return threshold value based on Yen's method."""
    counts, bin_centers = np.histogram(image, bins=range(nbins))

    if bin_centers.size == 1:
        return bin_centers[0]

    pmf = counts.astype(np.float32) / counts.sum()
    P1 = np.cumsum(pmf)
    P1_sq = np.cumsum(pmf ** 2)
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) *(P1[:-1] * (1.0 - P1[:-1])) ** 2)
    return bin_centers[crit.argmax()]


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

    parser = argparse.ArgumentParser(description='Trashold3d Yen parametrs.')
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
        ch1_result.append(threshold_yen(ch1))
    np_results_ch1 = np.array(ch1_result, np.uint8)
    imageio.mimwrite(output_path_0, np_results_ch1)



if __name__ == "__main__":
    main()