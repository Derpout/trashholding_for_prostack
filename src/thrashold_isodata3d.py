import sys
import numpy as np
import cv2
import imageio
import argparse


def threshold_isodata(image, nbins=256):
    counts, bin_centers = np.histogram(image, bins=range(nbins))
    if len(bin_centers) == 1:
        return bin_centers[0]

    counts = counts.astype(np.float32)
    csuml = np.cumsum(counts)
    csumh = csuml[-1] - csuml
    intensity_sum = counts * bin_centers
    csum_intensity = np.cumsum(intensity_sum)
    lower = csum_intensity[:-1] / csuml[:-1]
    higher = (csum_intensity[-1] - csum_intensity[:-1]) / csumh[:-1]
    all_mean = (lower + higher) / 2.0
    bin_width = bin_centers[1] - bin_centers[0]
    distances = all_mean - bin_centers[:-1]
    thresholds = bin_centers[:-1][(distances >= 0) & (distances < bin_width)]
    return thresholds[0]


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

    parser = argparse.ArgumentParser(description='Trashold3d isodata parametrs.')
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
        ch1_result.append(threshold_isodata(ch1))
    np_results_ch1 = np.array(ch1_result, np.uint8)
    imageio.mimwrite(output_path_0, np_results_ch1)


if __name__ == "__main__":
    main()