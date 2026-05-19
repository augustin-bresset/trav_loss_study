import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from src.datasets import Goose3D, Rellis3D, Outback
import os


def read_point_cloud(filename):
    return np.fromfile(filename, dtype=np.float32).reshape(-1, 4)[:, :3]


def detect_ground_height(pointcloud, labels=None, ground_label=None):
    # print(pointcloud.shape, labels.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
    if labels is not None:
        pcd.points = o3d.utility.Vector3dVector(pointcloud[labels == ground_label, :3])
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.1, ransac_n=3, num_iterations=1000
    )
    [a, b, c, d] = plane_model
    distance = d / np.sqrt(a**2 + b**2 + c**2)
    d_flag = distance > 0
    normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    heights = np.dot(pointcloud[:, :3], normal) + distance
    if not d_flag:
        heights *= -1
    return heights, -distance


def write_label(label_array, file_path):
    with open(file_path, "wb+") as file:
        for label in label_array:
            bin_label = np.float32(label)
            file.write(bin_label)


def points_from_csv(filename):
    lidar_depths = np.genfromtxt(filename, delimiter=",", skip_header=1)

    # Define the vertical and horizontal FOV
    vertical_fov = (-30, 30)  # degrees
    horizontal_fov = (0, 360)  # degrees

    # Dataset shape
    depth_array = lidar_depths  # Example random depth values

    # Generate angle values
    vertical_angles = np.linspace(
        vertical_fov[0], vertical_fov[1], depth_array.shape[1]
    )  # 151 points
    horizontal_angles = np.linspace(
        horizontal_fov[0], horizontal_fov[1], depth_array.shape[0]
    )  # 900 points

    # Convert angles to radians
    vertical_angles = -np.radians(vertical_angles)
    horizontal_angles = np.radians(horizontal_angles)

    # Generate a meshgrid for the angles
    phi, theta = np.meshgrid(
        horizontal_angles, vertical_angles, indexing="ij"
    )  # Azimuth (phi) and Elevation (theta)

    # Compute 3D coordinates
    x = depth_array * np.cos(theta) * np.cos(phi)
    y = depth_array * np.cos(theta) * np.sin(phi)
    z = depth_array * np.sin(theta)

    # Reshape into a (N, 3) point cloud format
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    return points


def load_labels(label_file, ext="bin"):
    if ext == "bin":
        return np.fromfile(label_file, dtype=np.float32)
    elif ext == "csv":
        return np.genfromtxt(label_file, delimiter=",", dtype=str).ravel()


def process_dataset(dataset, bin_dir, ground_label=None, ext="bin"):
    filenames, bin_dir = dataset.get_filenames(), dataset.get_bin_dir()
    label_files = (
        dataset.get_label_files() if ground_label else [*range(len(filenames))]
    )
    for filename, label_file in tqdm(
        zip(filenames, label_files), desc="Processing dataset"
    ):
        if not ext == "csv":
            points = read_point_cloud(filename)
        else:
            points = points_from_csv(filename)
            # print(points.shape)
        if ground_label:
            labels = load_labels(label_file, ext)
            assert len(points) == len(
                labels
            ), f"{len(points)} != {len(labels)}, {filename}, {label_file}"
            heights, _ = detect_ground_height(points, labels, ground_label)
        else:
            heights, _ = detect_ground_height(points)
        height_labels_filename = (
            filename.replace(".bin", ".height_label")
            .replace(bin_dir, "height_labels")
            .replace(".csv", ".height_label")
        )
        os.makedirs(os.path.dirname(height_labels_filename), exist_ok=True)
        heights = heights.astype(np.float32)
        heights.tofile(height_labels_filename)


def main():
    parser = argparse.ArgumentParser(
        description="Process point clouds to find height labels."
    )
    parser.add_argument(
        "--dataset",
        choices=["Goose3D", "Rellis3D", "Outback"],
        help="Name of the dataset",
    )
    parser.add_argument("--rootdir", help="Root directory of the dataset")
    args = parser.parse_args()

    datasets = []
    ground_label = None
    ext = "bin"

    if args.dataset == "Goose3D":
        datasets.append(Goose3D(root_dir=args.rootdir, split="train", max_samples=None))
        datasets.append(Goose3D(root_dir=args.rootdir, split="val", max_samples=None))
    elif args.dataset == "Rellis3D":
        datasets.append(
            Rellis3D(root_dir=args.rootdir, split="train", max_samples=None)
        )
        datasets.append(Rellis3D(root_dir=args.rootdir, split="val", max_samples=None))
    elif args.dataset == "Outback":
        datasets.append(Outback(root_dir=args.rootdir, split="train", max_samples=None))
        datasets.append(Outback(root_dir=args.rootdir, split="val", max_samples=None))
        ground_label = "ground"
        ext = "csv"

    for dataset in datasets:
        process_dataset(dataset, args.rootdir, ground_label, ext)


if __name__ == "__main__":
    main()
