import argparse
import subprocess
from pathlib import Path

import numpy as np
from pyntcloud import PyntCloud


def ctmconv(input_file, output_file):
    subprocess.run(["ctmconv", input_file, output_file])


def process_points(args, points):
    assert points.ndim == 2 and points.shape[-1] == 3

    if args.normalize:
        points = points - points.mean(axis=0)
        point_norms = np.sqrt((points**2).sum(axis=1))
        points = points / point_norms.max()

    if args.translate != [0, 0, 0]:
        points = points + np.array(args.translate)

    if args.scale != 1.0:
        points = points * args.scale

    if args.quantize:
        points = points.round()

    return points


def process_cloud(args, cloud):
    xyz = ["x", "y", "z"]
    assert cloud.points.columns.tolist() == xyz

    mesh = cloud.mesh
    cloud.points = cloud.points.astype(np.float64, copy=False)
    cloud.mesh = mesh  # Restore mesh.

    if args.num_points_resample > 0:
        cloud = cloud.get_sample(
            "mesh_random", n=args.num_points_resample, as_PyntCloud=True
        )

    cloud.points[xyz] = process_points(args, cloud.points[xyz].values)
    if args.unique:
        _, idx = np.unique(cloud.points[xyz].values, axis=0, return_index=True)
        cloud.points = cloud.points.iloc[idx]
    if "mesh" in args.output_attributes:
        cloud.mesh = mesh  # Restore mesh.

    if args.num_points_subsample > 0:
        assert len(cloud.points) >= args.num_points_subsample
        idx = np.random.choice(
            len(cloud.points), args.num_points_subsample, replace=False
        )
        cloud.points = cloud.points.iloc[idx]

    return cloud


def iterate_paths(args):
    if args.input_file:
        yield Path(args.input_file), Path(args.output_file)
        return

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    for input_filepath in input_root.rglob(f"*.{args.input_extension}"):
        filepath = Path(input_filepath).relative_to(input_root)
        output_filepath = (output_root / filepath).with_suffix(
            f".{args.output_extension}"
        )
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        yield input_filepath, output_filepath


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--input_extension", type=str, default="*")
    parser.add_argument("--output_extension", type=str, default="ply")
    parser.add_argument("--convert_only", type=bool, default=False)
    parser.add_argument(
        "--output_attributes",
        type=str,
        help="Attributes to copy from input to output, e.g. 'mesh'. Comma separated.",
        default="",
    )
    parser.add_argument("--normalize", type=bool, default=False)
    parser.add_argument("--translate", type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--quantize", type=bool, default=False)
    parser.add_argument("--unique", type=bool, default=False)
    parser.add_argument("--num_points_resample", type=int, default=0)
    parser.add_argument("--num_points_subsample", type=int, default=0)
    return parser


def parse_args(parser, argv=None):
    args = parser.parse_args(argv)

    if bool(args.input_file) == bool(args.input_dir):
        raise ValueError("Please specify exactly one of --input_file or --input_dir.")

    if bool(args.output_file) == bool(args.output_dir):
        raise ValueError("Please specify exactly one of --output_file or --output_dir.")

    if (bool(args.input_file) != bool(args.output_file)) or (
        bool(args.input_dir) != bool(args.output_dir)
    ):
        raise ValueError(
            "Please specify only files or only directories for input and output."
        )

    args.output_attributes = (
        args.output_attributes.strip().split(",") if args.output_attributes else []
    )

    if "mesh" in args.output_attributes:
        raise ValueError("mesh attribute is not supported.")
        if args.num_points_resample > 0:
            raise ValueError("--num_points_resample destroys the mesh attribute.")
        if args.num_points_subsample > 0:
            raise ValueError("--num_points_subsample destroys the mesh attribute.")

    return args


def main(argv=None):
    parser = build_parser()
    args = parse_args(parser, argv)

    for input_filepath, output_filepath in sorted(iterate_paths(args)):
        print(f"Processing\n  in:  {input_filepath}\n  out: {output_filepath}")

        if args.convert_only:
            assert (
                args.output_attributes == []
                and args.normalize is False
                and args.translate == [0, 0, 0]
                and args.scale == 1.0
                and args.quantize is False
                and args.unique is False
                and args.num_points_resample == 0
                and args.num_points_subsample == 0
            )
            ctmconv(input_filepath, output_filepath)
            continue

        cloud = PyntCloud.from_file(str(input_filepath))
        cloud = process_cloud(args, cloud)
        cloud.to_file(str(output_filepath), also_save=args.output_attributes)


if __name__ == "__main__":
    main()
