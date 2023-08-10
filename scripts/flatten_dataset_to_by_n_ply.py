"""
Flatten a modelnet dataset to the by_n_ply structure used in our other scripts.

.. code-block:: bash

    python flatten_dataset_to_by_n_ply.py --input_dir dataset=modelnet40,format=ply,normalize=True,num_points=8192 --output_dir by_n_ply/8192/modelnet40

TODO: Eliminate the need for this script by getting rid of the by_n_ply structure.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

LOADER_MAP = {
    "train": "train",
    "test": "infer",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Flatten dataset to by_n_ply")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indexes = defaultdict(int)

    for input_file in sorted(input_dir.rglob("*.ply")):
        m = re.match(
            r"^(.*)/(?P<label>\w+)/(?P<loader>\w+)/(?P<label_again>\w+)_(?P<src_index>\d+).ply$",
            str(input_file),
        )

        if m is None:
            raise ValueError(f"Could not parse {input_file} path")

        loader = LOADER_MAP[m.group("loader")]
        if loader != "infer":
            continue

        assert m.group("label") == m.group("label_again")
        label = m.group("label")
        index = indexes[label]
        indexes[label] += 1

        # MAGIC: Since we are iterating the labels in sorted order.
        label_idx = len(indexes) - 1

        output_file = output_dir / f"{loader}_{label_idx:02}_{index:03}.ply"
        print(f"{input_file} => {output_file}")
        output_file.symlink_to(input_file.resolve())


if __name__ == "__main__":
    main()
