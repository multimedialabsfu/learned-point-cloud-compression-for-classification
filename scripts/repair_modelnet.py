"""
Repair ModelNet40.

Some of the files in ModelNet40 are broken, e.g.,

.. code-block:: bash

    $ head -n 1 ModelNet40/chair/train/chair_0856.off
    OFF6586 5534 0

After running this script, they will be repaired:

.. code-block:: bash

    $ head -n 2 ModelNet40/chair/train/chair_0856.off
    OFF
    6586 5534 0

To download ModelNet40, run:

    $ wget --no-clobber 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    $ unzip ModelNet40.zip

Then, to repair the dataset, run:

.. code-block:: bash

    $ python repair_modelnet.py --input_dir=ModelNet40

Alternatively, you could probably use the following sed command,
which should probably do the same thing as this script:

.. code-block:: bash

    $ sed -i 's:^OFF\(.\+\)$:OFF\n\1:' ModelNet40/**/*.off

...This script may (or may not) be more robust.
"""

import argparse
import re
from pathlib import Path


def repair_modelnet_file(filepath):
    lines = []

    with open(filepath, "r") as f:
        for line in f:
            # Ensure that a newline follows "OFF".
            m = re.match(r"^([Oo][Ff][Ff])(.+)\n$", line)
            if m is not None:
                print("Missing newline after OFF. Repairing...\n")
                print(line)
                lines.extend([f"{m.group(1)}\n", f"{m.group(2)}\n"])
                print("".join(lines[:2]))
                break
            else:
                return  # No repair needed. Early exit.

        lines.extend(f.readlines())

    with open(filepath, "w") as f:
        f.writelines(lines)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    for input_filepath in sorted(Path(args.input_dir).rglob("*.off")):
        print(f"Processing\n  in:  {input_filepath}\n  out: {input_filepath}")
        repair_modelnet_file(input_filepath)


if __name__ == "__main__":
    main()
