#!/bin/bash

###############################################################################

# Converts OFF format into PLY and puts inside ply/ directory.

# Prerequisites:
# Current directory contains modelnet10/ and modelnet40/.

###############################################################################

fd -e off . modelnet10 | while read -r f; do
  echo "$f"
  mkdir -p "ply/${f%/*}"
  ctmconv "$f" "ply/${f%*.off}.ply"
done

###############################################################################

cp -r modelnet40 modelnet40_repaired

fd -e off . modelnet40_repaired | while read -r f; do
  echo "$f"
  head -n1 "$f"
  sed -i 's:^OFF\(.\+\)$:OFF\n\1:' "$f"  # Ensure that a newline follows "OFF".
  head -n1 "$f"
done

fd -e off . modelnet40_repaired | while read -r f; do
  echo "$f"
  mkdir -p "ply/${f%/*}"
  ctmconv "$f" "ply/${f%*.off}.ply"
done

mv ply/modelnet40_repaired ply/modelnet40

###############################################################################
