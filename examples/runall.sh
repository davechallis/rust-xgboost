#!/bin/bash

set -e

examples=(basic custom_objective generalised_linear_model)

for example in "${examples[@]}"
do
    echo "---------- Running example: $example ---------"
    (cd $example && cargo run)
    echo
done
