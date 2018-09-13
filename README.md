# pysvd3
A python wrapper for fast svd of 3x3 matrix

see https://github.com/ericjang/svd3 for details

# Installation
sh make.sh

# Benchmark
|                     | numpy with openblas | svd3 w/o patching |  svd3 w patching |
| --------------------|--------------------:|------------------:|-----------------:|
| \|\|USV* - A\|\|\_F | 1.047e-07           | 0.045             | 2.143e-06        |
| execution time      | 21.177ms            | 6.887ms           | 7.474ms          |
