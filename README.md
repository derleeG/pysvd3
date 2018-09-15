# pysvd3
A python wrapper for fast svd of 3x3 matrix

see https://github.com/ericjang/svd3 for details

# Installation
sh make.sh

# Benchmark 
testing method: average over 100000 randomly initialized matrix
## reconstruction error
|        method                |   evaluation        | numpy with openblas | svd3 w/o patching |  svd3 w patching |
| ---------------------------- | --------------------|--------------------:|------------------:|-----------------:|
| QR decomposition             | \|\|QR - A\|\|\_F /  \|\|A\|\|\_F | 3.716e-08  | 5.608e-03 | 8.253e-07        |
| singular value decomposition | \|\|USV* - A\|\|\_F /  \|\|A\|\|\_F| 5.891e-08 | 1.750e-02 | 1.259e-06        |
| polar decomposition          | \|\|UP - A\|\|\_F  /  \|\|A\|\|\_F | 3.948e-07 | 2.609e-02 | 4.506e-07        |

## execution time
|        method                |   evaluation        | numpy with openblas | svd3 w/o patching |  svd3 w patching |
| ---------------------------- | --------------------|--------------------:|------------------:|-----------------:|
| QR decomposition             | average             | 43.307us            | 2.771us           | 2.780us          |
| singular value decomposition | average             | 21.133us            | 4.374us           | 4.690us          |
| polar decomposition          | average             | 78.317us            | 3.201us           | 3.537us          |
