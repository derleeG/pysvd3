# pysvd3
A python wrapper for fast svd of 3x3 matrix

see https://github.com/ericjang/svd3 for details

# Installation
sh make.sh

# Benchmark
## reconstruction error
|        method                |   evaluation        | numpy with openblas | svd3 w/o patching |  svd3 w patching |
| ---------------------------- | --------------------|--------------------:|------------------:|-----------------:|
| QR decomposition             | \|\|QR - A\|\|\_F   | 7.275e-08           | 5.307e-03         | 1.214e-06        |
| singular value decomposition | \|\|USV* - A\|\|\_F | 1.046e-07           | 3.563e-02         | 2.145e-06        |
| polar decomposition          | \|\|UP - A\|\|\_F   | 6.435e-07           | 5.875e-02         | 1.072e-06        |

## execution time
|        method                |   evaluation        | numpy with openblas | svd3 w/o patching |  svd3 w patching |
| ---------------------------- | --------------------|--------------------:|------------------:|-----------------:|
| QR decomposition             | average             | 43.889ms            | 2.948ms           | 2.806ms          |
| singular value decomposition | average             | 21.976ms            | 4.625ms           | 4.686ms          |
| polar decomposition          | average             | 79.979ms            | 3.388ms           | 3.432ms          |
