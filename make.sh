# There is a small bug due to fast rsqrt implementation in the upstream c++ lib,
# need to patch it first
sh patch_rsqrt.sh

python setup.py build_ext --inplace
