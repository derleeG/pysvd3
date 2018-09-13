if [ ! -f lib/svd3/svd3.h.bak ]; then
    echo "Patching lib/svd3/svd3.h to improve accuracy"
    sed -i.bak '/inline float rsqrt(float x) {/a \
        return 1/sqrt(x);' lib/svd3/svd3.h
fi
