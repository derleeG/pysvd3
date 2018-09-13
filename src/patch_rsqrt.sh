if [ ! -f lib/svd3/svd3.h.bak ]; then
    echo "Patching lib/svd3/svd3.h to improve accuracy"
    sed -i.bak ' s/for (int i=0;i<4;i++)/for (int i=0; i<5;i++)/g; /inline float rsqrt(float x) {/a \
        return 1/sqrt(x);' lib/svd3/svd3.h
fi
