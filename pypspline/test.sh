#!/usr/bin/env sh

PYTHON=python2
rm -rf tests.log
for f in `ls tests/*.py`; do
    echo "--- testing $f ---"
    cat >> tests.log <<EOF
--- testing $f ---   
EOF
    $PYTHON $f >> tests.log
done
echo "Diff'ing the results"
diff tests.log tests.ref