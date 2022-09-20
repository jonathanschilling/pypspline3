#!/usr/bin/env sh

# find symbol in libraries
# alexander.pletzer@noaa.gov

sym=$1
dir=$2

echo " $0 called with $# arguments $1 $2 "
if [ $# -ne 2 ]; then
    echo "Usage: $0 symbol directory "
    echo "Example:"
    echo "$0 MPI_Init /usr/local/lib"
    exit 1
fi

for lib in `ls $dir/*.{a,so,o}`; do 
    echo "---inspecting $lib "
    nm $lib | grep -i $sym 
done
