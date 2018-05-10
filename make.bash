#! /bin/bash
PROGRAMS=$1
if [ -z "$PROGRAMS" ] || [ $PROGRAMS == "all" ]; then
    PROGRAMS="..."
fi

CGO_CFLAGS_ALLOW='(-fno-schedule-insns|-malign-double|-ffast-math)'

ln -sf $(pwd)/pre-commit .git/hooks/pre-commit || echo ""
ln -sf $(pwd)/post-commit .git/hooks/post-commit || echo ""

(cd opencl/kernels_src/cl && ./make.bash)  || exit 1
(cd opencl/kernels_src && ./make.bash)  || exit 1
go install -v github.com/mumax/3cl/cmd/$PROGRAMS || exit 1
#go vet github.com/mumax/3/... || echo ""
#(cd test && mumax3 -vet *.mx3) || exit 1
#(cd doc && mumax3 -vet *.mx3)  || exit 1

