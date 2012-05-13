#!/bin/sh

# Include common defs
source common.sh

#######################
# SCHRODINGER
#######################

echo $SCHRO_LIB
cd $SCHRO_LIB
rm -rf $LOCAL_OUTDIR
mkdir -p $LOCAL_OUTDIR/armv6 $LOCAL_OUTDIR/armv7 $LOCAL_OUTDIR/i386

:<< COMMENT
make clean 2> /dev/null
make distclean 2> /dev/null
setenv_arm6
bash autogen.sh
./configure --host=arm-apple-darwin6 --enable-shared=no
# Remove testsuite from build as it breaks when compiling static only
sed -i -e 's/testsuite//g' Makefile
make -j4
for i in `find . -name "lib*.a"`; do cp -rvf $i $LOCAL_OUTDIR/armv6; done
#merge_libfiles $LOCAL_OUTDIR/armv6 libschro_all.a
COMMENT

make clean 2> /dev/null
make distclean 2> /dev/null
setenv_arm7
bash autogen.sh
./configure --host=arm-apple-darwin7 --enable-shared=no
# Remove testsuite from build as it breaks when compiling static only
sed -i -e 's/testsuite//g' Makefile
make -j4
for i in `find . -name "lib*.a"`; do cp -rvf $i $LOCAL_OUTDIR/armv7; done
#merge_libfiles $LOCAL_OUTDIR/armv7 libschro_all.a

:<< COMMENT
make clean 2> /dev/null
make distclean 2> /dev/null
setenv_i386
bash autogen.sh
./configure --enable-shared=no
# Remove testsuite from build as it breaks when compiling static only
sed -i -e 's/testsuite//g' Makefile
make -j4
for i in `find . -name "lib*.a"`; do cp -rvf $i $LOCAL_OUTDIR/i386; done
#merge_libfiles $LOCAL_OUTDIR/i386 libschro_all.a
COMMENT

#create_outdir_lipo
for i in `find schroedinger -name "*.h"`;
do mkdir -p "$GLOBAL_OUTDIR/include/${i%/*}" && cp -rvfp $i $GLOBAL_OUTDIR/include/${i}; done
for i in `find $LOCAL_OUTDIR -name "lib*.a"`;
do mkdir -p "$GLOBAL_OUTDIR/lib/${i%/*}" && cp -rvfp $i $GLOBAL_OUTDIR/lib/${i}; done
cp -rvf $GLOBAL_OUTDIR/lib/${LOCAL_OUTDIR}/* $GLOBAL_OUTDIR/lib && rm -rvf $GLOBAL_OUTDIR/lib/${LOCAL_OUTDIR}

cd ..