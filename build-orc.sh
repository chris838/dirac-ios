#!/bin/sh

# Include common defs
source common.sh

#######################
# ORC
#######################

cd $ORC_LIB
rm -rf $LOCAL_OUTDIR
mkdir -p $LOCAL_OUTDIR/armv6 $LOCAL_OUTDIR/armv7 $LOCAL_OUTDIR/i386

#:<< COMMENT
make clean 2> /dev/null
make distclean 2> /dev/null
setenv_arm6
bash autogen.sh
./configure --host=arm-apple-darwin6 --enable-shared=no
make -j4
#find . -name "lib*.a"
cp -rvf orc/.libs/lib*.a $LOCAL_OUTDIR/armv6
cp -rvf orc-test/.libs/lib*.a $LOCAL_OUTDIR/armv6
cp -rvf tools/orcc $LOCAL_OUTDIR/armv6
echo "Creating pkg-config file"
cp -f orc-0.4.pc $PKG_CONFIG_PATH/orc-0.4.pc
sed -i -e "s@^prefix=.*@prefix=${GLOBAL_OUTDIR}@" $PKG_CONFIG_PATH/orc-0.4.pc
sed -i -e "s@^libdir=.*@libdir=\$\{exec_prefix\}/lib/armv6@" $PKG_CONFIG_PATH/orc-0.4.pc
sed -i -e "s@^toolsdir=.*@toolsdir=\$\{exec_prefix\}/bin/armv6@" $PKG_CONFIG_PATH/orc-0.4.pc
#COMMENT

make clean 2> /dev/null
make distclean 2> /dev/null
setenv_arm7
bash autogen.sh
./configure --host=arm-apple-darwin7 --enable-shared=no
make -j4
#find . -name "lib*.a"
cp -rvf orc/.libs/lib*.a $LOCAL_OUTDIR/armv7
cp -rvf orc-test/.libs/lib*.a $LOCAL_OUTDIR/armv7
cp -rvf tools/orcc $LOCAL_OUTDIR/armv7
echo "Creating pkg-config file"
cp -f orc-0.4.pc $PKG_CONFIG_PATH/orc-0.4.pc
sed -i -e "s@^prefix=.*@prefix=${GLOBAL_OUTDIR}@" $PKG_CONFIG_PATH/orc-0.4.pc
sed -i -e "s@^libdir=.*@libdir=\$\{exec_prefix\}/lib/armv7@" $PKG_CONFIG_PATH/orc-0.4.pc
sed -i -e "s@^toolsdir=.*@toolsdir=\$\{exec_prefix\}/bin/armv7@" $PKG_CONFIG_PATH/orc-0.4.pc

#:<< COMMENT
make clean 2> /dev/null
make distclean 2> /dev/null
setenv_i386
bash autogen.sh
./configure --enable-shared=no
make -j4
#find . -name "lib*.a"
cp -rvf orc/.libs/lib*.a $LOCAL_OUTDIR/i386
cp -rvf orc-test/.libs/lib*.a $LOCAL_OUTDIR/i386
cp -rvf tools/orcc $LOCAL_OUTDIR/i386
echo "Creating pkg-config file"
cp -f orc-0.4.pc $PKG_CONFIG_PATH/orc-0.4.pc
sed -i -e "s@^prefix=.*@prefix=${GLOBAL_OUTDIR}@" $PKG_CONFIG_PATH/orc-0.4.pc
sed -i -e "s@^libdir=.*@libdir=\$\{exec_prefix\}/lib/i386@" $PKG_CONFIG_PATH/orc-0.4.pc
sed -i -e "s@^toolsdir=.*@toolsdir=\$\{exec_prefix\}/bin/i386@" $PKG_CONFIG_PATH/orc-0.4.pc
#COMMENT

# Copy header files
mkdir -p $GLOBAL_OUTDIR/include/orc-0.4/orc && cp -rvf orc/*.h $GLOBAL_OUTDIR/include/orc-0.4/orc
mkdir -p $GLOBAL_OUTDIR/include/orc-0.4/orc-test && cp -rvf orc-test/*.h $GLOBAL_OUTDIR/include/orc-0.4/orc-test
mkdir -p $GLOBAL_OUTDIR/include/orc-0.4/testsuite/orcc && cp -rvf testsuite/orcc/*.h $GLOBAL_OUTDIR/include/orc-0.4/testsuite/orcc
# Copy libraries
for i in `find $LOCAL_OUTDIR -name "lib*.a"`;
do mkdir -p $GLOBAL_OUTDIR/lib/${i%/*} && cp -rvf $i ${GLOBAL_OUTDIR}/lib/${i}; done
cp -rvf $GLOBAL_OUTDIR/lib/${LOCAL_OUTDIR}/* $GLOBAL_OUTDIR/lib && rm -rvf $GLOBAL_OUTDIR/lib/${LOCAL_OUTDIR}
# Copy binaries
for i in `find $LOCAL_OUTDIR -name "orcc"`;
do mkdir -p $GLOBAL_OUTDIR/bin/${i%/*} && cp -rvf $i $GLOBAL_OUTDIR/bin/${i}; done
cp -rvf $GLOBAL_OUTDIR/bin/${LOCAL_OUTDIR}/* $GLOBAL_OUTDIR/bin && rm -rvf $GLOBAL_OUTDIR/bin/${LOCAL_OUTDIR}

cd ..
echo "Compiled orc"
