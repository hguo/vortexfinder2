#/bin/sh

./fixup_plugin.py $1 /Users/hguo/workspace/compile/ParaViewSuperbuild/build/paraview/src/paraview-build/lib/=@executable_path/../Libraries/
./fixup_plugin.py $1 /Users/hguo/workspace/projects/vortexfinder2/build_paraview/lib/=@executable_path/../Libraries/
./fixup_plugin.py $1 /Users/hguo/workspace/compile/ParaViewSuperbuild/build/install/lib/=@executable_path/../Frameworks/
