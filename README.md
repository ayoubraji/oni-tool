# oni-tool
This simple tool extracts rgb, rgb-d, depth and point clouds from oni file, generated from PrimeSense sensor, such as Asus Xtion and Microsoft Kinect.
It will be updated soon with new features.

# Dependencies
- OpenCV
- Openni
- PCL 1.7
- CMake

# Building
The tool has been tested on Linux (Ubuntu 16.04)
```
mkdir build
cd build
cmake ..
make

```
# Usage
```
./oni_converter [onifile_path]

```
