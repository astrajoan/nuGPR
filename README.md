# nuGPR

## Instructions for running the code

First ensure you have the following C++ packages installed:

* CUDA (minimum version 11.8)
* gFlags
* gLog
* nlohmann_json
* RAPIDS memory manager

To compile and the nuGPR program, from the root directory of this repo:
```
$ mkdir build && cd build
$ cmake ..
$ make -j$(nproc)
$ ./nuGPR --super
```
