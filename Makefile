.PHONY: all prepare build clean ninja core_unit_tests

all: prepare build

prepare:
	rm -rf build
	mkdir build

build:
	cd build && cmake .. && cmake --build .

core_unit_tests:
	cd build && cmake .. && cmake -DENABLE_CUDA=OFF --build . --target core_unit_tests

clean:
	rm -rf build

ninja:
	cd build && cmake -S .. -B . Ninja