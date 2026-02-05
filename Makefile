.PHONY: all prepare build clean

all: prepare build

prepare:
	rm -rf build
	mkdir build

build:
	cd build && cmake .. && cmake --build .

clean:
	rm -rf build