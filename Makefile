.PHONY: all preapare build clean

all: prepare build

prepare:
	rm -rf build
	mkdir build

build:
	cd build && cmake .. && cmake --build .

clean:
	rm -rf build