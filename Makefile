.PHONY: all prepare build clean ninja core_unit_tests

all: prepare_release build


# =============================== RELEASE ===============================

prepare_release:
	rm -rf build/release
	mkdir -p build/release

build:
	cd build/release && cmake ../.. -DCMAKE_BUILD_TYPE=Release && cmake --build .


# ================================ DEBUG ================================

prepare_debug:
	rm -rf build/debug
	mkdir -p build/debug

debug: prepare_debug
	cd build/debug && cmake ../.. -DCMAKE_BUILD_TYPE=Debug && cmake --build .


# ================================ UTILS ================================

core_unit_tests:
	cd build/release && cmake ../.. && ctest --output-on-failure

ninja:
	cd build/release && cmake ../.. -GNinja && cmake --build .

clean:
	rm -rf build