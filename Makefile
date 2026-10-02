CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Ithird_party

compose: src/compose.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

data:
	python3 tools/fetch.py

run: compose
	mkdir -p out && ./compose layout.txt out/composite.png

hugs:
	python3 tools/fetch_hugs.py

veins:
	python3 tools/veins.py --debug

clean:
	rm -f compose out/composite.png

.PHONY: data run hugs veins clean
