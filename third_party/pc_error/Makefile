.PHONY: all
all: pc_error

.PHONY: clean
clean:
	rm -f pc_error

pc_error: clockcom.cpp clockcom.hpp main.cpp pcc_distortion.cpp pcc_distortion.hpp pcc_processing.cpp pcc_processing.hpp
	g++ -lboost_program_options -O2 -o pc_error $^
