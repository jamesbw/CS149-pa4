DEBUG ?= 1

SOURCES = main.cc JPEGWriter.cc CpuReference.cc ImageCleaner.cc
LIBS = -ljpeg

ifeq ($(DEBUG),1)
CFLAGS += -g -fopenmp
else
CFLAGS += -fopenmp
endif

.PHONY: all
all:
	g++ $(CFLAGS) -o ImageCleaner $(SOURCES) $(LIBS)

.PHONY: clean
clean:
	rm -f *.o ImageCleaner
