NVCC      ?= nvcc
SIF       ?= /scratch/csee147/csee147env.sif
APPTAINER  = apptainer exec --nv $(SIF)
FFMPEG    ?= $(HOME)/.local/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
ARCH      ?= -arch=native
NVCCFLAGS  = -O3 -std=c++17 --use_fast_math $(ARCH) -Isrc
SRCS  = src/renderer.cu src/fluid_sim.cu src/main.cpp
HDRS  = src/common.cuh src/renderer.h src/fluid_sim.h
ifdef NANOVDB
NVCCFLAGS += -I$(NANOVDB) -DUSE_NANOVDB
endif
.PHONY: all run bender bender-run video clean
all: render
render: $(SRCS) $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@
run: render
	./render 120
bender:
	$(APPTAINER) make render
bender-run:
	$(APPTAINER) make run
video:
	$(FFMPEG) -y -framerate 30 -i frame.%04d.ppm -pix_fmt yuv420p smoke.mp4
clean:
	rm -f render frame.*.ppm