TARGET = main_cuda
SRC = main.cu
CC = nvcc
CFLAGS = -lcurand -O2

.PHONY: all clean

all: compile run

compile:
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

run:
	./$(TARGET)

clean:
	rm -rf $(TARGET)
