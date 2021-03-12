CC=gcc
CFLAGS=-std=gnu99 -Wall -Wextra
LDFLAGS=-lm -lgomp
GL_LDFLAGS=-lGL -lglfw

# Files
TARGETS=tiny_ising demo

# Rules
all: $(TARGETS)

tiny_ising: tiny_ising.o ising.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

demo: demo.o ising.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(GL_LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean all

