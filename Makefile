NODE_DIRS := $(wildcard nodes/*)

.PHONY: all clean $(NODE_DIRS)

all: $(NODE_DIRS)

$(NODE_DIRS):
	$(MAKE) -C $@

clean:
	for d in $(NODE_DIRS); do $(MAKE) -C $$d clean; done
