PROJECT = audio_publisher

# BRAND expects an executable <node>.bin. The node is pure Python, so the
# "binary" is the script itself with the executable bit set (the shebang
# selects the active virtualenv's interpreter).
all:
	cp $(PROJECT).py $(PROJECT).bin
	chmod +x $(PROJECT).bin

clean:
	$(RM) $(PROJECT).bin
