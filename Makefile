Sim: Handwriting.h LIFDesktop.c
	gcc -Wall -O2  -std=c99 LIFDesktop.c -o Sim

Output/flash.bin Handwriting.h: TextToBin
	./TextToBin > Handwriting.h
	@echo "New version of Handwriting.h has been generated"

TextToBin:
	g++ TextToBin.c -o TextToBin

.PHONY: clean

clean:
	rm -f Handwriting.h
	rm -f TextToBin
	rm -f Sim
