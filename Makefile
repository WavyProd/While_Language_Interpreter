run:
	pyinstaller --onefile while.py
	cp ./dist/while .
	rm -r dist build
	rm while.spec
