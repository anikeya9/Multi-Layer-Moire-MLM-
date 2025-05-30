# global vars
TOL1 := 0.00001 
TOL3 := 0.00001

RESULTSDIR := results
OUTFILE1 := $(RESULTSDIR)/layer12matches.csv 
OUTFILE2 := $(RESULTSDIR)/layer12matches_with_stable_basis.csv 
OUTDIR3  := $(RESULTSDIR)/layer3matches

VASPFILE1 := vaspfiles/bto.vasp 

all: match1 .WAIT select2 .WAIT match3

match1:
	python 1-match-lattice.py --firstlayer=$(VASPFILE1) --secondlayer=$(VASPFILE1) --tolerance=$(TOL1) --outfile=$(OUTFILE1)
	# cat $(OUTFILE1) | grep 22.62

select2:
	#python 2-select.py  --firstlayer=$(VASPFILE1) --secondlayer=$(VASPFILE1) --matchcsv=$(OUTFILE1) --outfile=$(OUTFILE2)
	python 2-select.py  --firstlayer=$(VASPFILE1) --matchcsv=$(OUTFILE1) --outfile=$(OUTFILE2)
	echo ""; bat $(OUTFILE2) || cat $(OUTFILE2)

match3:
	python 3-match-third.py --firstlayer=$(VASPFILE1) $(OUTFILE2) --tolerance=$(TOL3) --outdir=$(OUTDIR3)
	bat $(OUTDIR3)/*.csv || cat $(OUTDIR3)/*.csv   
	
select4:
	python 2-select.py --matchcsv=results/layer3matches/0.csv --firstlayer=vaspfiles/bto.vasp --outfile=$(OUTDIR3)/0-good-basis.csv
	python 2-select.py --matchcsv=results/layer3matches/1.csv --firstlayer=vaspfiles/bto.vasp --outfile=$(OUTDIR3)/1-good-basis.csv

format:
	python -m ruff format .

deps:
	uv pip install -r req.txt || pip install -r req.txt


git:
	git add .
	git commit -m auto_commit
	git push

clean:
	rm -r $(RESULTSDIR)/*.csv || true
	rm -r $(OUTDIR3)/*.csv || true
	echo ""; ls $(RESULTSDIR);
	echo ""; ls $(OUTDIR3)
