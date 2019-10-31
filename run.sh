bsub -J homo -q batch -R "span[ptile=36]" -n 8 -e .err -o tmp.out "python test_ex1.py 32 32"
sleep 10
bsub -J homo -q batch -R "span[ptile=36]" -n 8 -e .err -o tmp.out "python test_ex1.py 16 32"
