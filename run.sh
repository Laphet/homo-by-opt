bsub -J homo -q batch -R "span[ptile=36]" -n 8 -e output/b4r32.err -o output/b4r32.out "python test_ex1.py 4 32"
sleep 10
bsub -J homo -q batch -R "span[ptile=36]" -n 8 -e output/b8r32.err -o output/b8r32.out "python test_ex1.py 8 32"
sleep 10
bsub -J homo -q batch -R "span[ptile=36]" -n 8 -e output/b16r32.err -o output/b16r32.out "python test_ex1.py 16 32"
sleep 10
bsub -J homo -q batch -R "span[ptile=36]" -n 8 -e output/b32r32.err -o output/b32r32.out "python test_ex1.py 32 32"
sleep 10
bsub -J homo -q batch -R "span[ptile=36]" -n 8 -e output/ex1.err -o output/ex1.out "python ex1.py"
bsub -J homo -q batch -R "span[ptile=36]" -n 8 -e output/opt.err -o output/opt.out "python EllipticCoeffOpt.py"