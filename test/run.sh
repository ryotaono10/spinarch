ulimit -s unlimited
OMP_NUM_THREADS=8 
MKL_NUM_THREADS=8 
python3 ~/Tools/MyProg/ED-CIPSI/hphi_cipsi_lite_para_test.py namelist.def --build-procs 8 --accel-matvec --hb-preselect 
