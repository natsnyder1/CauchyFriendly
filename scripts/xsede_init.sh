module purge
module reset
#module load gpu/0.15.4
#module load cuda/11.0.2
#module load intel-mkl/2020.3.279
#module load py-pip/20.2
#module load python37
#pip3 install matplotlib numpy 
#pip3 install globus-cli 
module load cpu/0.15.4
module load gcc/10.2.0 #9.2.0
module load valgrind/3.15.0
module load intel-mkl/2019.1.144 #2018
module load py-pip/20.2
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/natsnyder1/cauchy/glpk-4.65/src/.libs
echo "Modules have been successfully loaded, expanse CPU cluster is initialized"
#echo "Please now run the 'request_gpus.sh' script to acquire the GPU resources"

