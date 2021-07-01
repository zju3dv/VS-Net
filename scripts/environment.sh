WORK_DIR=$(pwd)

cd ../utils/nn_cuda
rm -rf build
rm -rf dist
rm -rf *egg*
python setup.py install
cd $WORK_DIR

cd ../utils/ransac_voting_gpu
rm -rf build
rm -rf dist
rm -rf *egg*
python setup.py install
cd $WORK_DIR
