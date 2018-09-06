sudo apt-get update
conda update -y -n base conda
#conda install pytorch torchvision -y -c pytorch
conda install -y pytorch=0.3.0 -c soumith
conda install -y torchvision=0.1.8
conda install -y -c anaconda python-blosc
conda install -y -c anaconda mpi4py
conda install -y libgcc

# install and figure hdmedians
cd ~
sudo apt-get install -y gcc
source /home/ubuntu/anaconda2/bin/activate ~/anaconda2
git clone https://github.com/daleroberts/hdmedians.git
cd hdmedians
python setup.py install