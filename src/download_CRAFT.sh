chmod -R -f +w CRAFT || true
rm -rf CRAFT || true
git clone -n https://github.com/clovaai/CRAFT-pytorch.git CRAFT
cd ./CRAFT
git checkout e332dd8b718e291f51b66ff8f9ef2c98ee4474c8
git apply ../CRAFT.patch
mkdir ./weights
cd ./weights
gdown --id 1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ
gdown --id 1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf
gdown --id 1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO
cd ../..

