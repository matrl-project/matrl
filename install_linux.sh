pip install -r requirements.txt
sudo apt-get update && sudo apt-get install libopenmpi-dev
git clone https://github.com/openai/spinningup.git 
cd spinningup 
pip install -e . 
cd ../
git clone https://github.com/koulanurag/ma-gym.git 
cd ma-gym 
git checkout 8eca34e534c0f1f702c49517005bcee0cf3d9a3e
pip install -e . 
cd ../
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux 
mv mujoco200_linux ~/.mujoco/mujoco200
AutoROM
