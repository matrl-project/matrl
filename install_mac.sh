pip install -r requirements.txt
brew install openmpi
git clone https://github.com/openai/spinningup.git 
cd spinningup 
pip install -e . 
cd ../
git clone https://github.com/koulanurag/ma-gym.git 
cd ma-gym 
git checkout 8eca34e534c0f1f702c49517005bcee0cf3d9a3e
pip install -e . 
cd ../
wget https://www.roboti.us/download/mujoco200_macos.zip
unzip mujoco200_macos 
mv mujoco200_macos ~/.mujoco/mujoco200
brew install unrar
AutoROM
