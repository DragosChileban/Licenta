# Licenta

#### CrashSplat: 2D to 3D Vehicle Damage Segmentation in Gaussian Splatting by Dragos-Andrei Chileban - Installation and running guide

#### There are two git repositories that contain the source code of the project.

1. https://github.com/DragosChileban/Licenta - contains a clean and ready to install version of the application. Details are presented below.
2. https://github.com/DragosChileban/Thesis - contains the rest of the code used for training, running experiments, performing ablation studies. See the footnote for more details.

## 1. Installing necessary external dependencies (command line tools)
First you will install some modules needed for running 3D Gaussian Splatting.
I will include commands that I used for installing them on my system (MacOS), but also installation guides from the source modules. 
1. ### FFmpeg
<pre>
brew install ffmpeg         #for installing on MacOS
</pre>
[Download page for other systems](https://ffmpeg.org/download.html).

2. ### COLMAP
<pre>
brew install colmap         #for installing on MacOS
</pre>
[Download page for other systems](https://colmap.github.io/install.html).

3. ### OpenSplat
<pre>
brew install cmake
brew install opencv
brew install pytorch

sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

git clone https://github.com/pierotofy/OpenSplat OpenSplat
cd OpenSplat
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ -DGPU_RUNTIME=MPS .. && make -j$(sysctl -n hw.logicalcpu)
./opensplat
</pre>
They have a more detailed installation guide [here](https://github.com/pierotofy/OpenSplat).

*Note OpenSplat needs to be installed inside the root folder of the repo (/Licenta/OpenSplat should be a valid path).
In case you decide to install it somewhere else, you should modify the "splat_build_dir" variable from scripts/run_script.py with the location of the OpenSplat/build directory.

## 2. Cloning the github repository and installing python libraries

<pre>git clone https://github.com/DragosChileban/Licenta.git 
cd Licenta
conda create -n $name python=3.9   #create a conda virtual environment (replace $name with desired name)
conda activate $name               #activate the conda environment
pip install -r requirements.txt    #install required python libraries using pip 
pip install -e .                   #install the module in edit mode
</pre>

## 3. Download and copy the model checkpoints and demo examples
Next, you will need to download the weights for the segmentation model from [here](https://drive.google.com/drive/folders/1EoeuTbcXlxWOrH2i0p9ZygxUZFacvN3c?usp=sharing) and copy the file to Licenta/checkpoints.
The app also includes 3 examples of 3D car reconstructions with segmentation damages. You need to download them from [here](https://drive.google.com/drive/folders/19TriTDku4Z2L5vrRDScYspxySbk4oOQl?usp=sharing) and add in Licenta/CrashSplat/web_gui/demo.

## 4. Running the front-end app
<pre>
cd CrashSplat/web_gui/3dgs          #navigate to the folder where the JavaScript app is located
npm install                         #install npm packages
npm run dev                         #run the app and access the localhost url
</pre>

## 5. Running the backend-end server
<pre>
cd CrashSplat/web_gui/flask      #navigate to the folder where the Flask app is located
python app.py                    #run the app
</pre>

### *Note: The mentioned repository contains only the code for the graphical interface and the necessary function for running the main functionalities of the app. This thesis being research-oriented, we developed other programs and scripts for training models, running experiments, comparing visual results and creating meaningful diagrams. All of these can be accessed in this [repo](https://github.com/DragosChileban/Thesis).
