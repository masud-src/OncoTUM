# OncoTUM
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) 

OncoTUM is a **tum**our segmentation package for medical images, that are distorted due its **onco**logical disease. To 
perform the segmentation processes, the convolutional network (unet) of T. Henry et al. [1] is adapted. Therefore, the
code (https://github.com/lescientifik/open_brats2020) is tailored with this repository to the framework of OncoFEM. 
Due to that, no fork is done and the algorithm is appended about a modality agnostic mode and cpu mode. Furthermore, the 
results of tested modality agnostic modes are shown in the following.

* [Exemplary results](#results)
* [Integration of OncoTUM](#integration)
* [Software availability](#software)
* [Installation and machine requirements](#installation)
* [Tutorial](#tutorial)
* [How to](#howto)
* [Known bugs](#bugs)
* [How to cite](#howtocite)
* [Literature](#literature)

## <a id="results"></a> Examplary results

In the following the results of the modality agnostic modes are compared to the full modality mode. 

The following image shows the segmentation based only on the t1 image.

<p align="center">
 <img src="t1.png" alt="t1.png" width="800"/>
</p>

The next image shows the segmentation based on the t1gd image.

<p align="center">
 <img src="t1gd.png" alt="t1gd.png" width="800"/>
</p>

The next image shows the segmentation based on the t2 image.

<p align="center">
 <img src="t2.png" alt="t2.png" width="800"/>
</p>

The next image shows the segmentation based on the flair image.

<p align="center">
 <img src="flair.png" alt="flair.png" width="800"/>
</p>

The next image shows the segmentation based on the full modality image.
<p align="center">
 <img src="full.png" alt="full.png" width="800"/>
</p>

The algorithm is capable to take also just a reduced set of modalities, e.g. (t1, t1gd, flair). 

## <a id="integration"></a> Integration of OncoTUM
OncoTUM is part of a module based umbrella software for numerical simulations of patient-specific cancer diseases, see 
following figure. From given input states of medical images the disease is modelled and its evolution is simulated 
giving possible predictions. In this way, a digital cancer patient is created, which could be used as a basis for 
further research, as a decision-making tool for doctors in diagnosis and treatment and as an additional illustrative 
demonstrator for enabling patients understand their individual disease. All parts resolve to an open-access framework, 
that is ment to be an accelerator for the digital cancer patient. Each module can be installed and run independently. 
The current state of development comprises the following modules

- OncoFEM (https://github.com/masud-src/OncoFEM)
- OncoGEN (https://github.com/masud-src/OncoGEN)
- OncoTUM (https://github.com/masud-src/OncoTUM)
- OncoSTR (https://github.com/masud-src/OncoSTR)
<p align="center">
 <img src="workflow.png" alt="workflow.png" width="2000"/>
</p>
 
## <a id="software"></a> Software availability

You can either follow the installation instruction below or use the already pre-installed virtual boxes via the 
following Links:

- Version 0.1.0:  https://doi.org/10.18419/darus-3720

## <a id="installation"></a> Installation and Machine Requirements

This installation was tested on a virtual box created with a linux mint 21.2 cinnamon, 64 bit system and 8 GB RAM on a 
local machine (intel cpu i7-9700k with 3.6 GHz, 128 GB RAM). It is possible to install OncoTUM as a stand-alone or 
together with OncoFEM (https://github.com/masud-src/OncoFEM) and its anaconda environment. To do so activate that first
````bash
conda activate oncofem
````
- Continue with the following lines either in that environment or global
````bash
git clone https://github.com/masud-src/OncoTUM/
cd OncoTUM
pip install -r requirements.txt
````
- OncoTUM is installed on the local system with
````bash
python3 -m pip install .
````
- To use prepared weights download the necessary material provided on DaRUS 
(https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-4647) and place the 'data' folder next to
the oncotum folder. The following script will also set path variables and you should restart the terminal
````bash
chmod +x set_config.sh.
./set_config.sh
````
- The package can now be used. To test the correct installation, run a python script with the following code line.
````bash
import oncotum
````

## <a id="tutorial"></a> Tutorial

There is a tutorial for the umbrella software project provided on DaRUS 
(https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-4639). You can download and run the
tutorial_structure_segmentation.py file by run the following lines in your desired directory.
````bash
curl --output tutorial https:/darus.uni-stuttgart.de/api/access/dataset/:persistentId/?persistentId=doi:10.18419/darus-3679
````
To run this tutorial, you also need to download the first six training datasets from kaggle 
(https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation). Either you download from the web
interface and save it in the following location
````bash
tutorial/data/BraTS/
````
or you use the kaggle API. Be aware that this will download the full set and its recommended to use the web interface
````bash
kaggle datasets download -d awsaf49/brats20-dataset-training-validation -p .
unzip brats20-dataset-training-validation.zip "BraTS20_Training_001/*" "BraTS20_Training_002/*" "BraTS20_Training_003/*" "BraTS20_Training_004/*" "BraTS20_Training_005/*" "BraTS20_Training_006/*" unzip brats20-dataset-training-validation.zip "BraTS20_Training_001/*" "BraTS20_Training_002/*" "BraTS20_Training_003/*" "BraTS20_Training_004/*" "BraTS20_Training_005/*" "BraTS20_Training_006/*" -d ./tutorial/data/BraTS/
````
The tutorial can be started with
````bash
conda activate oncotum
python oncotum_tut_01_inference.py
````

## <a id="howto"></a> How To

You can modify the existing algorithms, respectively expand the existing by your own. Therefore, you can fork and ask 
for pull requests.

## <a id="bugs"></a> Known bugs

Sometimes the training in cpu modes fails in non-reproducible errors.

## <a id="howtocite"></a> How to cite

TBD

## <a id="literature"></a> Literature

<sup>1</sup> Henry, T. et al. (2021). Brain Tumor Segmentation with Self-ensembled, Deeply-Supervised 3D U-Net Neural 
             Networks: A BraTS 2020 Challenge Solution. In: Crimi, A., Bakas, S. (eds) Brainlesion: Glioma, Multiple 
             Sclerosis, Stroke and Traumatic Brain Injuries. BrainLes 2020. Lecture Notes in Computer Science(), 
             vol 12658. Springer, Cham. https://doi.org/10.1007/978-3-030-72084-1_30
