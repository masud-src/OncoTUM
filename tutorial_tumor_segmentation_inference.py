"""
MRI tumor segmentation inference tutorial

To initialize the inference of the tumor segmentation just a study is set, in order to generate a workspace. And an
input state is initialized. Here the BraTS_001 dataset is chosen, which can be used for assessing the segmentation
quality, see our respective paper. Since, the tumor segmentation is part of the mri module, such an object needs to be
initialized and the directory is manually set.

The tumor segmentation module is initialized via a setter method. The inference takes the images that are hold by the
mri object. With the initialization of the inference, the best model is automatically chosen, depending on the given
modalities. The preferred one of course is the full modality mode, followed by the t1gd and t1 models. Of course, the
used model can be changed because of the users choice. Therefore, the config attribute of the segmentation can be
manually set. For test purposes the user can chose different models with a switch 'select_model'. Note, that each model
was trained on a gpu (Nvidia a40, 48 GB VRAM, 32 core AMD epyc type 7452) and only runs with comparable hardware.
"""
import oncotum as ot
########################################################################################################################
# INPUT
path = "/home/marlon/Software/OncoTUM/data/OncoTUM/BraTS/BraTS20_Training_001/"
t1_ = path + "BraTS20_Training_001_t1.nii.gz"
t1ce_ = path + "BraTS20_Training_001_t1ce.nii.gz"
t2_ = path + "BraTS20_Training_001_t2.nii.gz"
flair_ = path + "BraTS20_Training_001_flair.nii.gz"
########################################################################################################################
# TUMOR SEGMENTATION
tum_seg = ot.TumorSegmentation()
tum_seg.mri.subj_id = "BraTS20_Training_001"
tum_seg.mri.t1_dir = t1_
tum_seg.mri.t1ce_dir = t1ce_
tum_seg.mri.t2_dir = t2_
tum_seg.mri.flair_dir = flair_
select_model = True
if select_model:
    model = "full"  # "full", "t1", "t1gd", "t2", "flair"
    tum_seg.config = "/home/marlon/Software/OncoTUM/data/" + model + "/hyperparam.yaml"
#tut_04_model = True
#if tut_04_model:
#    tum_seg.config = of.STUDIES_DIR + "/tut_04/der/tumor_segmentation/full_neural_net/hyperparam.yaml"
tum_seg.run_inference()
