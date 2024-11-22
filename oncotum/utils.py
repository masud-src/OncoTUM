"""
Definition of utility functions for the tumor segmentation

Variables:
    trs:                            List of transpositions
    flips:                          List of possible flips
    rots:                           List of possible rotations
    transform_list:                 List of transformations (transpositions and rotations)

Classes:
    Ranger:                         Optimizer class, from https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
    WeightSWA:                      SWA or fastSWA, from https://github.com/benathi/fastswa-semi-sup
    AverageMeter:                   Computes and stores the average and current value.
    ProgressMeter:                  Computes and stores the progress.
    Brats:                          Creates one or multiple datasets in the BraTS declaration. One Patient can have all
                                    or a subset of the standard modalities.
    EDiceLoss:                      Dice loss tailored to Brats need.

Functions:
    simple_tta:                     Perform all transpose/mirror transform possible only once.
    apply_simple_tta:               Applies the transpose and mirror transformations, can average over the whole stack.
    revert_tta_factory:             Reverts the transpose and mirror transformation into original state.
    get_datasets:                   Get the training data set from the given folder. For adaptive training mode the
                                    wanted patterns and a randomised set of images can be replaced by a blank image.
    master_do:                      Help calling function only on the rank0 process id ddp
    save_checkpoint:                Save Training state.
    reload_ckpt:                    Reloads the training onto set checkpoint with the scheduler.
    reload_ckpt_bis:                Reloads the training onto set checkpoint.
    calculate_metrics:              Calculates the metrics [Hausdorff, Dice, Sens, Spec] of the prediction.
    update_teacher_parameters:      Updates the teacher parameters. Use the true average until the exponential average
                                    is more correct.
    determinist_collate:            Pads the batch to its maximum shape and uses the default torch collate command.
    pad_batch_to_max_shape:         Pads a batch to the maximum possible shape.
    pad_batch1_to_compatible_size:  Pads a batch to compatible size.
    pad_single_to_compatible_size:  Pads a single modality into a compatible size.
    pad_or_crop_image:              Pads or crops the image into a target size.
    get_left_right_idx_should_pad:  Returns the right and left indices that should be cropped from the image.
    get_crop_slice:                 Returns the size of a cropped image.
    normalize:                      Basic min max scaler.
    irm_min_max_preprocess:         Remove outliers voxels first, then min-max scale.
    zscore_normalize:               Normalisation of z_score
    remove_unwanted_background:     Use to crop zero_value pixel from MRI image.
    random_crop2d:                  Crop randomly but identically all images given. Could be used to pass both mask and
                                    image at the same time. Anything else will throw.
    random_crop3d:                  Crop randomly but identically all images given. Could be used to pass both mask and
                                    image at the same time. Anything else will throw.
    randomise_blanks:               Changes randomly in the list of mri modality paths, one or more modality into a
                                    blank image for training of adaptive model
    count_parameters:               Count trainable parameters of neural network
    save_metrics:                   Saves the metrics into the respective folder.
"""
import pprint
import random
from typing import Any, Union
from matplotlib import pyplot as plt
from numpy import logical_and as l_and, logical_not as l_not
from random import randint, random, sample, uniform
from scipy.spatial.distance import directed_hausdorff
from torch import distributed as dist
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
import pathlib
import SimpleITK as sitk
import numpy as np
import os
from sklearn.model_selection import KFold
import math
import torch
from torch.optim.optimizer import Optimizer
import itertools as it
from itertools import combinations, product
import nibabel as nib
import copy
import time
from .models import DataAugmenter

ONCOTUM_DIR = os.environ['ONCOTUM']
FULL_MODEL_DIR = ONCOTUM_DIR + "/data/tumor_segmentation/full/hyperparam.yaml"
CYCLE_1_4_MODEL_T1_DIR = ONCOTUM_DIR + "/data/tumor_segmentation/t1/hyperparam.yaml"
CYCLE_1_4_MODEL_T1GD_DIR = ONCOTUM_DIR +  "/data/tumor_segmentation/t1gd/hyperparam.yaml"
CYCLE_1_4_MODEL_T2_DIR = ONCOTUM_DIR +  "/data/tumor_segmentation/t2/hyperparam.yaml"
CYCLE_1_4_MODEL_FLAIR_DIR = ONCOTUM_DIR +  "/data/tumor_segmentation/flair/hyperparam.yaml"
TUMOR_SEGMENTATION_WEIGHTS_DIR = [("FULL_MODEL_DIR", FULL_MODEL_DIR),
                                  ("CYCLE_1_4_MODEL_T1_DIR", CYCLE_1_4_MODEL_T1_DIR),
                                  ("CYCLE_1_4_MODEL_T1GD_DIR", CYCLE_1_4_MODEL_T1GD_DIR),
                                  ("CYCLE_1_4_MODEL_T2_DIR", CYCLE_1_4_MODEL_T2_DIR),
                                  ("CYCLE_1_4_MODEL_FLAIR_DIR", CYCLE_1_4_MODEL_FLAIR_DIR)]
TUMOR_SEGMENTATION_PATH = "tumor_segmentation/"
TRAINING_RUN = "/media/marlon/data/run/"
TRAINING_NULL_IMAGE = "/data/0_im.nii.gz"
HAUSSDORFF = "haussdorff"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
METRICS = [HAUSSDORFF, DICE, SENS, SPEC]


trs = list(combinations(range(2, 5), 2)) + [None]
flips = list(range(2, 5)) + [None]
rots = list(range(1, 4)) + [None]
transform_list = list(product(flips, rots))


class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.9, 0.999), eps=1e-8, weight_decay=0):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # now we can get to work...
        for group in self.param_groups:
            group["step_counter"] = 0  # print("group step counter init")

        # look ahead params
        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # lookahead weights
        self.slow_weights = [[p.clone().detach() for p in group['params']] for group in self.param_groups]

        # don't use grad for lookahead weights
        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        # loss = closure()

        # ------------ radam
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma > 4:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size)

                p.data.copy_(p_data_fp32)

        # ---------------- end radam step

        # look ahead tracking and updating if latest batch = k
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p, q in zip(group['params'], slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha, p.data - q.data)
                p.data.copy_(q.data)

        return loss


class WeightSWA(object):
    """
    SWA or fastSWA
    Taken from https://github.com/benathi/fastswa-semi-sup
    """

    def __init__(self, swa_model):
        self.num_params = 0
        self.swa_model = swa_model  # assume that the parameters are to be discarded at the first update

    def update(self, student_model):
        self.num_params += 1
        print("Updating SWA. Current num_params =", self.num_params)
        if self.num_params == 1:
            print("Loading State Dict")
            self.swa_model.load_state_dict(student_model.state_dict())
        else:
            inv = 1.0 / float(self.num_params)
            for swa_p, src_p in zip(self.swa_model.parameters(), student_model.parameters()):
                swa_p.data.add_(-inv * swa_p.data)
                swa_p.data.add_(inv * src_p.data)

    def reset(self):
        self.num_params = 0


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Computes and stores the progress.
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Brats(torch.utils.data.dataset.Dataset):
    """
    Creates one or multiple datasets in the BraTS declaration. One Patient can have all or a subset of the standard
    modalities.
    """
    def __init__(self, data, patterns:list[str], rand_blank:bool, benchmarking=False,
                 training=True, debug=False, data_aug=False, no_seg=False, normalisation="minmax"):
        super(Brats, self).__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.debug = debug
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.patterns = patterns
        if type(data) is list:
            self.create_multiple_dataset(data, rand_blank, no_seg)
        if type(data) is MRI:
            self.create_single_dataset(data, rand_blank, no_seg)

    def create_single_dataset(self, mri, rand_blank, no_seg=False):
        patient_id = mri.subj_id
        paths = [mri.t1_dir, mri.t1ce_dir, mri.t2_dir, mri.flair_dir]
        paths = randomise_blanks(rand_blank, paths)
        patient = dict((x.replace("_", ""), paths[i]) for i, x in enumerate(self.patterns))
        patient["id"] = patient_id
        patient["seg"] = mri.seg_dir if not no_seg else None
        self.datas.append(patient)

    def create_multiple_dataset(self, patients_dir, rand_blank, no_seg=False):
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            paths = [str(patient_dir) + os.sep + str(patient_id) + str(value) + ".nii.gz" for value in self.patterns]
            paths = randomise_blanks(rand_blank, paths)
            patient = dict((x.replace("_", ""), paths[i]) for i, x in enumerate(self.patterns))
            patient["id"] = patient_id
            patient["seg"] = str(patient_dir) + os.sep + str(patient_id) + "_seg.nii.gz" if not no_seg else None
            self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax":
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        patient_image = np.stack([patient_image[key] for key in patient_image])
        if _patient["seg"] is not None:
            et = patient_label == 4
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 4, patient_label == 1)
            wt = np.logical_or(tc, patient_label == 2)
            patient_label = np.stack([et, tc, wt])
        else:
            patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        return dict(patient_id=_patient["id"],
                    image=patient_image,
                    label=patient_label,
                    seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True,
                    )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas) if not self.debug else 3


class EDiceLoss(torch.nn.Module):
    """
    Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)

        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class MRI:
    """
    MRI is the base class for the pre-processing of the patient-specific input data. Herein, the basic structural
    modalities are available via the respective attribute. In order to homogenize and further pre-process more
    attributes about image properties and masks of tumor and brain tissue compartments are hold.

    *Attributes*:
        work_dir:           String of the working directory
        t1_dir:             String, direction of t1 modality
        t1ce_dir:           String, direction of t1ce modality
        t2_dir:             String, direction of t2 modality
        flair_dir:          String, direction of flair modality
        seg_dir:            String, direction of segmentation
        full_ana_modality:  Bool, check if all structural modalities are given (t1, t1ce, t2, flair)
        affine:             Array of image affine, each modality is co-registered to that
        shape:              Shape of the image each modality is co-registered to
        ede_mask:           Binary mask image of the edema
        act_mask:           Binary mask image of the active core
        nec_mask:           Binary mask image of the necrotic core

    *Methods*:
        set_affine_and_shape:   Loads first given measurement and takes affine and shape
        isFullModality:         Checks if input state has full structural modality
    """
    def __init__(self):
        self.work_dir = None
        self.subj_id = None
        self.t1_dir = None
        self.t1ce_dir = None
        self.t2_dir = None
        self.flair_dir = None
        self.seg_dir = None
        self.full_ana_modality = None
        self.affine = None
        self.shape = None
        self.ede_mask = None
        self.act_mask = None
        self.nec_mask = None

    def set_affine_and_shape(self, image: str) -> None:
        """
        Sets affine and shape. Affine and shape can be accessed via self.affine and self.shape.

        :param image: Directory to an image that will be used to set the affine and shape
        :return: None
        """
        try:
            image_ = nib.load(image)
            self.affine = image_.affine
            self.shape = image_.shape
        except:
            print("no nifti image, need to set affine after conversion.")

    def isFullModality(self) -> bool:
        """
        Checks if all structural gold standard entities are available. Returns boolean value and sets
        self.full_ana_modality

        :return: boolean value
        """
        list_available_modality = [self.t1_dir, self.t1ce_dir, self.t2_dir, self.flair_dir]
        if None in list_available_modality:
            self.full_ana_modality = False

        missing_paths = [path for path in list_available_modality if not os.path.exists(path)]
        if missing_paths:
            self.full_ana_modality = False

        return self.full_ana_modality


class ModelParam:
    """
    In this class, parameters describing the model are hold. The model is described with the following arguments.

    *Arguments*:
        arch:                                   String, Definition of the model architecture.
        training_data:                          String, for the path of the training data. Must be according to BraTS
        full_training_data:                     Bool, if full set of data should be used for training.
        input_patterns:                         List of strings, wherein the input channels are defined.
        random_blank_image:                     Bool, if true randomised images of a data set are set to a zero image.
        output_channel::                        Herein, the output channel can be set. Default: active, necrotic, edema
        width:                                  Int, defines the width of the first layer, which are then doubled
        optimizer:                              String, defines the optimization scheme (ranger, adam, sgd, adamw)
        dropout:                                Float, defines the dropout likelihood
        warm_restart:                           Bool, restart of a training (not tested)
        max_epochs:                             Int, number of training epochs
        batch_size:                             Int, number of batch size (T. Henry: leave it at one)
        lr:                                     Float, learning rate
        weight_decay:                           Float, weight decay
        no_float_precision_16:                  Bool, sets the precision of a lower size
        norm_layer:                             String, defines the norm layer of a model, only "group" is defined
        n_warm_epochs:                          Int, defines warm up epochs
        val_step_interval:                      Int, defines validation step interval in training
        deep_sup:                               Bool, switch for deep supervision
        fold:                                   Int, number of instances contained in your dataset
        stochastic_weight_averaging:            Bool, switch for weight averaging of mulliple sets
        stochastic_weight_averaging_repeat:     Int, intervall of average weighting
        workers:                                Int, number of working processes
        resume:                                 Bool, for resuming the training of a model (not tested)
    """
    def __init__(self):
        self.arch = "EquiUnet"
        self.training_data = None
        self.full_training_data = False
        self.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        self.random_blank_image = False
        self.output_channel = 3
        self.width = 48
        self.optimizer = "ranger"
        self.dropout = 0.0
        self.warm_restart = False
        self.max_epochs = 200
        self.batch_size = 1
        self.lr = 0.0001
        self.weight_decay = 0.0
        self.no_float_precision_16 = False
        self.norm_layer = "group"
        self.n_warm_epochs = 3
        self.val_step_intervall = 3
        self.deep_sup = False
        self.fold = 0
        self.stochastic_weight_averaging = False
        self.stochastic_weight_averaging_repeat = 5
        self.workers = 2
        self.resume = False


def mkdir_if_not_exist(directory: str) -> str:
    """
    Makes directory if not exists and returns the string

    :param dir: String of directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def image2array(image_dir:str) -> tuple[Any, Any, Any]:
    """
    Takes a directory of an image and gives a numpy array.

    *Arguments*:
        image_dir:      String of a Nifti image directory
    *Returns*:
        numpy array of image data, shape, affine
    """
    orig_image = nib.load(image_dir)
    return copy.deepcopy(orig_image.get_fdata()), orig_image.shape, orig_image.affine


def image2mask(image_dir:str, compartment:int=None, inner_compartments:list[int]=None) -> np.ndarray:
    """
    Gives deep copy of original image with selected compartments.

    *Arguments*:
        image_dir:          String to Nifti image
        compartment:        Int, identifier of compartment that shall be filtered
        inner_compartments: List of inner compartments that also are included in the mask
    *Returns*:
        mask:               Numpy array of the binary mask
    """
    mask, _, _ = image2array(image_dir)
    unique = list(np.unique(mask))
    unique.remove(compartment)
    for outer in unique:
        mask[np.isclose(mask, outer)] = 0.0
    mask[np.isclose(mask, compartment)] = 1.0
    if inner_compartments is not None:
        for comp in inner_compartments:
            mask[np.isclose(mask, comp)] = 1.0
            unique.remove(comp)
    return mask


def simple_tta(x):
    """Perform all transpose/mirror transform possible only once.

    Sample one of the potential transform and return the transformed image and a lambda function to revert the transform
    Random seed should be set before calling this function
    """
    out = [[x, lambda z: z]]
    for flip, rot in transform_list[:-1]:
        if flip and rot:
            trf_img = torch.rot90(x.flip(flip), rot, dims=(3, 4))
            back_trf = revert_tta_factory(flip, -rot)
        elif flip:
            trf_img = x.flip(flip)
            back_trf = revert_tta_factory(flip, None)
        elif rot:
            trf_img = torch.rot90(x, rot, dims=(3, 4))
            back_trf = revert_tta_factory(None, -rot)
        else:
            raise
        out.append([trf_img, back_trf])
    return out


def apply_simple_tta(model, x, average=True):
    """
    Applies the simple transpose and mirror transformations and can average over the whole stack.
    """
    todos = simple_tta(x)
    out = []
    for im, revert in todos:
        if model.deep_supervision:
            out.append(revert(model(im)[0]).sigmoid_().cpu())
        else:
            out.append(revert(model(im)).sigmoid_().cpu())
    if not average:
        return out
    return torch.stack(out).mean(dim=0)


def revert_tta_factory(flip, rot):
    """
    Reverts the transpose and mirror transformation into original state.
    """
    if flip and rot:
        return lambda x: torch.rot90(x.flip(flip), rot, dims=(3, 4))
    elif flip:
        return lambda x: x.flip(flip)
    elif rot:
        return lambda x: torch.rot90(x, rot, dims=(3, 4))
    else:
        raise


def get_datasets(folder: str, patterns:list[str], seed:int, debug:bool, rand_blank=False, no_seg=False, full=False,
                 fold_number=0, normalisation="minmax") -> Union[tuple[Brats, Brats], tuple[Brats, Brats, Brats]]:
    """
    Get the training data set from the given folder. For adaptive training mode the wanted patterns and a randomised
    set of images can be replaced by a blank image.

    *Arguments*:
        folder:         String of base training folder. Herein, subfolders should be named according to BraTS
        patterns:       List of strings, that identify the used input channels. Default is [_t1, _t1ce, _t2, flair]
        seed:           Int, number of seed for pseudo random numbers
        debug:          Bool for debugging
        rand_blank:     Bool, for randomly setting input channels blank
        no_seg:         Bool that differs the set of inference or validation and training.
        full:           Bool, using of the full dataset for training
        fold_number:    Int, for folding of the dataset
        normalisation:  String, default is "minmax", also "z_score" can be used.

    Returns:
        Tuple of training, bench or validation datasets
    """
    base_folder = pathlib.Path(folder).resolve()
    print(base_folder)
    assert base_folder.exists()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    if full:
        train_dataset = Brats(patients_dir, patterns, rand_blank, training=True, debug=debug, normalisation=normalisation)
        bench_dataset = Brats(patients_dir, patterns, rand_blank, training=False, benchmarking=True, debug=debug,
                              normalisation=normalisation)
        return train_dataset, bench_dataset
    if no_seg:
        return Brats(patients_dir, patterns, rand_blank, training=False, debug=debug, no_seg=no_seg, normalisation=normalisation)
    kfold = KFold(5, shuffle=True, random_state=seed)
    splits = list(kfold.split(patients_dir))
    train_idx, val_idx = splits[fold_number]
    print("first idx of train", train_idx[0])
    print("first idx of test", val_idx[0])
    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_idx]
    # return patients_dir
    train_dataset = Brats(train, patterns, rand_blank, training=True,  debug=debug, normalisation=normalisation)
    val_dataset = Brats(val, patterns, rand_blank, training=False, data_aug=False,  debug=debug, normalisation=normalisation)
    bench_dataset = Brats(val, patterns, rand_blank, training=False, benchmarking=True, debug=debug, normalisation=normalisation)
    return train_dataset, val_dataset, bench_dataset


def master_do(func, *args, **kwargs):
    """
    Help calling function only on the rank0 process id ddp
    """
    try:
        rank = dist.get_rank()
        if rank == 0:
            return func(*args, **kwargs)
    except AssertionError:
        # not in DDP setting, just do as usual
        func(*args, **kwargs)


def save_checkpoint(state: dict, save_folder: pathlib.Path):
    """
    Save Training state.
    """
    best_filename = f'{str(save_folder)}/model_best.pth.tar'
    torch.save(state, best_filename)


def reload_ckpt(args, model, optimizer, scheduler):
    """
    Reloads the training onto set checkpoint with the scheduler.
    """
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))


def reload_ckpt_bis(ckpt, model, optimizer=None):
    """
    Reloads the training onto set checkpoint.
    """
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        try:
            ngpus = torch.cuda.device_count()
            if ngpus == 0:
                checkpoint = torch.load(ckpt, map_location="cpu")
            else:
                checkpoint = torch.load(ckpt)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{ckpt}' (epoch {start_epoch})")
            return start_epoch
        except RuntimeError:
            # TO account for checkpoint from Alex nets
            print("Loading model Alex style")
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")


def calculate_metrics(preds, targets, patient, tta=False) -> list:
    """
    Calculates the metrics [Hausdorff, Dice, Sens, Spec] of the prediction.

    *Arguments*:
        preds:        Torch tensor of size 1*C*Z*Y*X
        targets:      Torch tensor of same shape
        patient:      String of the patient ID
        tta:          Bool, is tta performed for this run

    *Return*:
        metrics_list: List of the evaluated metrics of the prediction.
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []

    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )

        if np.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            sens = np.nan
            dice = 1 if np.sum(preds[i]) == 0 else 0
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            spec = tn / (tn + fp)
            haussdorf_dist = np.nan

        else:
            preds_coords = np.argwhere(preds[i])
            targets_coords = np.argwhere(targets[i])
            haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]

            tp = np.sum(l_and(preds[i], targets[i]))
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)

            dice = 2 * tp / (2 * tp + fp + fn)

        metrics[HAUSSDORFF] = haussdorf_dist
        metrics[DICE] = dice
        metrics[SENS] = sens
        metrics[SPEC] = spec
        pp.pprint(metrics)
        metrics_list.append(metrics)

    return metrics_list


def update_teacher_parameters(model, teacher_model, global_step, alpha=0.99 / 0.999) -> None:
    """
    Updates the teacher parameters. Use the true average until the exponential average is more correct.
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, param in zip(teacher_model.parameters(), model.parameters()):
        teacher_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def determinist_collate(batch):
    """
    Pads the batch to its maximum shape and uses the default torch collate command.
    """
    batch = pad_batch_to_max_shape(batch)
    return torch.utils.data._utils.collate.default_collate(batch)


def pad_batch_to_max_shape(batch):
    """
    Pads a batch to the maximum possible shape.
    """
    shapes = (sample['label'].shape for sample in batch)
    _, z_sizes, y_sizes, x_sizes = list(zip(*shapes))
    maxs = [int(max(z_sizes)), int(max(y_sizes)), int(max(x_sizes))]
    for i, max_ in enumerate(maxs):
        max_stride = 16
        if max_ % max_stride != 0:
            # Make it divisible by 16
            maxs[i] = ((max_ // max_stride) + 1) * max_stride
    zmax, ymax, xmax = maxs
    for elem in batch:
        exple = elem['label']
        zpad, ypad, xpad = zmax - exple.shape[1], ymax - exple.shape[2], xmax - exple.shape[3]
        assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
        # free data augmentation
        left_zpad, left_ypad, left_xpad = [randint(0, pad) for pad in (zpad, ypad, xpad)]
        right_zpad, right_ypad, right_xpad = [pad - left_pad for pad, left_pad in zip((zpad, ypad, xpad), (left_zpad, left_ypad, left_xpad))]
        pads = (left_xpad, right_xpad, left_ypad, right_ypad, left_zpad, right_zpad)
        elem['image'], elem['label'] = F.pad(elem['image'], pads), F.pad(elem['label'], pads)
    return batch


def pad_batch1_to_compatible_size(batch):
    """
    Pads a batch to compatible size.
    """
    print(batch.shape)
    shape = batch.shape
    zyx = list(shape[-3:])
    for i, dim in enumerate(zyx):
        max_stride = 16
        if dim % max_stride != 0:
            # Make it divisible by 16
            zyx[i] = ((dim // max_stride) + 1) * max_stride
    zmax, ymax, xmax = zyx
    zpad, ypad, xpad = zmax - batch.size(2), ymax - batch.size(3), xmax - batch.size(4)
    assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
    pads = (0, xpad, 0, ypad, 0, zpad)
    batch = F.pad(batch, pads)
    return batch, (zpad, ypad, xpad)


def pad_single_to_compatible_size(batch):
    """
    Pads a single modality into a compatible size.
    """
    print(batch.shape)
    shape = batch.shape
    zyx = list(shape[-3:])
    for i, dim in enumerate(zyx):
        max_stride = 16
        if dim % max_stride != 0:
            # Make it divisible by 16
            zyx[i] = ((dim // max_stride) + 1) * max_stride
    zmax, ymax, xmax = zyx
    zpad, ypad, xpad = zmax - batch.size(1), ymax - batch.size(2), xmax - batch.size(3)
    assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
    pads = (0, xpad, 0, ypad, 0, zpad)
    batch = F.pad(batch, pads)
    return batch, (zpad, ypad, xpad)


def pad_or_crop_image(image, seg=None, target_size=(128, 144, 144)):
    """
    Pads or crops the image into a target size.
    """
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg
    return image


def get_left_right_idx_should_pad(target_size, dim):
    """
        Returns the right and left indices that should be cropped from the image.
    """
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right


def get_crop_slice(target_size, dim):
    """
    Returns the size of a cropped image.
    """
    if dim > target_size:
        crop_extent = dim - target_size
        left = randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)


def normalize(image):
    """
    Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    if scale != 0.0:
        image = (image - min_) / scale
    return image


def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """
    Main pre-processing function used for the challenge (seems to work the best).

    Remove outliers voxels first, then min-max scale.

    Warnings
    --------
    This will not do it channel wise!!
    """

    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image


def zscore_normalise(img: np.ndarray) -> np.ndarray:
    """
    Normalisation of z_score.
    """
    slices = (img != 0)
    if slices is not None:
        img[slices] = (img[slices] - np.mean(img[slices])) / np.std(img[slices])
    return img


def remove_unwanted_background(image, threshold=1e-5):
    """
    Use to crop zero_value pixel from MRI image.
    """
    dim = len(image.shape)
    non_zero_idx = np.nonzero(image > threshold)
    min_idx = [np.min(idx) for idx in non_zero_idx]
    # +1 because slicing is like range: not inclusive!!
    max_idx = [np.max(idx) + 1 for idx in non_zero_idx]
    bbox = tuple(slice(_min, _max) for _min, _max in zip(min_idx, max_idx))
    return image[bbox]


def random_crop2d(*images, min_perc=0.5, max_perc=1.) -> list:
    """
    Crop randomly but identically all images given. Could be used to pass both mask and image at the same time.
    Anything else will throw.

    Warnings
    --------
    Only works for channel first images. (No channel image will not work).
    """
    if len(set(tuple(image.shape) for image in images)) > 1:
        raise ValueError("Image shapes do not match")
    shape = images[0].shape
    new_sizes = [int(dim * random.uniform(min_perc, max_perc)) for dim in shape]
    min_idx = [random.randint(0, ax_size - size) for ax_size, size in zip(shape, new_sizes)]
    max_idx = [min_id + size for min_id, size in zip(min_idx, new_sizes)]
    bbox = list(slice(min_, max(max_, 1)) for min_, max_ in zip(min_idx, max_idx))
    # DO not crop channel axis...
    bbox[0] = slice(0, shape[0])
    # prevent warning
    bbox = tuple(bbox)
    cropped_images = [image[bbox] for image in images]
    if len(cropped_images) == 1:
        return cropped_images[0]
    else:
        return cropped_images


def random_crop3d(*images, min_perc=0.5, max_perc=1.) -> list:
    """
    Crop randomly but identically all images given. Could be used to pass both mask and image at the same time.
    Anything else will throw.

    Warnings
    --------
    Only works for channel first images. (No channel image will not work).
    """
    return random_crop2d(min_perc, max_perc, *images)


def randomise_blanks(rand:bool, paths:list[str]) -> list[str]:
    """
    Changes randomly in the list of mri modality paths, one or more modality into a blank image for training of adaptive
    model.

    *Arguments*:
        rand:           Bool, true if adaptive training should be used
        paths:          List of strings for paths:

    *Returns*:
        modified_list:  List of strings with randomly modified list of strings
    """
    modified_list = paths.copy()
    if rand:
        num_blanks = randint(0, len(paths) - 1)
        indices = sample(range(len(paths)), num_blanks)
        for index in indices:
            modified_list[index] = TRAINING_NULL_IMAGE
    return modified_list


def count_parameters(model) -> int:
    """
    Count trainable parameters of neural network
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_metrics(epoch:int, metrics:list, swa:bool, writer, current_epoch:int, teacher=False, save_folder=None) -> None:
    """
    Saves the metrics into the respective folder.
    """
    metrics = list(zip(*metrics))
    # print(metrics)
    # TODO check if doing it directly to numpy work
    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    # print(metrics)
    labels = ("ET", "TC", "WT")
    metrics = {key: value for key, value in zip(labels, metrics)}
    # print(metrics)
    fig, ax = plt.subplots()
    ax.set_title("Dice metrics")
    ax.boxplot(metrics.values(), labels=metrics.keys())
    ax.set_ylim(0, 1)
    writer.add_figure(f"val/plot", fig, global_step=epoch)
    print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    with open(f"{save_folder}/val{'_teacher' if teacher else ''}.txt", mode="a") as f:
        print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
              [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()], file=f)
    for key, value in metrics.items():
        tag = f"val{'_teacher' if teacher else ''}{'_swa' if swa else ''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)


def step(data_loader, model, criterion: EDiceLoss, metric, deep_supervision, optimizer, epoch, writer,
         scaler=None, scheduler=None, swa=False, save_folder=None, no_fp16=False, patients_perf=None) -> int:
    """
    Performs a training step epoch.

    *Arguments*:
        data_loader:
        model:
        criterion:
        metric:
        deep_supervision:
        optimizer:
        epoch:
        writer:
        scaler:
        scheduler:
        swa:
        save_folder:
        no_fp16:
        patients_perf:
    """
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mode = "train" if model.training else "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(batch_per_epoch, [batch_time, data_time, losses],
        prefix=str(mode) + "Epoch: [" + str(epoch) + "]")

    end = time.perf_counter()
    metrics = []
    print(f"fp 16: {not no_fp16}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).cuda()
    else:
        data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).cpu()


    for i, batch in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)
        inputs = batch["image"]
        patient_id = batch["patient_id"]
        nan_mask = torch.isnan(inputs)
        if device == "cuda":
            targets = batch["label"].cuda(non_blocking=True)
            inputs = torch.where(nan_mask, torch.tensor(0.0, dtype=torch.float16), inputs).cuda()
            with torch.cuda.amp.autocast(enabled=not no_fp16):
                if mode == "train":
                    inputs = data_aug(inputs)
                if deep_supervision:
                    segs, deeps = model(inputs)
                    if mode == "train":  # revert the data aug
                        segs, deeps = data_aug.reverse([segs, deeps])
                    loss_ = torch.stack([criterion(segs, targets)] + [criterion(deep, targets) for deep in deeps])
                    print(f"main loss: {loss_}")
                    loss_ = torch.mean(loss_)
                else:
                    segs = model(inputs)
                    if mode == "train":
                        segs = data_aug.reverse(segs)
                    loss_ = criterion(segs, targets)
                if patients_perf is not None:
                    patients_perf.append(dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item()))

                writer.add_scalar(f"Loss/{mode}{'_swa' if swa else ''}", loss_.item(),
                                  global_step=batch_per_epoch * epoch + i)

                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses.update(loss_.item())
                else:
                    print("NaN in model loss!!")

                if not model.training:
                    metric_ = metric(segs, targets)
                    metrics.extend(metric_)
        else:
            targets = batch["label"].cpu()
            inputs = torch.where(nan_mask, torch.tensor(0.0, dtype=torch.float16), inputs).cpu()
            with torch.cpu.amp.autocast():
                if mode == "train":
                    inputs = data_aug(inputs)
                if deep_supervision:
                    segs, deeps = model(inputs)
                    if mode == "train":  # revert the data aug
                        segs, deeps = data_aug.reverse([segs, deeps])
                    loss_ = torch.stack([criterion(segs, targets)] + [criterion(deep, targets) for deep in deeps])
                    print(f"main loss: {loss_}")
                    loss_ = torch.mean(loss_)
                else:
                    segs = model(inputs)
                    if mode == "train":
                        segs = data_aug.reverse(segs)
                    loss_ = criterion(segs, targets)
                if patients_perf is not None:
                    patients_perf.append(dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item()))

                writer.add_scalar(f"Loss/{mode}{'_swa' if swa else ''}",loss_.item(),
                                  global_step=batch_per_epoch * epoch + i)

                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses.update(loss_.item())
                else:
                    print("NaN in model loss!!")

                if not model.training:
                    metric_ = metric(segs, targets)
                    metrics.extend(metric_)

        # compute gradient and do SGD step
        if model.training:
            if device == "cuda":
                scaler.scale(loss_).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch * batch_per_epoch + i)
            else:
                loss_.backward()
                optimizer.step()

            optimizer.zero_grad()
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch * batch_per_epoch + i)

        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

    if not model.training:
        save_metrics(epoch, metrics, swa, writer, epoch, False, save_folder)

    if mode == "train":
        writer.add_scalar(f"SummaryLoss/train", losses.avg, epoch)
    else:
        writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    return losses.avg

