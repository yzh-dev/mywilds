import torch
import torch.nn as nn
import torch.nn.functional as F
from models.initializer import initialize_model
from algorithms.ERM import ERM
from algorithms.single_model_algorithm import SingleModelAlgorithm
from wilds.common.utils import split_into_groups
from configs.supported import process_outputs_functions
import copy
from utils import load
import re

class DropoutModel(nn.Module):
    def __init__(self, featurizer, classifier, dropout_rate):
        super().__init__()
        self.featurizer = featurizer
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = classifier
    def forward(self, x):
        features = self.featurizer(x)
        features_sparse = self.dropout(features)
        return self.classifier(features_sparse)

class NoisyStudent(SingleModelAlgorithm):
    """
    Noisy Student.
    This algorithm was originally proposed as a semi-supervised learning algorithm.

    One run of this codebase gives us one iteration (load a teacher, train student). To run another iteration,
    re-run the previous command, pointing config.teacher_model_path to the trained student weights.

    To warm start the student model, point config.pretrained_model_path to config.teacher_model_path

    Based on the original paper, loss is of the form
        \ell_s + \ell_u
    where 
        \ell_s = cross-entropy with true labels; student predicts with noise
        \ell_u = cross-entropy with pseudolabel generated without noise; student predicts with noise
    The student is noised using:
        - Input images are augmented using RandAugment
        - Single dropout layer before final classifier (fc) layer
        - TODO: stochastic depth with linearly decaying survival probability from last to first

    This code only supports hard pseudolabeling and a teacher that is the same class as the student (e.g. both densenet121s)

    Original paper:
        @inproceedings{xie2020self,
            title={Self-training with noisy student improves imagenet classification},
            author={Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V},
            booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
            pages={10687--10698},
            year={2020}
            }
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        # check that we had a teacher model (and thus computed pseudolabels in run_expt.py)
        assert config.teacher_model_path is not None
        # initialize student model with dropout before last layer
        featurizer, classifier = initialize_model(config, d_out=d_out, is_featurizer=True)
        student_model = DropoutModel(featurizer, classifier, config.dropout_rate).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=student_model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # algorithm hyperparameters
        # auxiliary information
        *_, last_layer = featurizer.named_children()
        self.last_layer_name = last_layer[0]

    def state_dict(self):
        """
        Overrides function called when saving the model. We want to be able to directly load the saved student into the teacher,
        so we need to reformat the state dict to match the teacher's state dict.
        """
        def omit(k):
            return k.startswith('featurizer') or k.startswith(self.last_layer_name)
        def fmt(k):
            return re.sub('featurizer.', '', k)            
        state = super().state_dict()
        state = { fmt(k):v for k,v in state.items() if not omit(k) }
        return state
        
    def process_batch(self, labeled_batch, unlabeled_batch=None):
        # Labeled examples
        x, y_true, metadata = labeled_batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        outputs = self.model(x)
        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata
        }
        # Unlabeled examples
        if unlabeled_batch is not None:
            x, y_pseudo, metadata = unlabeled_batch # x should be strongly augmented
            x = x.to(self.device)
            g = self.grouper.metadata_to_group(metadata).to(self.device)
            y_pseudo = y_pseudo.to(self.device)
            outputs = self.model(x)
            results['unlabeled_metadata'] = metadata
            results['unlabeled_y_pseudo'] = y_pseudo 
            results['unlabeled_y_pred'] = outputs
            results['unlabeled_g'] = g
        return results

    def objective(self, results):
        labeled_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        if 'unlabeled_y_pred' in results: 
            unlabeled_loss = self.loss.compute(results['unlabeled_y_pred'], results['unlabeled_y_pseudo'], return_dict=False)
        else: unlabeled_loss = 0
        return labeled_loss + unlabeled_loss 
