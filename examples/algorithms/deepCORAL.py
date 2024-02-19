import torch
from models.initializer import initialize_model
from algorithms.single_model_algorithm import SingleModelAlgorithm
from wilds.common.utils import split_into_groups
from utils import concat_input


class DeepCORAL(SingleModelAlgorithm):
    """
    Deep CORAL.
    This algorithm was originally proposed as an unsupervised domain adaptation algorithm.

    Original paper:
        @inproceedings{sun2016deep,
          title={Deep CORAL: Correlation alignment for deep domain adaptation},
          author={Sun, Baochen and Saenko, Kate},
          booktitle={European Conference on Computer Vision},
          pages={443--450},
          year={2016},
          organization={Springer}
        }

    The original CORAL loss is the distance between second-order statistics (covariances)
    of the source and target features.

    The CORAL penalty function below is adapted from DomainBed's implementation:
    https://github.com/facebookresearch/DomainBed/blob/1a61f7ff44b02776619803a1dd12f952528ca531/domainbed/algorithms.py#L539
    """

    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        # check config
        assert config.train_loader == 'group'
        assert config.uniform_over_groups
        assert config.distinct_groups
        # initialize models
        featurizer, classifier = initialize_model(config, d_out=d_out, is_featurizer=True)
        featurizer = featurizer.to(config.device)
        classifier = classifier.to(config.device)
        model = torch.nn.Sequential(featurizer, classifier)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # penalty_weight超参数
        self.penalty_weight = config.coral_penalty_weight
        # additional logging
        self.logged_fields.append('penalty')
        # set model components
        self.featurizer = featurizer
        self.classifier = classifier

    #     损失函数解释：参考P40 ⭐⭐⭐_22ILCR Extending the WILDS Benchmark for Unsupervised Adaptation.pdf
    # 计算任意两个group之间的coral penalty，参考7. Mean and covars.py文件中的验证方式，与论文中的协方差计算结果相同
    def coral_penalty(self, x, y):
        if x.dim() > 2:
            # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
            # we flatten to Tensors of size (*, feature dimensionality)
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))

        mean_x = x.mean(0, keepdim=True)  # group1的均值:[1,feat_out]
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x  # 归一化,shape:[batch,feat_out]
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)  # group1的协方差矩阵shape:[feat_out,feat_out]
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()  # 均值差转化为标量
        cova_diff = (cova_x - cova_y).pow(2).mean()  # 协方差转化为标量

        return mean_diff + cova_diff

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - features (Tensor): featurizer output for batch and unlabeled batch
                - y_pred (Tensor): full model output for batch and unlabeled batch
        """
        # 真实标签
        x, y_true, metadata = batch
        y_true = y_true.to(self.device)  # [batch,]
        g = self.grouper.metadata_to_group(metadata).to(self.device)  # 获取对应的group：[batch,]

        results = {
            'g': g,
            'y_true': y_true,
            'metadata': metadata,
        }

        if unlabeled_batch is not None:
            unlabeled_x, unlabeled_metadata = unlabeled_batch
            x = concat_input(x, unlabeled_x)
            unlabeled_g = self.grouper.metadata_to_group(unlabeled_metadata).to(self.device)
            results['unlabeled_g'] = unlabeled_g
        # 前向过程，计算特征和预测值
        x = x.to(self.device)
        features = self.featurizer(x)
        outputs = self.classifier(features)
        y_pred = outputs[: len(y_true)]

        results['features'] = features
        results['y_pred'] = y_pred
        return results

    # 计算CORAL的损失函数
    def objective(self, results):
        if self.is_training:
            features = results.pop('features')

            # Split into groups
            groups = concat_input(results['g'], results['unlabeled_g']) if 'unlabeled_g' in results else results['g']
            unique_groups, group_indices, _ = split_into_groups(groups)
            n_groups_per_batch = unique_groups.numel()

            # 计算两两group之间的coral penalty
            penalty = torch.zeros(1, device=self.device)
            for i_group in range(n_groups_per_batch):
                for j_group in range(i_group + 1, n_groups_per_batch):
                    penalty += self.coral_penalty(features[group_indices[i_group]], features[group_indices[j_group]])
            if n_groups_per_batch > 1:
                penalty /= (n_groups_per_batch * (n_groups_per_batch - 1) / 2)  # get the mean penalty
        else:
            penalty = 0.

        self.save_metric_for_logging(results, 'penalty', penalty)
        avg_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        return avg_loss + penalty * self.penalty_weight
