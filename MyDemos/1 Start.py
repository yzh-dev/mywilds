from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.data_loaders import get_eval_loader

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="iwildcam", download=True, root_dir='D:\ML\Dataset\iwildcamDataset',
                      split_scheme='official')
# %%
# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)

# Prepare the standard data loader
# %%
# For training, the WILDS package provides two types of data loaders："standard" 和 "group"
# The standard data loader shuffles examples in the training set, and is used for the standard approach of empirical risk minimization (ERM), where we minimize the average loss.
train_loader = get_train_loader("standard", train_data, batch_size=16)
for labeled_batch in train_loader:
    # The metadata contains information like the domain identity, e.g., which camera a photo was taken from
    x, y, metadata = labeled_batch
    break

# %%
# To allow algorithms to leverage domain annotations as well as other groupings over the available metadata, the WILDS package provides Grouper objects. These Grouper objects are helper objects that extract group annotations from metadata, allowing users to specify the grouping scheme in a flexible fashion. They are used to initialize group-aware data loaders (as discussed in #Data loading) and to implement algorithms that rely on domain annotations (e.g., Group DRO). In the following code snippet, we initialize and use a Grouper that extracts the domain annotations on the iWildCam dataset, where the domain is location.
# Initialize grouper, which extracts domain information, In this example, we form domains based on location
grouper = CombinatorialGrouper(dataset, ['location'])

# To support other algorithms that rely on specific data loading schemes, we also provide the group data loader.
# In each minibatch, the group loader first samples a specified number of groups, and then samples a fixed number of examples from each of those groups.
# This is useful for training with a fixed number of examples per domain.
# In this example, we sample 2 domains per batch, and 8 examples per domain
# 在每个域中采样相同数量的样本，可以通过参数uniform_over_groups控制
train_loader = get_train_loader("group", train_data, grouper=grouper, n_groups_per_batch=2, batch_size=16)
for labeled_batch in train_loader:
    # The metadata contains information like the domain identity, e.g., which camera a photo was taken from
    x, y, metadata = labeled_batch  # metadata保存了图片拍摄时间、地点等原始信息
    z = grouper.metadata_to_group(metadata)  # z is a tensor of domain identities，z是一个域标识的张量
    break
# --------------------------------------------------------------------------------------------------------------
# %%


# --------------------------------------------------------------------------------------------------------------
# %%
# we also provide a data loader for evaluation, which loads examples without shuffling (unlike the training loaders)
test_data = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
)

# Prepare the evaluation data loader
test_loader = get_eval_loader("standard", test_data, batch_size=16)
for test_batch in test_loader:
    x, y, metadata = test_batch
    break

# (Optional) Load unlabeled data
# dataset = get_dataset(dataset="iwildcam", download=True, unlabeled=True, root_dir='D:\ML\Dataset\wildcamDataset', split_scheme='official')
# unlabeled_data = dataset.get_subset(
#     "test_unlabeled",
#     transform=transforms.Compose(
#         [transforms.Resize((448, 448)), transforms.ToTensor()]
#     ),
# )
# unlabeled_loader = get_train_loader("standard", unlabeled_data, batch_size=16)
# for unlabeled_batch in unlabeled_loader:
#     x, metadata = unlabeled_batch
#     break
