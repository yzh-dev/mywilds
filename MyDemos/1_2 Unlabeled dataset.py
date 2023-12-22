from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.data_loaders import get_eval_loader

#%%
# (Optional) Load unlabeled data
dataset = get_dataset(dataset="iwildcam", download=True, unlabeled=True, root_dir='D:\ML\Dataset\iwildcamDataset')
unlabeled_data = dataset.get_subset(
    # "test_unlabeled",
    "extra_unlabeled",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)
unlabeled_loader = get_train_loader("standard", unlabeled_data, batch_size=16)
for unlabeled_batch in unlabeled_loader:
    x, metadata = unlabeled_batch
    break
