import anndata as ad
import torch
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class SLEDataLoader(Dataset):
    def __init__(self, split="train"):
        """
        Initialize data loader

        :param split: "train", "dev" or "test"
        """
        adata = ad.read_h5ad('data/normct_all.h5ad')
        mask_select = (adata.obs['split'] == split)
        adata = adata[mask_select]

        # load self attributes
        self.adata = adata
        self.labels = adata.obs['sleflare'].reset_index(drop=True)
        self.vals = adata.X.toarray().reshape(adata.shape[0], 8, 18190)
        self.nsample = self.vals.shape[0]

    def __len__(self):
        return self.nsample

    def _get_item(self, index):
        """
        Return a single sample and its class label

        :param index: numerical sample index
        :return: a tuple of sample data and sample label
        """
        point_set = self.vals[index, :, :]
        cls = self.labels[index]
        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

def build_dataloader(split="train"):
    """
    Feed data into the pytorch framework

    :param split: "train", "dev" or "test"
    :return: pytorch dataloader object
    """
    dataset = SLEDataLoader(split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=(split == 'train'),
                                             num_workers=0, pin_memory=True)
    return dataloader


if __name__ == '__main__':
    from tqdm import tqdm
    data = SLEDataLoader('dev')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=True, num_workers=0, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for epoch in range(3):
        for point, label in tqdm(DataLoader):
            # point = point.numpy()
            point = point.to(device)
            label = label.to(device)
