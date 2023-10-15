import glob
import os
import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import DatasetBase, Datum


@DATASET_REGISTRY.register()
class TerraIncognitaDG(DatasetBase):
    """A TerraIncognita (Beery et al., 2018) subset.

    Statistics:
        - Around 24,330 images.
        - 10 classes related to wild animals shot by camera traps at different locations.
        - 4 domains: location_38, location_43, location_46, location_100.
        - URL: https://beerys.github.io/CaltechCameraTraps.

    Reference:
        - Beery et al. Recognition in Terra Incognita. ECCV 2018.
    """
    dataset_dir = 'terra_incognita'
    domains = ['location_38', 'location_43', 'location_46', 'location_100']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = TerraIncognitaDG.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS
        )
        val = TerraIncognitaDG.read_data(
            self.dataset_dir, cfg.DATASET.TARGET_DOMAINS
        )
        test = TerraIncognitaDG.read_data(
            self.dataset_dir, cfg.DATASET.TARGET_DOMAINS
        )

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_data(dataset_dir, input_domains):
        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []
            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, '*.jpg'))
                for impath in impaths:
                    items_.append((impath, label))
            return items_
        items = []

        for domain, dname in enumerate(input_domains):
            split_dir = osp.join(dataset_dir, dname)
            impath_label_list = _load_data_from_directory(split_dir)
            for impath, label in impath_label_list:
                class_name = impath.split(os.sep)[-2].lower()
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=class_name
                )
                items.append(item)
        return items