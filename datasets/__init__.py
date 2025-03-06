import logging
from typing import Callable
from datasets.celeba.celeba import get_celeba_dataset
from datasets.funnybirds.funnybirds import get_funnybirds
from datasets.funnybirds.funnybirds_attributes import get_funnybirds_attributes

logger = logging.getLogger(__name__)
DATASETS = {
    "funnybirds_forced_concept": get_funnybirds,
    "funnybirds": get_funnybirds,
    "funnybirds_attributes": get_funnybirds_attributes,
    "celeba": get_celeba_dataset
}


def get_dataset(dataset_name: str) -> Callable:
    """
    Get dataset by name.
    :param dataset_name: Name of the dataset.
    :return: Dataset.

    """
    if dataset_name in DATASETS:
        dataset = DATASETS[dataset_name]
        logger.info(f"Loading {dataset_name}")
        return dataset
    else:
        raise KeyError(f"DATASET {dataset_name} not defined.")
    
def get_dataset_kwargs(config):
    dataset_specific_kwargs = {
        "label_map_path": config["label_map_path"],
        "classes": config.get("classes", None),
        "train": True
    } if "imagenet" in config['dataset_name'] else {}

    return dataset_specific_kwargs
