import itertools
from copy import deepcopy
from typing import List
import warnings

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from .shapes import square, circle, triangle, cross, plus
from .colors import color_adjustment, COLORS
from .textures import (
    solid,
    random_spots,
    regular_spots,
    polka,
    chequerboard,
    striped_diagonal,
    striped_vertical,
    striped_horizontal,
    striped_diagonal_alt,
)

SEED_MAX = 1000000

SHAPES = {
    "square": square,
    "circle": circle,
    "triangle": triangle,
    "cross": cross,
    "plus": plus,
}
TEXTURES = {
    "solid" : {
        "solid_plain": solid,
    },
    "spots": {
        "random": random_spots,
        "regular": regular_spots,
        "polka": polka,
        "chequerboard": chequerboard,
    },
    "stripes": {
        "horizontal": striped_horizontal,
        "vertical": striped_vertical,
        "diagonal": striped_diagonal,
        "diagonal_alt": striped_diagonal_alt,
    },
}

location_keywords = {
    "left": ["<127", ">0"],
    "right": [">127", ">0"],
    "bot": [">0", ">127"],
    "top": [">0", "<127"],
}


class Element(Dataset):
    def __init__(self, size, shape, color, texture=None, color_seed=None, texture_seed=None):
        self.size = size
        self.shape = shape
        self.color = color
        self.texture = texture
        self.color_seed = color_seed
        self.texture_seed = texture_seed

        self.config = {k: v for k, v in vars(self).items()}

        self.img = SHAPES[shape](size)

        # Set to True to ignore this element when determining class
        self.ignore = False

        # Apply texture
        if texture == "solid":
            texture = None
        if texture is not None:
            texture_parts = texture.split("_")
            texture_fn = TEXTURES[texture_parts[0]]["_".join(texture_parts[1:])]
            self.img = texture_fn(self.img, seed=texture_seed)

        # Add colour
        self.color_value = color_adjustment(color_seed)
        self.img = (self.img.reshape((size, size, 1)) * (self.color_value * COLORS[color])).astype(np.uint8)

        # Set all black pixels to white
        self.img[(self.img == [0, 0, 0]).all(axis=2)] = [255, 255, 255]

    def __len__(self):
        return len(self.img)

    def __repr__(self):
        return "Element <" + ", ".join([f"{k}: {v}" for k, v in self.config.items()]) + ">"

    def belongs_to_class(self, class_config):
        # This flag means we ignore the element when determining class
        # Currently used if the element was not placed in the image
        if self.ignore:
            return False
        belongs = True
        for k, v in class_config.items():
            if v is None:
                continue
            else:
                if self.config[k] != v:
                    belongs = False
                    break
        return belongs


class ElementImage:
    def __init__(self, elements: List[Element], size: int = 224, loc_seed: int = None, loc_restrictions=None, place_remaining_randomly=True):
        self.elements = elements
        self.size = size
        self.loc_seed = loc_seed
        self.class_labels = None
        self.class_labels_oh = None
        self.loc_restrictions = loc_restrictions
        self.place_remaining_randomly = place_remaining_randomly

        self.img = np.zeros((size, size, 3), dtype=np.uint8) + 255
        self.locs = self.choose_locations(size, elements, loc_seed, loc_restrictions=loc_restrictions, place_remaining_randomly=place_remaining_randomly)
        if not place_remaining_randomly:
            # Ignore elements that could not be placed. Important for class labelling
            for i in range(len(self.elements)):
                if self.locs[i] is None:
                    elements[i].ignore = True

        self.config = {
            "elements": [v.config for v in self.elements],
            "size": self.size,
            "loc_seed": self.loc_seed,
            "loc_restrictions": self.loc_restrictions
        }
        self.info = {
            **self.config,
            "locs": self.locs,
            "class_labels": self.class_labels,
            "class_labels_oh": self.class_labels_oh,
        }

        for i in range(len(elements)):
            self.img = self.place_element(self.img, self.elements[i], self.locs[i])

    @staticmethod
    def place_element(canvas, element, loc):
        if loc is not None:
            canvas[loc[0]:loc[0] + len(element), loc[1]:loc[1] + len(element)] = element.img
        return canvas

    def __len__(self):
        return len(self.img)

    def __repr__(self):
        return "ElementImage <" + ", ".join([f"{k}: {v}" for k, v in self.config.items()]) + ">"

    def belongs_to_class(self, class_config):
        for i, element in enumerate(self.elements):
            config_without_loc = {k: v for k, v in class_config.items() if k != "loc"}
            if element.belongs_to_class(config_without_loc):
                if "loc" in class_config.keys():
                    restriction = location_keywords[class_config["loc"]]
                    loc = self.locs[i]
                    failed_restriction = self.check_fail_loc_restriction(
                        restriction, loc[0], loc[1], element.size
                    )
                    if not failed_restriction:
                        return True
                else:
                    return True
        return False

    def update_class_labels(self, class_configs):
        classes = []
        for i, config in enumerate(class_configs):
            if self.belongs_to_class(config):
                classes.append(i)
        self.class_labels = classes
        self.class_labels_oh = np.eye(len(class_configs))[classes].sum(axis=0)
        self.info["class_labels"] = self.class_labels
        self.info["class_labels_oh"] = self.class_labels_oh
        return classes

    @staticmethod
    def from_config(config):
        # Only select relevant keys
        # Allows self.info to be fed in as well as self.config
        config = {k: config[k] for k in ["elements", "size", "loc_seed"]}
        elements = [Element(**v) for v in config["elements"]]
        config["elements"] = elements
        return ElementImage(**config)

    @staticmethod
    def choose_locations(canvas_size, elements, seed=None, max_attempts=100, raise_error=False, loc_restrictions=None, place_remaining_randomly=True):
        """
        x and y are actually the wrong way round.
        But they're just variable names so doesn't matter.
        """
        rng = np.random.default_rng(seed)
        element_sizes = [len(element) for element in elements]
        locations = []
        for element_size in element_sizes:
            placed = False
            attempts = 0
            while (not placed) and (attempts < max_attempts):
                x, y = rng.integers(0, canvas_size - element_size, 2)
                attempts += 1

                overlap = False
                for i, coord in enumerate(locations):
                    if coord is not None:
                        overlap = check_overlap(x, y, element_size, element_sizes[i], coord)
                    if overlap:
                        break
                if overlap:
                    # If it overlaps, try a different position
                    continue

                if loc_restrictions is not None:
                    failed_restriction = False
                    for loc_restriction in loc_restrictions:
                        failed_restriction = ElementImage.check_fail_loc_restriction(loc_restriction, x, y, element_size)
                        if failed_restriction:
                            break
                    if failed_restriction:
                        # If it fails a location restriction, try a different position
                        continue

                locations.append((x, y))
                placed = True
            if not placed:
                if raise_error:
                    raise ValueError("Not all elements could be placed without overlapping!")
                elif place_remaining_randomly:
                    remaining_elements = len(element_sizes) - len(locations)
                    warnings.warn(
                        f"Not all elements could be placed without overlapping! Placing remaining {remaining_elements} elements randomly."
                    )
                    for i in range(remaining_elements):
                        x, y = rng.integers(0, canvas_size - element_sizes[len(locations)], 2)
                        locations.append((x, y))
                    break
                else:
                    remaining_elements = len(element_sizes) - len(locations)
                    warnings.warn(
                        f"Not all elements could be placed without overlapping! Missing {remaining_elements} elements."
                    )
                    for i in range(remaining_elements):
                        locations.append(None)
                    break
        return locations

    @staticmethod
    def check_fail_loc_restriction(loc_restriction, x, y, element_size):
        """
        Returns True if the location/size of the element does not pass the restriction
        """
        x_passes = check_restriction(loc_restriction[0], y, element_size)
        y_passes = check_restriction(loc_restriction[1], x, element_size)
        if (not x_passes) or (not y_passes):
            return True
        else:
            return False


def check_restriction(restriction, v, element_size):
    command = restriction[0]
    value = int(restriction[1:])
    if command == ">":
        if v > value:
            return True
    elif command == "<":
        # Because we place by the top_left corner
        # we need to shift the placement for left/top placements
        if v + element_size < value:
            return True
    else:
        raise ValueError(f"Command {command} not recognised!")


def check_overlap(x, y, element_size, element_size_2, coord):
    if (
        (x < coord[0] + element_size_2)
        and (x + element_size > coord[0])
        and (y < coord[1] + element_size_2)
        and (y + element_size > coord[1])
    ):
        return True
    else:
        return False


class ElementDataset:
    def __init__(
        self,
        allowed,
        class_configs,
        n,
        img_size,
        element_n,
        element_size,
        element_size_delta,
        element_seed,
        loc_seed,
        allowed_combinations=None,
        loc_restrictions=None,
        place_remaining_randomly=True,
    ):
        self.allowed = allowed
        self.class_configs = class_configs
        self.n = n
        self.img_size = img_size
        self.element_n = element_n
        self.element_size = element_size
        self.element_size_delta = element_size_delta
        self.element_seed = element_seed
        self.loc_seed = loc_seed
        self.allowed_combinations = allowed_combinations
        if self.allowed_combinations is not None:
            if type(self.allowed_combinations[0]) is list:
                self.allowed_combinations = [tuple(v) for v in self.allowed_combinations]
        self.loc_restrictions = loc_restrictions
        self.place_remaining_randomly = place_remaining_randomly

        self.config = {k: v for k, v in vars(self).items()}
        self.allowed["sizes"] = (self.element_size - self.element_size_delta,
                                 self.element_size + self.element_size_delta)

        self.element_rng = np.random.default_rng(self.element_seed)
        self.loc_rng = np.random.default_rng(self.loc_seed)

        self.element_seeds = self.element_rng.integers(0, SEED_MAX, self.n)
        self.loc_seeds = self.loc_rng.integers(0, SEED_MAX, self.n)

        self.transform = T.Compose([
            T.ToTensor(),
            ])

    def __getitem__(self, idx):
        item = self.get_item(idx)
        img = self.transform(item.img)
        return [img, item.class_labels_oh]

    def get_item(self, idx):
        element_configs = self.choose_element_configs(
            self.element_n,
            self.allowed["sizes"],
            self.allowed["shapes"],
            self.allowed["colors"],
            self.allowed["textures"],
            self.element_seeds[idx],
            allowed_combinations=self.allowed_combinations
        )
        elements = [Element(**v) for v in element_configs]
        img = ElementImage(elements, self.img_size, self.loc_seeds[idx],
                           loc_restrictions=self.loc_restrictions,
                           place_remaining_randomly=self.place_remaining_randomly)
        img.update_class_labels(self.class_configs)
        return img

    def __len__(self):
        return self.n

    @staticmethod
    def choose_element_configs(n, allowed_sizes, allowed_shapes, allowed_colors, allowed_textures, seed=None, allowed_combinations=None, max_iter=1000):
        rng = np.random.default_rng(seed)

        if allowed_combinations is None:
            allowed_combinations = list(itertools.product(allowed_shapes, allowed_colors, allowed_textures))

        configs = []
        i = 0
        while len(configs) < n:
            i += 1
            config = dict(
                size=rng.integers(*allowed_sizes),
                shape=rng.choice(allowed_shapes),
                color=rng.choice(allowed_colors),
                color_seed=rng.integers(SEED_MAX),
                texture=rng.choice(allowed_textures),
                texture_seed=rng.integers(SEED_MAX),
            )
            if (config["shape"], config["color"], config["texture"]) in allowed_combinations:
                configs.append(config)
            if i > max_iter:
                raise TimeoutError(f"Only {len(configs)} out of {n} chosen "
                                   f"in the max no. iterations ({max_iter})")
        return configs

    def save_imgs(self, prefix, indices=None, extension=".png"):
        if indices is None:
            indices = range(self.n)

        for i in indices:
            img = self.get_item(i)
            img = Image.fromarray(img.img)
            img.save(f"{prefix}{i:04d}{extension}")


class GroupedElementDataset(ElementDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_label = None
        all_class_labels = np.zeros((self.n, len(self.class_configs)))
        for i in range(len(self)):
            img = self.get_item(i)
            all_class_labels[i] = img.class_labels_oh

        self.all_class_labels = all_class_labels
        # Create a list of img indices which belong to each class
        self.idx_array = [np.where(all_class_labels[:, i])[0] for i in range(all_class_labels.shape[1])]

    def __getitem__(self, idx):
        item = self.get_item(self.idx_array[self.current_label][idx])
        img = self.transform(item.img)
        return [img, item.class_labels_oh]

    def __len__(self):
        if self.current_label is None:
            return self.n
        else:
            return len(self.idx_array[self.current_label])


class ConceptElementDatasetCreator:
    """On call, returns an ElementDataset (or child of).
     The dataset is either made up of one of these three things:
     - a single class
     - a single concept
     - a random selection of images
     For use with CAVs."""
    def __init__(self, allowed, class_configs, dataset_kwargs, allowed_combinations=None):
        self.allowed = allowed
        self.class_configs = class_configs
        self.allowed_combinations = allowed_combinations
        self.dataset_kwargs = dataset_kwargs
        self.n_classes = len(class_configs)

        self.class_dataset = GroupedElementDataset(allowed=allowed, class_configs=class_configs, allowed_combinations=self.allowed_combinations, **self.dataset_kwargs)

        self.base_element_seed = 2023
        self.base_loc_seed = 1997
        self.seed_max = 1000000

    def __call__(self, concept):
        concept, loc_restriction = self.check_for_location_keywords(concept)
        try:
            concept = int(concept)
            if loc_restriction is None:
                self.class_dataset.current_label = concept
                return self.class_dataset
            else:
                # If applying a location restriction we need to recreate the class
                # dataset so that the images are labelled correctly.
                # Expensive as we have to loop through all the images again.
                class_dataset = GroupedElementDataset(
                    allowed=self.allowed,
                    class_configs=self.class_configs,
                    allowed_combinations=self.allowed_combinations,
                    loc_restrictions=loc_restriction,
                    place_remaining_randomly=False,
                    **self.dataset_kwargs
                )
                class_dataset.current_label = concept
                return class_dataset

        except ValueError:
            if "random" in concept:
                random_idx = int(concept.split("_")[-1])
                return self.create_random_dataset(random_idx, loc_restriction)
            else:
                return self.create_concept_dataset(concept, loc_restriction)

    def create_random_dataset(self, idx, loc_restriction=None):
        # If there is a location restriction, don't place all elements if they don't fit
        if loc_restriction is None:
            place_remaining_randomly = True
        else:
            place_remaining_randomly = False

        # Probably more complicated than it needs to be, but this makes it very unlikely
        # that we'll have two random datasets with the same random seed or that they're
        # the same as in the train or test sets of the model itself
        element_rng = np.random.default_rng(self.base_element_seed + idx)
        loc_rng = np.random.default_rng(self.base_loc_seed + idx)
        element_seed = element_rng.integers(self.seed_max)
        loc_seed = loc_rng.integers(self.seed_max)

        kwargs = deepcopy(self.dataset_kwargs)
        kwargs["element_seed"] = element_seed
        kwargs["loc_seed"] = loc_seed
        dataset = ElementDataset(
            self.allowed,
            self.class_configs,
            allowed_combinations=self.allowed_combinations,
            loc_restrictions=loc_restriction,
            place_remaining_randomly=place_remaining_randomly,
            **kwargs
        )
        return dataset

    def create_concept_dataset(self, concept, loc_restriction=None):

        # If there is a location restriction, don't place all elements if they don't fit
        if loc_restriction is None:
            place_remaining_randomly = True
        else:
            place_remaining_randomly = False

        concept_type = None
        for k, v in self.allowed.items():
            if concept in v:
                concept_type = k
                break
        if concept_type is None:
            raise ValueError(f"Concept {concept} not found in allowed options!")

        allowed_concept = deepcopy(self.allowed)
        allowed_concept[concept_type] = [concept]

        dataset = ElementDataset(
            allowed_concept,
            self.class_configs,
            allowed_combinations=self.allowed_combinations,
            loc_restrictions=loc_restriction,
            place_remaining_randomly=place_remaining_randomly,
            **self.dataset_kwargs
        )
        return dataset

    def check_for_location_keywords(self, concept):
        """
        Check if the concept has a location restriction applied to it

        :param concept: Name of the concept with optional appended location restriction (str)
        :return: A tuple of Concept name, and location restriction  (str, Union(None, List[List[str]]))
        """
        # location has to be appended to the concept name with _
        potential_location_keyword = concept.split("_")[-1]
        if potential_location_keyword in location_keywords.keys():
            location_restriction = location_keywords[potential_location_keyword]
            concept = "_".join(concept.split("_")[:-1])
            return concept, [location_restriction]
        else:
            # If not recognised, assume no location restrictions
            return concept, None

