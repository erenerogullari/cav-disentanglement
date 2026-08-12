import math
from datetime import datetime
from datetime import timedelta
import cv2

import numpy as np
import torch
import torchvision.transforms as T
from PIL import ImageDraw, Image, ImageEnhance


def get_artifact_kwargs(config):
    artifact_kwargs = {}
    artifact_type = config.get("artifact_type", None)
    if artifact_type == "channel":
        artifact_kwargs = {
            'op_type': config.get('op_type', 'add'),
            'channel': config.get('channel', 0),
            'value': config.get('value', 100)
        }
    elif artifact_type == "lsb":
        artifact_kwargs = {
            'lsb_trigger': config.get('lsb_trigger', "ThisIsASecretCleverHansTrigger"),
            'lsb_factor': config.get("lsb_factor", 3),
            'start_bit': config.get("start_bit", 0)
        }
    elif artifact_type == "random_mnist":
        artifact_kwargs = {
            'shift_factor': config['shift_factor'],
            'datapath_mnist': config['datapath_mnist']
        }
    elif artifact_type == "ch_time":
        artifact_kwargs = {
            "time_format": config.get("time_format", "time")
        }
    elif artifact_type == "color_mnist":
        artifact_kwargs = {
            "color_id": config["attacked_classes"][0]
        }
    elif artifact_type == "white_color":
        artifact_kwargs = {
            "alpha": config.get("alpha", .3)
        }
    elif artifact_type == "brightness":
        artifact_kwargs = {
            "factor": config.get("brightness_factor", config.get("factor", 1.4))
        }
    elif artifact_type == "defective_lead":
        artifact_kwargs = {
            "lead_ids": config.get("lead_ids", [1]),
            "seq_length": config.get("seq_length", 100)
        }
    elif artifact_type == "amplified_beat":
        artifact_kwargs = {
            "lead_ids": config.get("lead_ids", [3]),
            "peak_ids": config.get("peak_ids", [2]),
            "amplify_factor": config.get("amplify_factor", 3)
        }
    if "entanglement_factor" in config.keys():
        artifact_kwargs["entanglement_factor"] = config["entanglement_factor"]
    return artifact_kwargs


def insert_artifact(img, artifact_type, **kwargs):
    if artifact_type == "ch_time":
        return insert_artifact_ch_time(img, **kwargs)
    elif artifact_type == "ch_text":
        return insert_artifact_ch_text(img, **kwargs)
    elif artifact_type == "microscope":
        return insert_microscope(img, **kwargs)
    elif artifact_type == "channel":
        return insert_artifact_channel(img, **kwargs)
    elif artifact_type == "white_color":
        return insert_artifact_white_color(img, **kwargs)
    elif artifact_type == "brightness":
        return insert_artifact_brightness(img, **kwargs)
    elif artifact_type == "red_color":
        return insert_artifact_red_color(img, **kwargs)
    elif artifact_type == "lsb":
        return insert_lsb_trigger(img, **kwargs)
    elif artifact_type == "constant_box":
        return insert_constant_box(img, **kwargs)
    elif artifact_type == "random_box":
        return insert_random_box(img, **kwargs)
    elif artifact_type == "checkerboard":
        return insert_checkerboard(img, **kwargs)
    elif artifact_type == "watermark":
        return insert_watermark(img, **kwargs)
    elif artifact_type == "random_mnist":
        return insert_random_mnist(img, **kwargs)
    elif artifact_type == "color_mnist":
        return color_digit(img, **kwargs)
    else:
        raise ValueError(f"Unknown artifact_type: {artifact_type}")

def color_digit(img, **kwargs):
    assert "color_id" in kwargs
    color_id = kwargs["color_id"]
    COLOR_MAP = [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 128, 255),
        (0, 0, 255),
        (127, 0, 255),
        (255, 0, 255),
        (255, 0, 127)
    ]

    color = COLOR_MAP[color_id]
    img_np = np.array(img)
    img_corrupted = (img_np * color / 255).round().astype(np.uint8)
    mask = img_np[0].round().squeeze()
    return Image.fromarray(img_corrupted), mask

def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = np.random.randint(int_delta)
    return start + timedelta(seconds=random_second)

def insert_artifact_ch_time(img, **kwargs):
    time_format = kwargs.get("time_format", "datetime")
    time_only = time_format == "time"
    d1 = datetime.strptime('01/01/2020', '%m/%d/%Y')
    d2 = datetime.strptime('12/31/2022', '%m/%d/%Y')
    kwargs["reserved_length"] = 60 if time_only else 100
    date = random_date(d1, d2)
    if time_only:
        kwargs["min_val"] = 125
        kwargs["max_val"] = 0
        date = date.strftime("%H:%M:%S")
    kwargs["text"] = str(date)
    color = (
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255), 
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255), 
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255)
    )
    kwargs["color"] = color

    return insert_artifact_ch_text(img, **kwargs)

def insert_random_mnist(img, **kwargs):
    shift_factor = kwargs['shift_factor']
    data_mnist = kwargs['data_mnist']
    random_idx = np.random.choice(len(data_mnist))
    
    img = np.array(img)
    transforms = T.Compose([T.Resize(img.shape[1]), T.ToTensor()])
    img_mnist = transforms(data_mnist[random_idx][0])

    img = img + shift_factor * (np.moveaxis(img_mnist.numpy(), 0, 2) * 255.)
    img = np.clip(img, 0, 255.).astype(np.uint8)
    mask = img_mnist.round().squeeze()
    return Image.fromarray(img), mask


def insert_constant_box(img, **kwargs):
    img = np.array(img)
    size = kwargs.get('size', 2)
    offset = kwargs.get('offset', 1)
    img[offset:offset + size, offset:offset + size, :] = 255
    mask = torch.zeros(img.shape[:2])
    mask[offset:offset + size, offset:offset + size] = 1

    return Image.fromarray(img), mask


def _sample_patch_position(shape, height, width, occupied_mask=None, max_attempts=64):
    max_x = shape[0] - height - 1
    max_y = shape[1] - width - 1
    if max_x <= 1 or max_y <= 1:
        raise ValueError(
            f"Artifact of size {height}x{width} does not fit image shape {shape[:2]}."
        )

    occupied = None
    if occupied_mask is not None:
        occupied = torch.as_tensor(occupied_mask).bool()
        if tuple(occupied.shape) != tuple(shape[:2]):
            raise ValueError(
                "occupied_mask must match the image shape, got "
                f"{tuple(occupied.shape)} and {tuple(shape[:2])}."
            )

    for _ in range(max_attempts):
        posx = int(np.random.randint(1, max_x))
        posy = int(np.random.randint(1, max_y))
        if occupied is None or not occupied[
            posx : posx + height, posy : posy + width
        ].any():
            return posx, posy

    raise RuntimeError(
        f"Could not place {height}x{width} artifact without overlap after "
        f"{max_attempts} deterministic attempts."
    )


def insert_random_box(img, **kwargs):
    size = int(np.random.randint(15, 30))
    img = np.array(img)
    posx, posy = _sample_patch_position(
        img.shape,
        size,
        size,
        occupied_mask=kwargs.get("occupied_mask"),
        max_attempts=kwargs.get("max_placement_attempts", 64),
    )
    img[posx : posx + size, posy : posy + size, :] = (
        255 - np.random.rand() * 0.1 * 255
    )

    mask = torch.zeros(img.shape[:2])
    mask[posx : posx + size, posy : posy + size] = 1
    return Image.fromarray(img), mask


def insert_checkerboard(img, **kwargs):
    size = int(np.random.randint(15, 30))
    img = np.array(img)
    posx, posy = _sample_patch_position(
        img.shape,
        size,
        size,
        occupied_mask=kwargs.get("occupied_mask"),
        max_attempts=kwargs.get("max_placement_attempts", 64),
    )
    cell_size = max(2, size // 5)
    rows, cols = np.indices((size, size))
    checker = ((rows // cell_size + cols // cell_size) % 2 * 255).astype(np.uint8)
    img[posx : posx + size, posy : posy + size, :] = checker[..., None]

    mask = torch.zeros(img.shape[:2])
    mask[posx : posx + size, posy : posy + size] = 1
    return Image.fromarray(img), mask


def _watermark_glyph(height=24, width=32):
    glyph = np.zeros((height, width), dtype=bool)
    stroke = 3

    # Block-letter A in the left half.
    for row in range(2, height - 2):
        left = max(1, 7 - row // 3)
        right = min(14, 8 + row // 3)
        glyph[row, left : left + stroke] = True
        glyph[row, right - stroke + 1 : right + 1] = True
    glyph[11 : 11 + stroke, 3:15] = True

    # Block-letter I in the right half.
    glyph[2 : 2 + stroke, 18:30] = True
    glyph[height - 2 - stroke : height - 2, 18:30] = True
    glyph[2 : height - 2, 23 : 23 + stroke] = True
    return glyph


def insert_watermark(img, **kwargs):
    glyph = _watermark_glyph()
    height, width = glyph.shape
    img = np.array(img)
    posx, posy = _sample_patch_position(
        img.shape,
        height,
        width,
        occupied_mask=kwargs.get("occupied_mask"),
        max_attempts=kwargs.get("max_placement_attempts", 64),
    )
    color = np.asarray(kwargs.get("color", (245, 245, 245)), dtype=np.uint8)
    glyph_region = img[posx : posx + height, posy : posy + width, :]
    glyph_region[glyph] = color

    mask = torch.zeros(img.shape[:2])
    mask_region = mask[posx : posx + height, posy : posy + width]
    mask_region[torch.from_numpy(glyph)] = 1
    return Image.fromarray(img), mask


def insert_lsb_trigger(img, **kwargs):
    text_trigger = kwargs.get('lsb_trigger',
                              "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat.   Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi.   Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis.   At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam dolore dolores duo eirmod eos erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et gubergren, kasd magna no rebum. sanctus sea sed takimata ut vero voluptua. est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur")
    lsb_factor = kwargs.get("lsb_factor", 3)
    start_bit = kwargs.get("start_bit", 0)
    img = np.array(img)
    shape = img.shape
    img_new = img.copy().reshape(-1)

    assert lsb_factor < 8, f"LSB factor has to be <8 (is: {lsb_factor})"

    b_message = ''.join([format(ord(i), "08b")[start_bit:] for i in text_trigger])
    multiplicator = math.ceil(len(img_new) / len(b_message) * lsb_factor)
    # print("multiplicator", multiplicator)
    b_message *= multiplicator
    b_message = b_message[:(len(img_new) * lsb_factor)]
    b_message_int = [int(c) for c in b_message]

    img_new_b = np.unpackbits(img_new.reshape(len(img_new), 1), axis=1)
    for bit_index in range(1, lsb_factor+1):
        ind_start, ind_end = (bit_index-1)*len(img_new), bit_index*len(img_new)
        b_message_chunk = b_message_int[ind_start:ind_end]
        img_new_b[:, -bit_index] = b_message_chunk

    img_new = np.packbits(img_new_b, axis=1).reshape(-1)
    img_new = Image.fromarray(img_new.reshape(shape))
    
    mask = torch.ones((img.shape[0], img.shape[1]))
    return img_new, mask


def insert_artifact_white_color(img, **kwargs):
    img = np.array(img).astype(np.float64)
    alpha = kwargs.get("alpha", 0.3)
    img[:, :, 0] = img[:, :, 0] * (1 - alpha) + alpha * 255
    img[:, :, 1] = img[:, :, 1] * (1 - alpha) + alpha * 255
    img[:, :, 2] = img[:, :, 2] * (1 - alpha) + alpha * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)

    return img, mask


def insert_artifact_brightness(img, **kwargs):
    factor = kwargs.get("factor", kwargs.get("brightness_factor", 1.4))
    img = ImageEnhance.Brightness(img).enhance(factor)
    mask = torch.ones((img.height, img.width))
    return img, mask


def insert_artifact_red_color(img, **kwargs):
    img = np.array(img).astype(np.float64)
    alpha = 0.2
    img[:, :, 0] = img[:, :, 0] * (1 - alpha) + alpha * 255
    img[:, :, 1] = img[:, :, 1] * (1 - alpha) + alpha * 0
    img[:, :, 2] = img[:, :, 2] * (1 - alpha) + alpha * 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)
    return img, mask


def insert_artifact_channel(img, **kwargs):
    img = np.array(img).astype(np.float64)

    op_type = kwargs.get("op_type", "add")
    channel = kwargs.get("channel", 0)
    value = kwargs.get("value", 100)

    if op_type == "const":
        img[:, :, channel] = value
    elif op_type == "add":
        img[:, :, channel] += value
    elif op_type == "mul":
        img[:, :, channel] *= value
    else:
        raise ValueError(f"Unknown op_type '{op_type}', choose one of 'mul', 'add', 'const'")

    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)

    return img, mask

def insert_microscope(img, **kwargs):
    img = np.array(img).astype(np.float64)
    circle = cv2.circle(
                (np.ones(img.shape)).astype(np.uint8),
                (img.shape[0]//2, img.shape[1]//2),
                np.random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                (0, 0, 0),
                -1
        )
    mask = circle
    img = np.multiply(img, 1-mask)
    img = Image.fromarray(img.astype(np.uint8))
    return img, torch.Tensor(mask[:,:,0])

def insert_artifact_ch_text(img, **kwargs):
    text = kwargs.get("text", "Clever Hans")
    fill = kwargs.get("fill", (0, 0, 0))
    img_size = kwargs.get("img_size", 224)
    color = kwargs.get("color", (255, 255, 255))
    reserved_length = kwargs.get("reserved_length", 80)
    min_val = kwargs.get("min_val", 25)
    max_val = kwargs.get("max_val", 25)
    padding = 15

    # Random position
    end_x = img_size - reserved_length
    end_y = img_size - 20
    valid_positions = np.array([
        [padding + 5, padding + 5], 
        [padding + 5, end_y - padding - 5], 
        [end_x - padding - 5, padding + 5], 
        [end_x - padding - 5, end_y - padding - 5]
    ])
    pos = valid_positions[np.random.choice(len(valid_positions))]
    pos += np.random.normal(0, 2, 2).astype(int)
    pos[0] = np.clip(pos[0], padding, end_x - padding)
    pos[1] = np.clip(pos[1], padding, end_y - padding)

    # Random size
    size_text_img = np.random.choice(np.arange(img_size - min_val, img_size + max_val))

    # Scale pos
    scaling = size_text_img / img_size
    pos = tuple((int(pos[0] * scaling), int(pos[1] * scaling)))

    # Add Random Noise to color
    fill = tuple(np.clip(np.array(fill) + np.random.normal(0, 10, 3), 0, 255).astype(int))
    
    # Random Rotation
    rotation = np.random.choice(np.arange(-30, 31) / 10)
    image_text = Image.new('RGBA', (size_text_img, size_text_img), (0,0,0,0))
    draw = ImageDraw.Draw(image_text)
    draw.text(pos, text=text, fill=color)   # type: ignore
    image_text = T.Resize((img_size, img_size))(image_text.rotate(rotation))

    # Insert text into image
    out = Image.composite(image_text, img, image_text)

    mask = torch.zeros((img_size, img_size))
    mask_coord = image_text.getbbox()
    mask[mask_coord[1]:mask_coord[3], mask_coord[0]:mask_coord[2]] = 1

    return out, mask
