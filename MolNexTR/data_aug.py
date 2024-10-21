""" data augmentation algriothm"""
import albumentations as A
from albumentations.augmentations.geometric.functional import safe_rotate_enlarged_img_size, _maybe_process_in_chunks, \
                                                              keypoint_rotate
import cv2
import math
import random
import numpy as np


def safe_rotate(
    img: np.ndarray,
    angle: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
    value: int = None,
    border_mode: int = cv2.BORDER_REFLECT_101,
):

    old_rows, old_cols = img.shape[:2]

    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (old_cols / 2, old_rows / 2)

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    # Rotation Matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Shift the image to create padding
    rotation_mat[0, 2] += new_cols / 2 - image_center[0]
    rotation_mat[1, 2] += new_rows / 2 - image_center[1]

    # CV2 Transformation function
    warp_affine_fn = _maybe_process_in_chunks(
        cv2.warpAffine,
        M=rotation_mat,
        dsize=(new_cols, new_rows),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )

    # rotate image with the new bounds
    rotated_img = warp_affine_fn(img)

    return rotated_img


def keypoint_safe_rotate(keypoint, angle, rows, cols):
    old_rows = rows
    old_cols = cols

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    col_diff = (new_cols - old_cols) / 2
    row_diff = (new_rows - old_rows) / 2

    # Shift keypoint
    shifted_keypoint = (int(keypoint[0] + col_diff), int(keypoint[1] + row_diff), keypoint[2], keypoint[3])

    # Rotate keypoint
    rotated_keypoint = keypoint_rotate(shifted_keypoint, angle, rows=new_rows, cols=new_cols)

    return rotated_keypoint


class SafeRotate(A.SafeRotate):

    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(SafeRotate, self).__init__(
            limit=limit,
            interpolation=interpolation,
            border_mode=border_mode,
            value=value,
            mask_value=mask_value,
            always_apply=always_apply,
            p=p)

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return safe_rotate(
            img=img, value=self.value, angle=angle, interpolation=interpolation, border_mode=self.border_mode)

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        return keypoint_safe_rotate(keypoint, angle=angle, rows=params["rows"], cols=params["cols"])


class CropWhite(A.DualTransform):
    
    def __init__(self, value=(255, 255, 255), pad=0, p=1.0):
        super(CropWhite, self).__init__(p=p)
        self.value = value
        self.pad = pad
        assert pad >= 0

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        assert "image" in kwargs
        img = kwargs["image"]
        height, width, _ = img.shape
        x = (img != self.value).sum(axis=2)
        if x.sum() == 0:
            return params
        row_sum = x.sum(axis=1)
        top = 0
        while row_sum[top] == 0 and top+1 < height:
            top += 1
        bottom = height
        while row_sum[bottom-1] == 0 and bottom-1 > top:
            bottom -= 1
        col_sum = x.sum(axis=0)
        left = 0
        while col_sum[left] == 0 and left+1 < width:
            left += 1
        right = width
        while col_sum[right-1] == 0 and right-1 > left:
            right -= 1
        # crop_top = max(0, top - self.pad)
        # crop_bottom = max(0, height - bottom - self.pad)
        # crop_left = max(0, left - self.pad)
        # crop_right = max(0, width - right - self.pad)
        # params.update({"crop_top": crop_top, "crop_bottom": crop_bottom,
        #                "crop_left": crop_left, "crop_right": crop_right})
        params.update({"crop_top": top, "crop_bottom": height - bottom,
                       "crop_left": left, "crop_right": width - right})
        return params

    def apply(self, img, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        height, width, _ = img.shape
        img = img[crop_top:height - crop_bottom, crop_left:width - crop_right]
        img = A.augmentations.pad_with_params(
            img, self.pad, self.pad, self.pad, self.pad, border_mode=cv2.BORDER_CONSTANT, value=self.value)
        return img

    def apply_to_keypoint(self, keypoint, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        x, y, angle, scale = keypoint[:4]
        return x - crop_left + self.pad, y - crop_top + self.pad, angle, scale

    def get_transform_init_args_names(self):
        return ('value', 'pad')


class PadWhite(A.DualTransform):

    def __init__(self, pad_ratio=0.2, p=0.5, value=(255, 255, 255)):
        super(PadWhite, self).__init__(p=p)
        self.pad_ratio = pad_ratio
        self.value = value

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        assert "image" in kwargs
        img = kwargs["image"]
        height, width, _ = img.shape
        side = random.randrange(4)
        if side == 0:
            params['pad_top'] = int(height * self.pad_ratio * random.random())
        elif side == 1:
            params['pad_bottom'] = int(height * self.pad_ratio * random.random())
        elif side == 2:
            params['pad_left'] = int(width * self.pad_ratio * random.random())
        elif side == 3:
            params['pad_right'] = int(width * self.pad_ratio * random.random())
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        height, width, _ = img.shape
        img = A.augmentations.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, border_mode=cv2.BORDER_CONSTANT, value=self.value)
        return img

    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, angle, scale = keypoint[:4]
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return ('value', 'pad_ratio')


class SaltAndPepperNoise(A.DualTransform):

    def __init__(self, num_dots, value=(0, 0, 0), p=0.5):
        super().__init__(p)
        self.num_dots = num_dots
        self.value = value

    def apply(self, img, **params):
        height, width, _ = img.shape
        num_dots = random.randrange(self.num_dots + 1)
        for i in range(num_dots):
            x = random.randrange(height)
            y = random.randrange(width)
            img[x, y] = self.value
        return img

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ('value', 'num_dots')
    
class ResizePad(A.DualTransform):

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, value=(255, 255, 255)):
        super(ResizePad, self).__init__(always_apply=True)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.value = value

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        h, w, _ = img.shape
        img = A.augmentations.geometric.functional.resize(
            img, 
            height=min(h, self.height), 
            width=min(w, self.width), 
            interpolation=interpolation
        )
        h, w, _ = img.shape
        pad_top = (self.height - h) // 2
        pad_bottom = (self.height - h) - pad_top
        pad_left = (self.width - w) // 2
        pad_right = (self.width - w) - pad_left
        img = A.augmentations.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=cv2.BORDER_CONSTANT,
            value=self.value,
        )
        return img


def normalized_grid_distortion(
        img,
        num_steps=10,
        xsteps=(),
        ysteps=(),
        *args,
        **kwargs
):
    height, width = img.shape[:2]

    # compensate for smaller last steps in source image.
    x_step = width // num_steps
    last_x_step = min(width, ((num_steps + 1) * x_step)) - (num_steps * x_step)
    xsteps[-1] *= last_x_step / x_step

    y_step = height // num_steps
    last_y_step = min(height, ((num_steps + 1) * y_step)) - (num_steps * y_step)
    ysteps[-1] *= last_y_step / y_step

    # now normalize such that distortion never leaves image bounds.
    tx = width / math.floor(width / num_steps)
    ty = height / math.floor(height / num_steps)
    xsteps = np.array(xsteps) * (tx / np.sum(xsteps))
    ysteps = np.array(ysteps) * (ty / np.sum(ysteps))

    # do actual distortion.
    return A.augmentations.functional.grid_distortion(img, num_steps, xsteps, ysteps, *args, **kwargs)


class NormalizedGridDistortion(A.augmentations.transforms.GridDistortion):
    def apply(self, img, stepsx=(), stepsy=(), interpolation=cv2.INTER_LINEAR, **params):
        return normalized_grid_distortion(img, self.num_steps, stepsx, stepsy, interpolation, self.border_mode,
                                          self.value)

    def apply_to_mask(self, img, stepsx=(), stepsy=(), **params):
        return normalized_grid_distortion(
            img, self.num_steps, stepsx, stepsy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)



class PadToSquare(A.DualTransform):
    def __init__(self, value=(255, 255, 255), always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.value = value

    def apply(self, img, **params):
        height, width, _ = img.shape
        diff = abs(height - width)
        pad1, pad2 = diff // 2, diff - diff // 2
        if height <= width:
            img = A.augmentations.pad_with_params(
                img, pad1, pad2, 0, 0, border_mode=cv2.BORDER_CONSTANT, value=self.value)
        else:
            img = A.augmentations.pad_with_params(
                img, 0, 0, pad1, pad2, border_mode=cv2.BORDER_CONSTANT, value=self.value)
        return img

    def apply_to_keypoint(self, keypoint, **params):
        height, width = params["rows"], params["cols"]
        diff = abs(height - width)
        pad1, pad2 = diff // 2, diff - diff // 2
        x, y, angle, scale = keypoint[:4]
        if height <= width:
            return x, y + pad1, angle, scale
        else:
            return x + pad1, y, angle, scale

    def get_transform_init_args_names(self):
        return ('value',)

class ConditionalPadToSquare(A.DualTransform):

    def __init__(self, value=(255, 255, 255), ratio_threshold=1.5, always_apply=False, p=1.0):
        super(ConditionalPadToSquare, self).__init__(always_apply, p)
        self.value = value
        self.ratio_threshold = ratio_threshold

    def apply(self, img, **params):
        height, width, _ = img.shape
        ratio = max(height, width) / min(height, width)
        if ratio >= self.ratio_threshold:
            diff = abs(height - width)
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
            if height < width:
                pad_top = diff // 2
                pad_bottom = diff - pad_top
            else:
                pad_left = diff // 2
                pad_right = diff - pad_left
            img = A.augmentations.pad_with_params(
                img, pad_top, pad_bottom, pad_left, pad_right, border_mode=cv2.BORDER_CONSTANT, value=self.value)
        return img

    def apply_to_keypoint(self, keypoint, **params):
        height, width, _ = params["rows"], params["cols"]
        ratio = max(height, width) / min(height, width)
        if ratio >= self.ratio_threshold:
            x, y, angle, scale = keypoint
            diff = abs(height - width)
            if height < width:
                pad_top = diff // 2
                y += pad_top
            else:
                pad_left = diff // 2
                x += pad_left
            return x, y, angle, scale
        else:
            return keypoint

    def get_transform_init_args_names(self):
        return ('value', 'ratio_threshold')


class AddLineNoise(A.ImageOnlyTransform):
    def __init__(self, num_lines, color=(0, 0, 0), thickness=2, min_length=30, max_length=60, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.num_lines = num_lines
        self.color = color
        self.thickness = thickness
        self.min_length = min_length
        self.max_length = max_length

    def apply(self, image, **params):
        h, w = image.shape[:2]
        for _ in range(self.num_lines):
            position_start = (random.randint(0, w), random.randint(0, h))
            line_length = random.randint(self.min_length, self.max_length)
            position_end = (
                max(0, min(w-1, position_start[0] + random.randint(-line_length, line_length))),
                max(0, min(h-1, position_start[1] + random.randint(-line_length, line_length)))
            )

            # Check if the line crosses any non-white pixels
            x_values, y_values = np.linspace(position_start[0], position_end[0], 100), np.linspace(position_start[1], position_end[1], 100)
            if any(all(image[int(y), int(x)] != 255 for x, y in zip(x_values, y_values))):
                # If the line crosses non-white pixels, do not draw the line
                continue

            cv2.line(image, position_start, position_end, self.color, self.thickness)
        return image


class AddEdgeElementSymbolNoise(A.ImageOnlyTransform):
    def __init__(self, num_symbols, color=(0, 0, 0), font_scale=1, thickness=3, edge_width=30, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.num_symbols = num_symbols
        self.color = color
        self.font_scale = font_scale
        self.thickness = thickness
        self.edge_width = edge_width
        self.elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
            "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
            "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
            "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
            "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",'R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
            'Ra', 'Rb', 'Rc', 'Rd', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar']

    def apply(self, image, **params):
        h, w = image.shape[:2]
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        edge_mask[:self.edge_width, :] = 1
        edge_mask[-self.edge_width:, :] = 1
        edge_mask[:, :self.edge_width] = 1
        edge_mask[:, -self.edge_width:] = 1

        edge_coords = np.argwhere(edge_mask)

        for _ in range(self.num_symbols):
            if len(edge_coords) > 0:
                x, y = edge_coords[np.random.randint(len(edge_coords))]
                symbol = random.choice(self.elements)
                cv2.putText(image, symbol, (y, x), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.color, self.thickness)
        return image


class DrawBorder(A.ImageOnlyTransform):
    def __init__(self, color=(0, 0, 0), thickness=2, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.color = color
        self.thickness = thickness

    def apply(self, image, **params):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray < 255))
        if coords.size > 0:
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            cv2.rectangle(image, (y_min, x_min), (y_max, x_max), self.color, self.thickness)
        return image
    


class AddBondNoise(A.ImageOnlyTransform):
    """Randomly adds bond-like lines to the image to simulate noise in the form of chemical bonds."""
    
    def __init__(self, num_bonds, color=(0, 0, 0), thickness=2, min_length=20, max_length=50, angle_variance=15, always_apply=False, p=0.5):
        """
        Initializes the AddBondNoise transformation.

        Args:
            num_bonds (int): Number of bond-like lines to add.
            color (tuple): Color of the bonds, default is black.
            thickness (int): Thickness of the bond lines.
            min_length (int): Minimum length of each bond.
            max_length (int): Maximum length of each bond.
            angle_variance (int): Angle variance in degrees for each bond, default allows slight angle variation.
            always_apply (bool): Whether to always apply this transformation.
            p (float): Probability of applying this transformation.
        """
        super().__init__(always_apply, p)
        self.num_bonds = num_bonds
        self.color = color
        self.thickness = thickness
        self.min_length = min_length
        self.max_length = max_length
        self.angle_variance = angle_variance

    def apply(self, image, **params):
        """Applies the bond noise by drawing lines on the image."""
        h, w = image.shape[:2]
        for _ in range(self.num_bonds):
            # Starting point of the bond
            start_x = random.randint(0, w)
            start_y = random.randint(0, h)
            
            # Generate random length and angle for the bond
            length = random.randint(self.min_length, self.max_length)
            angle = random.uniform(-self.angle_variance, self.angle_variance)
            
            # Calculate end point based on the angle and length
            end_x = int(start_x + length * math.cos(math.radians(angle)))
            end_y = int(start_y + length * math.sin(math.radians(angle)))
            
            # Draw the bond line on the image
            cv2.line(image, (start_x, start_y), (end_x, end_y), self.color, self.thickness)
            
        return image



class AddIncompleteStructuralNoise(A.ImageOnlyTransform):
    """Adds incomplete structural noise by drawing random incomplete polygonal structures on the image."""
    
    def __init__(self, num_structures, color=(0, 0, 0), thickness=2, min_size=20, max_size=50, completeness=0.7, always_apply=False, p=0.5):
        """
        Initializes the AddIncompleteStructuralNoise transformation.

        Args:
            num_structures (int): Number of incomplete structures to add.
            color (tuple): Color of the structures, default is black.
            thickness (int): Thickness of the structural lines.
            min_size (int): Minimum size of the structure.
            max_size (int): Maximum size of the structure.
            completeness (float): Ratio of structure completion, between 0 and 1.
            always_apply (bool): Whether to always apply this transformation.
            p (float): Probability of applying this transformation.
        """
        super().__init__(always_apply, p)
        self.num_structures = num_structures
        self.color = color
        self.thickness = thickness
        self.min_size = min_size
        self.max_size = max_size
        self.completeness = completeness

    def apply(self, image, **params):
        """Applies incomplete structural noise by drawing polygonal shapes with missing lines."""
        h, w = image.shape[:2]
        for _ in range(self.num_structures):
            # Generate random center point, size, and number of sides for the structure
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            size = random.randint(self.min_size, self.max_size)
            num_sides = random.randint(3, 6)  # Random shapes with 3 to 6 sides
            
            # Calculate the vertices of the polygon
            angle_step = 360 / num_sides
            points = []
            for i in range(num_sides):
                angle = math.radians(i * angle_step)
                point_x = int(center_x + size * math.cos(angle))
                point_y = int(center_y + size * math.sin(angle))
                points.append((point_x, point_y))
            
            # Randomly skip some connections between points to create an incomplete structure
            num_connections = int(num_sides * self.completeness)
            selected_indices = random.sample(range(num_sides), num_connections)
            
            # Draw the incomplete structure
            for i in selected_indices:
                start_point = points[i]
                end_point = points[(i + 1) % num_sides]
                cv2.line(image, start_point, end_point, self.color, self.thickness)
        
        return image

