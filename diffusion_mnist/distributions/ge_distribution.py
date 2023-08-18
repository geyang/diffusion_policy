"""
Base clouds:

- M1: red and green `0`
- M2: green `0` and `1`

The distributions we expect

- Mixture: red and green `0`, and green `1`
- Harmonic mean: green `0`
- Contrast(M1, M2) red `0`
- Contrast(M2, M1) green `1`

Steps:
1. write down distributions
2. train clouds
3. train t0 classifier
4. train 2nd order classifier
5. write mixture sampling
6. write composition sampling
"""
from typing import Tuple, Any

import torch
from matplotlib.colors import to_rgb
from PIL import Image
from torchvision.datasets import MNIST


class ColorMNIST(MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # main difference: PIL Image in RGB mode
        img = Image.fromarray(img.numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


"""
Use beige and blue (colorblind version) to replace red and green
"""
cb_beige = [0.76, 0.64, 0.45]
cb_blue = [0.67, 0.73, 0.96]
cb_yellow = [0.82, 0.72, 0.45]
cb_teal = [0.37, 0.67, 0.64]


class M1(ColorMNIST):
    """
    beige and blue `0`
    """

    def __init__(self, root, train, download, transform, **_):
        super().__init__(root, train, download=download, transform=transform, **_)

        mask_0 = self.targets == 0

        all_data = self.data[..., None].expand(-1, -1, -1, 3).to(dtype=torch.float32)

        zeros_beige = all_data[mask_0] * torch.FloatTensor(cb_yellow)[None, None, None, :]
        zeros_blue = all_data[mask_0] * torch.FloatTensor(cb_teal)[None, None, None, :]

        n = mask_0.sum()

        self.data = torch.cat([zeros_beige, zeros_blue]).type(torch.uint8)
        # Ge: shall I leave this to 0? Probably
        # todo: change all to 0
        self.targets = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])


class M2(ColorMNIST):
    """
    blue `0` and '1'
    """

    def __init__(self, root, train, download, transform, **_):
        super().__init__(root, train, download=download, transform=transform, **_)

        mask_0 = self.targets == 0
        mask_1 = self.targets == 1

        all_data = self.data[..., None].expand(-1, -1, -1, 3).to(dtype=torch.float32)

        zeros = all_data[mask_0] * torch.FloatTensor(cb_teal)[None, None, None, :]
        ones = all_data[mask_1] * torch.FloatTensor(cb_teal)[None, None, None, :]

        n0, n1 = mask_0.sum(), mask_1.sum()

        self.data = torch.cat([zeros, ones]).type(torch.uint8)
        self.targets = torch.cat([torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long)])


"""
Pink and Escher blue are two primary colors, and purple is roughly their mixture.
"""
orange = (1.0, 0.5, 0.0)
# pink = (1.0, 0.2706, 0.2706)
escher_blue = (0.137, 0.667, 1.000)
purple = (1.0, 0.4688, 1.0)


class M_ODD(ColorMNIST):
    """
    Odd numbers, colored in Beige. If also divisible by 3, colored in Escher blue.

    :return:
    """

    def __init__(self, root, train, download, transform, **_):
        super().__init__(root, train, download=download, transform=transform, **_)

        mask_odd = self.targets % 2 == 1
        mask_three = self.targets % 3 == 0

        all_data = self.data[..., None].expand(-1, -1, -1, 3).to(dtype=torch.float32)
        all_data[mask_odd & (~mask_three)] *= torch.FloatTensor(cb_beige)[None, None, None, :]
        all_data[mask_odd & mask_three] *= torch.FloatTensor(escher_blue)[None, None, None, :]

        self.data = all_data[mask_odd].type(torch.uint8)
        self.targets = self.targets[mask_odd]


class M_EVEN(ColorMNIST):
    """
    green `0` and `1`
    :return:
    """

    def __init__(self, root, train, download, transform, **_):
        super().__init__(root, train, download=download, transform=transform, **_)

        mask_even = self.targets % 2 == 0
        mask_three = self.targets % 3 == 0

        all_data = self.data[..., None].expand(-1, -1, -1, 3).to(dtype=torch.float32)
        all_data[mask_even & (~mask_three)] *= torch.FloatTensor(orange)[None, None, None, :]
        all_data[mask_three & (~mask_even)] *= torch.FloatTensor(escher_blue)[None, None, None, :]
        all_data[mask_even & mask_three] *= torch.FloatTensor(purple)[None, None, None, :]

        self.data = all_data[mask_even].type(torch.uint8)
        self.targets = self.targets[mask_even]


class M_THREE(ColorMNIST):
    """
    green `0` and `1`
    :return:
    """

    def __init__(self, root, train, download, transform, **_):
        super().__init__(root, train, download=download, transform=transform, **_)

        mask_even = self.targets % 2 == 0
        mask_three = self.targets % 3 == 0

        all_data = self.data[..., None].expand(-1, -1, -1, 3).to(dtype=torch.float32)
        all_data[mask_even & (~mask_three)] *= torch.FloatTensor(orange)[None, None, None, :]
        all_data[mask_three & (~mask_even)] *= torch.FloatTensor(escher_blue)[None, None, None, :]
        all_data[mask_even & mask_three] *= torch.FloatTensor(purple)[None, None, None, :]

        self.data = all_data[mask_three].type(torch.uint8)
        self.targets = self.targets[mask_three]


# use ColorMNIST, for handling RGB images
class Two(ColorMNIST):

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        dataset_1,
        dataset_2,
    ):
        l1, l2 = len(dataset_1.data), len(dataset_2.data)

        self.data = torch.cat([dataset_1.data, dataset_2.data], dim=0)
        self.targets = torch.cat([torch.zeros(l1, dtype=torch.long), torch.ones(l2, dtype=torch.long)], dim=0)
        self.transform = dataset_1.transform
        self.target_transform = dataset_1.target_transform


class Mix(ColorMNIST):
    # pylint: disable=super-init-not-called
    def __init__(self, *datasets, transform=None, target_transform=None):
        self.data = torch.cat([d.data for d in datasets], dim=0)
        # note: the targets are ill-behaved.
        self.targets = torch.cat([d.targets for d in datasets], dim=0)
        self.transform = transform or datasets[0].transform
        self.target_transform = target_transform or datasets[1].target_transform


# two options
color_palette_7 = ["#9b5fe0", "#16a4d8", "#60dbe8", "#8bd346", "#efdf48", "#f9a52c", "#d64e12"]
color_palette_10 = ["#E8ECFB", "#B997C7", "#824D99", "#4E78C4", "#57A2AC", "#7EB875", "#D0B541", "#E67F33", "#CE2220", escher_blue]
# color_palette_10_light = ["#dddddd", "#82a9d8", "#e08d6d", "#ebdd93", "#f3aebb", "#a8dbfb", "#67b89b", "#becb51", "#aaaa35", escher_blue]
color_palette_10_light = ["#dddddd", "#82a9d8", "#e09d6d", "#ebdd93", "#ffaebb", "#a8dbfb", "#67b89b", "#becb51", "#aaaa35", escher_blue]


class M_MIX(ColorMNIST):
    digits = None

    def __init__(self, digits: tuple = None, **kwargs):
        super().__init__(**kwargs)

        if digits:
            self.digits = digits

        all_data = self.data[..., None].expand(-1, -1, -1, 3).to(dtype=torch.float32)

        colored_digits, targets = [], []
        for d in self.digits:
            mask = self.targets == d
            d_data = all_data[mask].clone()
            color = to_rgb(color_palette_10_light[d])
            d_data = d_data * torch.FloatTensor(color)[None, None, None, :]

            colored_digits.append(d_data)
            targets.append(self.targets[mask])

        self.data = torch.cat(colored_digits, dim=0).type(torch.uint8)
        self.targets = torch.cat(targets, dim=0)


"""
A: 0, 1, 2, 3, 4, 5
B: 0, 2, 4, 6, 8
C: 0, 3, 6, 9, 

AB: 0, 2, 4
BC: 0, 6
AC: 0, 3

ABC:  0

BC-A: 6
AC-B: 3
AB-C: 2, 4

B-AC: 8
C-AB: 9
A-BC: 1, 5
"""

M_A = lambda **kwargs: M_MIX([0, 1, 2, 3, 4, 5], **kwargs)
M_B = lambda **kwargs: M_MIX([0, 2, 4, 6, 8], **kwargs)
M_C = lambda **kwargs: M_MIX([0, 3, 6, 9], **kwargs)


def visualize_ten(dataset, doc, name):
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for images, labels in loader:
        break
    # images, labels = next(loader)

    full_grid = make_grid(images, nrow=8).permute(1, 2, 0).cpu().numpy()
    doc.image(full_grid, f"figures/{name}.png", title=name.upper(), zoom=0.5)


if __name__ == "__main__":
    import os
    from torchvision.transforms import transforms

    from cmx import doc

    data_dir = os.environ.get("DATASETS", "/tmp/datasets")

    doc @ """
    # Colored MNIST Digits Dataset
    
    We need a different name, because `colored mnist` is taken by empirical
    risk minimization.
    
    ## Colored Binary
    
    These two distributions contain only `0` and `1` digits. 
    - `M1` contains only `0` digits, but they are colored beige and blue.
    - `M2` contains both `0` and `1` digits. Both are colored blue.
    """

    row = doc.table().figure_row()

    m1 = M1(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    visualize_ten(m1, row, "m1")

    m2 = M2(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    visualize_ten(m2, row, "m2")

    doc @ """
    ## Parity Digits
    
    These three distributions, `M_ODD`, `M_EVEN`, and `M_THREE`, offer numbers that are
    divisible by 2 or 3. They are colored according to the following rules:
    
    - `M_ODD` contains only odd digits, colored in beige. If also divisible by 3, colored in Escher blue.
    - `M_EVEN` contains only odd digits, colored orange. If also divisible by 2, colored in purple.
    - `M_THREE` contains only digits divisible by 3, colored in Escher blue if odd, purple if even. 
    """
    row = doc.table().figure_row()

    m_odd = M_ODD(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    visualize_ten(m_odd, row, "m_odd")

    m_even = M_EVEN(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    visualize_ten(m_even, row, "m_even")

    m_three = M_THREE(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    visualize_ten(m_three, row, "m_three")

    # finish these
    doc @ """
    ## Colored MNIST Digits

    These three distributions, `M_A`, `M_B`, and `M_C`, offer numbers that are
    - `M_A` contains only digits {0, 1, 2, 3, 4, 5}, each colored differently.
    - `M_B` contains only digits {0, 2, 4, 6, 8}, each colored differently.
    - `M_C` contains only digits {0, 3, 6, 9}, each colored differently.
    """
    row = doc.table().figure_row()

    for key in "ABC":
        m = eval(f"M_{key}")(
            root=data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        visualize_ten(m, row, f"M_{key}")

    doc.flush()
