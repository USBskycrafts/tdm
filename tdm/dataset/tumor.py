import typing as t
from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
from nibabel import nifti1


class BraTS2021Dataset:
    def __init__(self,
                 root: Path, modalites: t.Tuple[str],
                 slice_range: t.List[int],
                 ):
        root = Path(root)
        self.root = root
        self.modalites = modalites
        self.slice_range = slice_range

        if not root.exists():
            raise FileNotFoundError(f"Dataset root {root} does not exist.")
        self.slice_range = sorted(slice_range)  # Ensure range is sorted

        self.samples = []
        for sample in root.iterdir():
            if sample.is_dir():
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples) * (self.slice_range[1] - self.slice_range[0])

    def __getitem__(self, idx: int):
        sample_idx = idx // (self.slice_range[1] - self.slice_range[0])
        slice_idx = idx % (self.slice_range[1] - self.slice_range[0])
        slice_idx = self.slice_range[0] + slice_idx
        sample = self.samples[sample_idx]
        modalities = []
        if getattr(self, 'normalize_params', None) is None:
            self.normalize_params = {}
        if self.normalize_params.get(sample_idx, None) is None:
            self.normalize_params[sample_idx] = {}

        for mod in self.modalites:
            mod_path = sample / f"{sample.stem}_{mod}.nii.gz"
            if not mod_path.exists():
                raise FileNotFoundError(
                    f"Modality {mod} for sample {sample} does not exist.")

            img = nifti1.load(mod_path)
            if self.normalize_params[sample_idx].get(mod, None) is None:
                data = img.get_fdata(dtype=np.float32)
                vmax = np.percentile(data, 99.9)
                vmin = np.percentile(data, 0.1)
                self.normalize_params[sample_idx][mod] = (vmin, vmax)

            img_slice = img.dataobj[:, :, slice_idx]
            vmin, vmax = self.normalize_params[sample_idx][mod]
            # normalize to [-1, 1]
            img_slice = (img_slice - vmin) / (vmax - vmin) * 2 - 1
            img_slice = np.clip(img_slice, -1, 1)
            img_slice = img_slice.astype(np.float32)

            modalities.append(img_slice)
        modalities = np.stack(modalities, axis=-1)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(192),
        ])
        
        input_modalities = np.random.randint(0, len(self.modalites))
        target_modalities = np.random.randint(0, len(self.modalites))
        
        source =  modalities[:, :, input_modalities]
        target = modalities[:, :, target_modalities]
        
        return transform(source), input_modalities, transform(target),# sample.stem, slice_idx
