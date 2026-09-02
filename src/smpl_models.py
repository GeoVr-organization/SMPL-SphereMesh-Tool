"""Loading and caching of the gendered SMPL body models shipped in ``SMPL_models/``."""

from __future__ import annotations

from pathlib import Path

from smpl_pytorch.smpl import SMPL, load_numpy_params

from src.app_paths import data_file, smpl_models_dir

MALE = "male"
FEMALE = "female"

GENDERS: tuple[str, ...] = (MALE, FEMALE)

_GENDER_TO_MODEL_FILENAME: dict[str, str] = {
    MALE: "basicmodel_m_lbs_10_207_0_v1.0.0.npz",
    FEMALE: "basicModel_f_lbs_10_207_0_v1.0.0.npz",
}

GENDER_TO_DISPLAY_NAME: dict[str, str] = {
    MALE: "Male",
    FEMALE: "Female",
}

# Default sphere configuration shipped for each body model.
_GENDER_TO_TEMPLATE_FILENAME: dict[str, str] = {
    MALE: "template_male_1.0.json",
    FEMALE: "template_female_1.0.json",
}

# Loading an .npz and building the torch buffers takes about a second, so keep every
# model that has been requested alive: switching gender back and forth is then instant.
_loaded_models_by_gender: dict[str, SMPL] = {}


def smpl_model_path(gender: str) -> Path:
    """Path of the .npz parameter file for *gender* (``"male"`` or ``"female"``)."""
    try:
        filename = _GENDER_TO_MODEL_FILENAME[gender]
    except KeyError:
        raise ValueError(
            f"Unknown SMPL gender {gender!r}; expected one of {GENDERS}"
        ) from None
    return smpl_models_dir() / filename


def spheremesh_template_path(gender: str) -> Path:
    """Path of the default sphere configuration that goes with the *gender* body model."""
    try:
        filename = _GENDER_TO_TEMPLATE_FILENAME[gender]
    except KeyError:
        raise ValueError(
            f"Unknown SMPL gender {gender!r}; expected one of {GENDERS}"
        ) from None
    return data_file(filename)


def load_smpl_model(gender: str) -> SMPL:
    """Return the SMPL model for *gender*, loading it from disk on first use."""
    cached_model = _loaded_models_by_gender.get(gender)
    if cached_model is not None:
        return cached_model

    model_path = smpl_model_path(gender)
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL model file not found: {model_path}")

    smpl_model = SMPL(load_numpy_params(model_path))
    _loaded_models_by_gender[gender] = smpl_model
    return smpl_model
