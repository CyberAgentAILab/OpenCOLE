import itertools
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import fsspec

from opencole.renderer.utils import read_json_records

logger = logging.getLogger(__name__)

FontDict = Dict[str, List[Dict[str, Any]]]


FONT_WEIGHTS = {
    "thin",
    "light",
    "extralight",
    "regular",
    "medium",
    "semibold",
    "bold",
    "extrabold",
    "black",
}

FONT_STYLES = {"regular", "bold", "italic", "bolditalic"}


class FontManager(object):
    """Font face manager.

    Example::

        # Use the font manager to lookup a font face.
        fm = FontManager("/path/to/fonts.pickle")
        typeface = skia.Typeface.MakeFromData(fm.lookup("Montserrat"))

        # Build a font dictionary from font metadata files.
        fm = FontManager()
        fm.build("/path/to/fonts/*.json", "/path/to/assets")
        fm.save("/path/to/fonts.pickle")
    """

    def __init__(self, input_path: Optional[str] = None) -> None:
        self._fonts: Optional[FontDict] = None

        if input_path is not None:
            self.load(input_path)

    def build(self, file_pattern: str, assets_path: str) -> None:
        """Build fonts dictionary from font metadata files."""
        self._fonts = _load_fonts(file_pattern)
        for family in self._fonts.values():
            for font in family:
                font["bytes"] = _load_fontfile(font, assets_path)
        logger.info("Loaded %d font families", len(self._fonts))

    def save(self, output_path: str) -> None:
        """Save fonts to a pickle file."""
        assert self._fonts is not None, "Fonts not loaded yet."
        logger.info("Saving fonts to %s", output_path)
        with fsspec.open(output_path, "wb") as f:
            pickle.dump(self._fonts, f)

    def load(self, input_path: str) -> None:
        """Load fonts from a pickle file."""
        logger.info("Loading fonts from %s", input_path)
        with fsspec.open(input_path, "rb") as f:
            self._fonts = pickle.load(f)
        assert self._fonts is not None, "No font loaded."
        logger.info("Loaded %d  font families", len(self._fonts))

    def lookup(
        self,
        font_family: str,
        font_weight: str = "regular",
        font_style: str = "regular",
    ) -> bytes:
        """Lookup the specified font face."""
        assert self._fonts is not None, "Fonts not loaded yet."
        assert font_weight in FONT_WEIGHTS, f"Invalid font weight: {font_weight}"
        assert font_style in FONT_STYLES, f"Invalid font style: {font_style}"

        family = []
        for i, family_name in enumerate([font_family, "Montserrat"]):
            try:
                family = self._fonts[normalize_family(family_name)]
                if i > 0:
                    logger.warning(
                        f"Font family fallback to {family[0]['fontFamily']}."
                    )
                break
            except KeyError:
                logger.warning(f"Font family not found: {font_family}")

        if not family:
            family = next(iter(self._fonts.values()))
            logger.warning(f"Font family fallback to {family[0]['fontFamily']}.")

        font_weight = get_font_weight(font_family, font_weight)
        font_style = get_font_style(font_family, font_style)
        try:
            font = next(
                font
                for font in family
                if font.get("fontWeight", "regular") == font_weight
                and font.get("fontStyle", "regular") == font_style
            )
        except StopIteration:
            font = family[0]
            logger.warning(
                f"Font style for {font['fontFamily']} not found: {font_weight} "
                f"{font_style}, fallback to {font.get('fontWeight', 'regular')} "
                f"{font.get('fontStyle', 'regular')}"
            )
        return font["bytes"]


def _load_fonts(file_pattern: str) -> FontDict:
    """Read font metadata."""

    def _sort_key(font: Dict[str, Any]) -> int:
        """Sort fonts by weight and style."""
        return len(FONT_STYLES) * (font.get("fontWeight", "regular") == "regular") + (
            font.get("fontStyle", "regular") == "regular"
        )

    logger.info("Loading fonts from %s", file_pattern)
    return {
        family: sorted(it, key=_sort_key, reverse=True)
        for family, it in itertools.groupby(
            read_json_records(file_pattern),
            key=lambda f: normalize_family(f["fontFamily"]),
        )
    }


def _load_fontfile(font: Dict[str, Any], assets_path: str) -> bytes:
    """Load font file content."""
    path = os.path.join(assets_path, font["mediaId"])
    logger.debug(f"Loading {path}")
    with fsspec.open(path) as f:
        return f.read()


def normalize_family(name: str) -> str:
    """Normalize font name."""
    name = name.replace("_", " ").title()
    name = name.replace(" Bold", "")
    name = name.replace(" Regular", "")
    name = name.replace(" Light", "")
    name = name.replace(" Italic", "")
    name = name.replace(" Medium", "")

    FONT_MAP = {
        "Arkana Script": "Arkana",
        "Blogger": "Blogger Sans",
        "Delius Swash": "Delius Swash Caps",
        "Elsie Swash": "Elsie Swash Caps",
        "Gluk Glametrix": "Gluk Foglihtenno06",
        "Gluk Znikomitno25": "Gluk Foglihtenno06",
        "Im Fell": "Im Fell Dw Pica Sc",
        "Medieval Sharp": "Medievalsharp",
        "Playlist Caps": "Playlist",
        "Rissa Typeface": "Rissatypeface",
        "Selima": "Selima Script",
        "Six": "Six Caps",
        "V T323": "Vt323",
        # The rest is unknown.
        "Different Summer": "Montserrat",
        "Dukomdesign Constantine": "Montserrat",
        "Sunday": "Montserrat",
    }
    return FONT_MAP.get(name, name)


def get_font_weight(name: str, default: str) -> str:
    """Get font weight from the font name."""
    key = name.replace("_", " ").lower().split(" ")[-1]
    return key if key in FONT_WEIGHTS else default


def get_font_style(name: str, default: str) -> str:
    """Get font style from the font name."""
    key = name.replace("_", " ").lower().split(" ")[-1]
    return key if key in FONT_STYLES else default
