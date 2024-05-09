import html
import logging
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


Numeric = Union[int, float]


class AnimationProp(BaseModel):
    """Animation properties."""

    delay: Numeric
    type: str
    direction: Optional[str] = None
    appearanceType: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class DesignElement(BaseModel):
    """Base class for all visual elements."""

    uuid: str
    type: str
    width: Numeric = Field(gt=0.0)
    height: Numeric = Field(gt=0.0)
    angle: Numeric = 0
    left: Numeric = 0  # Required, but not always present.
    top: Numeric = 0  # Required, but not always present.
    opacity: Numeric = Field(default=1.0, ge=0.0, le=1.0)
    locked: Optional[bool] = None
    lockMode: Optional[Literal["Move", "Full", "TextEditOnly"]] = None
    isFreeItem: Optional[bool] = None
    animations: Optional[List[Any]] = None
    animationDuration: Optional[Numeric] = None
    templateAsset: Optional[bool] = None
    forSubscribers: Optional[bool] = None
    isUnlimitedPlus: Optional[bool] = None
    animationProps: Optional[AnimationProp] = None
    position: Optional[Dict[str, Numeric]] = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def check_position(self) -> "DesignElement":
        """Check that position is set."""
        if self.position is not None:
            if self.position["left"] != self.left:
                self.left = self.position["left"]
            if self.position["top"] != self.top:
                self.top = self.position["top"]
        else:
            self.position = dict(left=self.left, top=self.top)
        return self


class TextMapItem(BaseModel):
    """Map item for text properties."""

    startIndex: int
    endIndex: int
    value: Optional[Union[bool, str, Numeric]] = None
    type: Optional[str] = None

    # TODO: Define each map item type.


class BaseTextEffect(BaseModel):
    """Base text effect."""

    type: str
    enabled: bool

    model_config = ConfigDict(extra="allow")


class TextBoxTextEffect(BaseTextEffect):
    """Text box text effect."""

    type: Literal["textBox"]
    color: str
    blendOpacity: Numeric
    spread: Numeric = 0
    radius: Numeric = 0


class CurvedTextEffect(BaseTextEffect):
    """Curved text effect."""

    type: Literal["curved"]
    angle: Numeric
    baseWidth: Optional[Numeric] = None


class OutlineTextEffect(BaseTextEffect):
    """Outline text effect."""

    type: Literal["outline"]
    color: str
    opacity: Numeric
    thickness: Numeric
    hasInnerText: bool


class DropShadowTextEffect(BaseTextEffect):
    """Drop shadow text effect."""

    type: Literal["dropShadow"]
    color: str
    opacity: Numeric
    blendOpacity: Numeric
    size: Numeric
    blur: Numeric
    distance: Numeric
    angle: Numeric


class EchoTextEffect(BaseTextEffect):
    """Echo text effect."""

    type: Literal["echo"]
    distance: Numeric
    angle: Numeric


class GlitchTextEffect(BaseTextEffect):
    """Glitch text effect."""

    type: Literal["glitch"]
    size: Numeric
    angle: Numeric


class ReflectionTextEffect(BaseTextEffect):
    """Reflection text effect."""

    type: Literal["reflection"]
    size: Numeric
    distance: Numeric
    transparency: Numeric


TextEffect = Union[
    TextBoxTextEffect,
    CurvedTextEffect,
    OutlineTextEffect,
    DropShadowTextEffect,
    EchoTextEffect,
    GlitchTextEffect,
    ReflectionTextEffect,
]


class TextProperty(BaseModel):
    """Properties for text elements."""

    text: str
    fontSize: Numeric = Field(gt=0.0)
    font: str
    lineHeight: Numeric = Field(gt=0.0)
    textAlign: Literal["left", "right", "center", "justify"]
    capitalize: bool
    letterSpacing: Numeric
    colorMap: List[TextMapItem]
    wordBreak: Optional[Literal["breakWord", "breakAll"]] = None
    underline: Optional[bool] = None
    boldMap: Optional[List[TextMapItem]] = None
    italicMap: Optional[List[TextMapItem]] = None
    underlineMap: Optional[List[TextMapItem]] = None
    linkMap: Optional[List[TextMapItem]] = None
    lineMap: Optional[List[TextMapItem]] = None
    opacityMap: Optional[List[TextMapItem]] = None
    weightMap: Optional[List[TextMapItem]] = None
    styleMap: Optional[List[TextMapItem]] = None
    effects: Optional[List[TextEffect]] = None

    @field_validator("text")
    @classmethod
    def unescape(cls, value: str) -> str:
        return html.unescape(value)

    @model_validator(mode="after")
    def check_linemap(self) -> "TextProperty":
        if self.lineMap is None:
            self.lineMap = [
                TextMapItem(
                    startIndex=0, endIndex=len(self.text) - 1, value="line", type="line"
                )
            ]
        return self


class TextElement(DesignElement, TextProperty):
    """Text element."""

    type: Literal["textElement"]
