import json
import random
from typing import Any, TypeVar

from langchain_core.pydantic_v1 import BaseModel


class _BaseModel(BaseModel):  # type: ignore
    @staticmethod
    def get_mock() -> "_BaseModel":
        raise NotImplementedError


class Captions(_BaseModel):  # type: ignore
    background: str
    objects: str

    @staticmethod
    def get_mock() -> "Captions":
        return Captions(background="dummy", objects="dummy")


class Headings(_BaseModel):  # type: ignore
    heading: list[str] = []
    sub_heading: list[str] = []

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        assert len(self.heading) > 0 or len(self.sub_heading) > 0, (
            self.heading,
            self.sub_heading,
        )

    @staticmethod
    def get_mock() -> "Headings":
        return Headings(
            heading=[
                "dummy",
            ],
            sub_heading=[
                "dummy",
            ],
        )


class Detail(_BaseModel):  # type: ignore
    """A base class for defining the interface and type hints"""

    def serialize_t2i_input(self) -> str:
        """Convert the instance to a string for the input of the text-to-image model."""
        raise NotImplementedError

    def serialize_tlmm_input(self, shuffle: bool = True) -> str:
        """
        Convert the instance to a string for the input of the TypographyLMM model.
        note: shuffling is important to avoid the element order leakage.
        """
        raise NotImplementedError

    def to_hfds(self) -> dict[str, Any]:
        """Convert the instance to an HFDS format instance (dict-like format)."""
        raise NotImplementedError

    @staticmethod
    def from_hfds(self) -> "Detail":
        """Convert an HFDS format instance (dict-like format) to the instance."""
        raise NotImplementedError

    @staticmethod
    def non_strict_load(inputs: dict[str, Any]) -> "DetailV1":
        """If some values are missing, just fill that part by dummy values."""
        raise NotImplementedError


ChildOfDetail = TypeVar("ChildOfDetail", bound=Detail)  # to specify subclass of Detail


class DetailV1(Detail):  # type: ignore
    description: str
    keywords: list[str] = []
    captions: Captions
    headings: Headings

    @staticmethod
    def get_mock() -> "DetailV1":
        return DetailV1(
            description="dummy",
            keywords=[
                "dummy",
            ],
            captions=Captions.get_mock(),
            headings=Headings.get_mock(),
        )

    @staticmethod
    def non_strict_load(inputs: dict[str, Any]) -> "DetailV1":
        """
        If some values are missing, just fill that part by dummy values.
        """
        if "description" not in inputs or not isinstance(inputs["description"], str):
            inputs["description"] = ""
        if "keywords" not in inputs or not isinstance(inputs["keywords"], list):
            inputs["keywords"] = []

        if "captions" not in inputs:
            inputs["captions"] = Captions.get_mock()
        else:
            captions = inputs["captions"]
            if "background" not in captions:
                captions["background"] = "dummy"
            else:
                captions["background"] = _convert_to_pure_str(captions["background"])
            if "objects" not in captions:
                captions["objects"] = "dummy"
            else:
                captions["objects"] = _convert_to_pure_str(captions["objects"])

        if "headings" not in inputs:
            inputs["headings"] = Headings.get_mock()
        else:
            headings = inputs["headings"]
            if (
                "heading" not in headings
                or headings["heading"] is None
                or len(headings["heading"]) == 0
            ):
                headings["heading"] = ["dummy"]
            else:
                headings["heading"] = [
                    _convert_to_pure_str(h) for h in headings["heading"]
                ]

            if "sub-heading" in headings:
                headings["sub_heading"] = headings["sub-heading"]
                del headings["sub-heading"]
            if (
                "sub_heading" not in headings
                or headings["sub_heading"] is None
                or len(headings["heading"]) == 0
            ):
                headings["sub_heading"] = ["dummy"]
            else:
                headings["sub_heading"] = [
                    _convert_to_pure_str(h) for h in headings["sub_heading"]
                ]

        return DetailV1(**inputs)

    def serialize_t2i_input(self) -> str:
        return f"{self.captions.background} {self.captions.objects}"

    def serialize_tlmm_input(self, shuffle: bool = True) -> str:
        texts = self.headings.heading + self.headings.sub_heading
        if shuffle:
            texts = random.sample(texts, len(texts))
        return json.dumps(texts)

    def serialize_tlmm_context(self) -> str:
        return self.json(exclude={"headings"})

    @staticmethod
    def from_hfds(hfds: dict[str, Any]) -> "DetailV1":
        return DetailV1(
            keywords=hfds["keywords"],
            description=hfds["description"],
            captions=Captions(
                background=hfds["captions_background"],
                objects=hfds["captions_objects"],
            ),
            headings=Headings(
                heading=hfds["headings_heading"],
                sub_heading=hfds["headings_sub_heading"],
            ),
        )

    def to_hfds(self) -> dict[str, Any]:
        return {
            "keywords": self.keywords,
            "description": self.description,
            "captions_background": self.captions.background,
            "captions_objects": self.captions.objects,
            "headings_heading": self.headings.heading,
            "headings_sub_heading": self.headings.sub_heading,
        }


def _remove_special_chars(text: str) -> str:
    return text.replace("{", " ").replace("}", " ")


def _convert_to_pure_str(data: Any) -> str:
    if isinstance(data, str):
        return data
    else:
        return _remove_special_chars(str(data))


class Intention(_BaseModel):  # type: ignore
    pass


ChildOfIntention = TypeVar(
    "ChildOfIntention", bound=Intention
)  # to specify subclass of Detail


class IntentionV1(Intention):  # type: ignore
    content: str
    # use pydantic object just in case, since this object might be extended in the future

    @staticmethod
    def get_mock() -> "IntentionV1":
        return IntentionV1(content="dummy")
