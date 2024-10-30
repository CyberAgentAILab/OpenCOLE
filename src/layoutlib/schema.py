import random
from functools import partial
from pathlib import Path
from typing import Any

import datasets as ds
import yaml  # type: ignore
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    create_model,
    validator,
)

from layoutlib.util import mock_string

# note: we dynamically generate pydantic models.
# these two are just for avoiding type errors.
# TODO: use typing.TypeAlias when we drop support for 3.9
Layout = BaseModel
Element = BaseModel


class NumericalAttribute(BaseModel):  # type: ignore
    """
    Define how each attribute in an element is tokenized.
    """

    quantization: str | None = None
    num_bin: int | None = None
    # note1: no support on multi-dimensional array
    # note2: specify as list in yaml to avoid error
    shape: tuple[int] = (1,)
    vmin: float = 0.0
    vmax: float = 1.0

    @validator("quantization")
    def check_quantization_vocab(cls, field):  # type: ignore
        assert field in ["linear", "kmeans", None], field
        return field

    @validator("num_bin")
    def check_num_bin(cls, field):  # type: ignore
        assert field > 0, field
        return field

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.quantization is not None:
            assert self.num_bin is not None, self.num_bin


class ElementSchema(BaseModel):  # type: ignore
    """
    Define how each element in a layout is tokenized.
    """

    attribute_order: list[str] = []  # order of the attributes to be mentioned.
    string_attributes: list[str] = []  # ds.Value(dtype='string')
    numerical_attributes: dict[str, NumericalAttribute] = {}  # ds.Value(dtype='float')
    categorical_attributes: list[str] = []  # ds.ClassLabel (handled as int)

    @validator("attribute_order")
    def check_num_bin(cls, field):  # type: ignore
        assert len(field) > 0, field
        return field

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # check key match
        set_out = set(self.attribute_order)
        set_tgt = self.string_keys | self.numerical_keys | self.categorical_keys
        assert set_out == set_tgt, (
            set_out,
            set_tgt,
        )

    @staticmethod
    def load_from_yaml(path: str) -> "ElementSchema":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return ElementSchema(**config)

    @property
    def string_keys(self) -> set[str]:
        return set(self.string_attributes)

    @property
    def numerical_keys(self) -> set[str]:
        return set(self.numerical_attributes.keys())

    @property
    def categorical_keys(self) -> set[str]:
        return set(self.categorical_attributes)


ELEMENT_SCHEMA_ROOT = Path(__file__).parent / "config" / "element"
ELEMENT_SCHEMA_NAMES = [p.stem for p in ELEMENT_SCHEMA_ROOT.glob("*.yaml")]


def element_schema_factory(name: str | None = None) -> ElementSchema:
    """
    Generate a layout schema from a name if provided, otherwise randomly pick one.
    """
    if name is None:
        name = random.choice(ELEMENT_SCHEMA_NAMES)
    return ElementSchema.load_from_yaml(f"{ELEMENT_SCHEMA_ROOT}/{name}.yaml")


DESCRIPTION = {"text": "Dummy text"}


def description_factory(key: str) -> str:
    return DESCRIPTION.get(key, "")


def get_element_pydantic_model(
    schema: ElementSchema,
    class_label_mappings: dict[str, ds.ClassLabel] = {},
    skip_showing_choices: list[str] = [],
) -> Element:  # type: ignore
    """
    Generate a pydantic class that is passed as pydantic_object to PydanticOutputParser.
    Fields are dynamically generated based on the tokenizer,
    since we want to know the data type **after** the conversion.
    description: used to enrich description for parser.get_format_instructions()
    validator: used to check the format violation for parser.parse()
    """

    assert all([key in class_label_mappings for key in schema.categorical_keys]), (
        class_label_mappings,
        schema.categorical_keys,
    )
    assert all([key in class_label_mappings for key in skip_showing_choices]), (
        class_label_mappings,
        skip_showing_choices,
    )

    # class_names_ignore_keys = [
    #     "font",
    # ]  # TODO: generalize the rule
    pydantic_kwargs, validators = {}, {}
    for key in schema.attribute_order:
        description = description_factory(key)
        type_ = str

        if key in schema.categorical_keys:
            class_names = class_label_mappings[key].names

            # in case there are too many class names, we can avoid listing the names
            # TODO: should we always include ""?
            if key in skip_showing_choices:
                description += ""
            else:
                description += "choices: " + ", ".join(
                    [f'"{name}"' for name in class_names]
                )

            validators[f"{key}_validator"] = validator(key, allow_reuse=True)(
                partial(_is_categorical_key, class_names=class_names, key=key)
            )

        elif key in schema.numerical_keys:
            column: NumericalAttribute = schema.numerical_attributes[key]
            if column.quantization is None:
                description += f"range: {column.vmin} <= {key} <= {column.vmax}"
                type_ = float  # type: ignore
                validators[f"{key}_validator"] = validator(key, allow_reuse=True)(
                    partial(
                        _is_float_in_reasonable_range,
                        key=key,
                        vmin=column.vmin,
                        vmax=column.vmax,
                    )
                )
            else:
                vmin, vmax = 0, column.num_bin - 1  # type: ignore
                description += f"range: {vmin} <= {key} <= {vmax}"
                type_ = int  # type: ignore
                validators[f"{key}_validator"] = validator(key, allow_reuse=True)(
                    partial(_is_valid_for_bucketizer, key=key, vmin=vmin, vmax=vmax)
                )

        field = Field(..., description=description)  # note: ... means requried
        pydantic_kwargs[key] = (type_, field)

    pydantic_kwargs["__validators__"] = validators  # type: ignore

    return create_model("Element", **pydantic_kwargs)  # type: ignore


def get_layout_pydantic_model(
    schema: ElementSchema,
    class_label_mappings: dict[str, ds.ClassLabel] = {},
    skip_showing_choices: list[str] = [],
) -> Layout:  # type: ignore
    """
    Dynamically generate a pydantic class for layout.
    The class is equal to the following:

    class Layout(BaseModel):
        elements: list[Element] = []
    """
    # TODO: support canvas information
    Element = get_element_pydantic_model(
        schema=schema,
        class_label_mappings=class_label_mappings,
        skip_showing_choices=skip_showing_choices,
    )
    return create_model("Layout", elements=(list[Element], []))  # type: ignore


def _is_categorical_key(cls: object, v: str, key: str, class_names: list[str]) -> str:
    assert v in class_names, f"{v} should be in {class_names} for {key}"
    return v


def _is_valid_for_bucketizer(
    cls: object, v: int, key: str, vmin: int, vmax: int
) -> int:
    assert vmin <= v <= vmax, f"{v} should be in [{vmin}, {vmax}] for {key}"
    return v


def _is_float_in_reasonable_range(
    cls: object, v: float, key: str, vmin: float, vmax: float
) -> float:
    assert vmin <= v <= vmax, f"{v} should be in [{vmin}, {vmax}] for {key}"
    return v


def mock_element(schema: ElementSchema, features: ds.Features) -> dict[str, Any]:
    layout = {}
    for key in schema.attribute_order:
        if key in schema.numerical_keys:
            attribute = schema.numerical_attributes[key]
            vmin, vmax = attribute.vmin, attribute.vmax
            layout[key] = vmin + (vmax - vmin) * random.random()
        elif key in schema.categorical_keys:
            layout[key] = random.randint(0, features[key].num_classes - 1)
        elif key in schema.string_keys:
            # string variables
            layout[key] = mock_string()  # type: ignore
        else:
            raise NotImplementedError
    return layout


def mock_element_features(schema: ElementSchema, max_size: int = 10) -> ds.Features:
    features = ds.Features.from_dict({})
    for attribute in schema.categorical_attributes:
        names = [mock_string(max_size) for _ in range(random.randint(1, max_size))]
        features[attribute] = ds.ClassLabel(names=names)

    for attribute in schema.string_attributes:
        features[attribute] = ds.Value(dtype="string")

    for attribute in schema.numerical_attributes:
        features[attribute] = ds.Value(dtype="float32")

    return features
