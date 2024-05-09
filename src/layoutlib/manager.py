import pickle
from copy import deepcopy
from typing import Optional

import datasets as ds
import fsspec
import numpy as np
from sklearn.cluster import KMeans

from layoutlib.bucketizer import BaseBucketizer, bucketizer_factory
from layoutlib.hfds.util import ListOnlyExample
from layoutlib.schema import (
    Element,
    ElementSchema,
    Layout,
    element_schema_factory,
    get_element_pydantic_model,
    get_layout_pydantic_model,
)
from layoutlib.util import (
    dict_of_list_to_list_of_dict,
    list_of_dict_to_dict_of_list,
)


class LayoutManager:
    """
    Represent layout in different formats, following the schema.
    """

    def __init__(
        self,
        schema_name: Optional[str] = None,
        schema: Optional[ElementSchema] = None,
        weight_path: Optional[str] = None,
        class_label_mappings: dict[str, ds.ClassLabel] = {},
    ) -> None:
        if schema is not None:
            self._schema = schema
        else:
            assert schema_name is not None
            self._schema = element_schema_factory(schema_name)
        self._class_element = get_element_pydantic_model(
            schema=self._schema, class_label_mappings=class_label_mappings
        )
        self._class_layout = get_layout_pydantic_model(
            schema=self._schema, class_label_mappings=class_label_mappings
        )

        # note: these values are not overridden. To get values, please use getter.
        self._class_label_mappings = class_label_mappings

        self._num_vocabularies = {}
        for key in self._schema.categorical_attributes:
            self._num_vocabularies[key] = len(self._class_label_mappings[key].names)

        for value in self._schema.numerical_attributes.values():
            if value.quantization in [
                "kmeans",
            ]:
                assert weight_path is not None
                fs, path_prefix = fsspec.core.url_to_fs(weight_path)
                with fs.open(path_prefix, "rb") as f:
                    weights: dict[str, KMeans] = pickle.load(f)
                break

        self._bucketizers = {}
        for key, value in self._schema.numerical_attributes.items():
            num_bin, quantization = value.num_bin, value.quantization
            vmin, vmax = value.vmin, value.vmax
            if quantization is None:
                continue

            bucketizer_kwargs = {
                "num_bin": num_bin,
                "vmin": vmin,
                "vmax": vmax,
                "key": key,
            }
            if quantization == "linear":
                self._num_vocabularies[key] = num_bin  # type: ignore
            elif quantization == "kmeans":
                bucketizer_kwargs["model"] = weights[f"{key}-{num_bin}"]

            self._bucketizers[key] = bucketizer_factory(quantization)(
                **bucketizer_kwargs
            )

    def tokenize_attributes(
        self,
        inputs: ListOnlyExample,
    ) -> ListOnlyExample:
        outputs = {}
        for key in self.schema.attribute_order:
            if key in self.schema.categorical_keys:
                feature = self._class_label_mappings[key]
                outputs[key] = [feature.int2str(ind) for ind in inputs[key]]
            elif key in self.schema.numerical_keys:
                if key in self.bucketizers:
                    outputs[key] = (
                        self.bucketizers[key].encode(np.array(inputs[key])).tolist()  # type: ignore
                    )
                else:
                    outputs[key] = deepcopy(inputs[key])
            else:
                outputs[key] = deepcopy(inputs[key])
        return outputs

    def detokenize_attributes(self, inputs: ListOnlyExample) -> ListOnlyExample:
        outputs = {}
        for key in self.schema.attribute_order:
            if key in self.schema.categorical_keys:
                feature = self._class_label_mappings[key]
                outputs[key] = [feature.str2int(x) for x in inputs[key]]
            elif key in self.schema.numerical_keys:
                if key in self.bucketizers:
                    outputs[key] = (
                        self.bucketizers[key].decode(np.array(inputs[key])).tolist()
                    )  # type: ignore
                else:
                    outputs[key] = deepcopy(inputs[key])
            else:
                outputs[key] = deepcopy(inputs[key])

        return outputs

    def hfds_to_layout_instance(
        self,
        hfds_inputs: ListOnlyExample,
    ) -> Layout:
        tokenized_inputs = self.tokenize_attributes(hfds_inputs)
        tokenized_inputs_ld = dict_of_list_to_list_of_dict(tokenized_inputs)
        elements = [self.class_element(**x) for x in tokenized_inputs_ld]
        return self.class_layout(elements=elements)

    def layout_instance_to_hfds(
        self,
        layout_instance: Layout,
    ) -> ListOnlyExample:
        inputs_dl = list_of_dict_to_dict_of_list(
            [e.dict() for e in layout_instance.elements]
        )  # type: ignore
        outputs = self.detokenize_attributes(inputs_dl)
        return outputs

    def name_to_length(self, name: str) -> int:
        return self._num_vocabularies[name]

    @property
    def class_element(self) -> type[Element]:
        return self._class_element  # type: ignore

    @property
    def class_layout(self) -> type[Layout]:
        return self._class_layout  # type: ignore

    @property
    def class_label_mappings(self) -> dict[str, ds.ClassLabel]:
        return self._class_label_mappings

    @property
    def bucketizers(self) -> dict[str, BaseBucketizer]:
        return self._bucketizers

    @property
    def schema(self) -> ElementSchema:
        return self._schema
