import dataclasses
import json
from enum import Enum, auto
from typing import Any, Callable, List

import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)

from layoutlib.schema import Layout


def pos_converter(input_: float, ratio: float) -> float:
    return (input_ - 0.5) / ratio + 0.5


def scale_converter(input_: float, ratio: float) -> float:
    return input_ / ratio


def padding_layout_transform(
    inputs: dict,
    canvas_width: int,
    canvas_height: int,
    method: str = "to_square",
) -> dict:
    """
    Non-square images are padded to be square. This transform is to adjust layout for the padding.
    https://github.com/haotian-liu/LLaVA/blob/7775b12d6b20cd69089be7a18ea02615a59621cd/llava/mm_utils.py#L14-L25
    """
    # assert _is_row_like(inputs)
    assert method in ["to_square", "from_square"]

    if canvas_width == canvas_height:
        return inputs
    elif canvas_width > canvas_height:
        # extend / shrink along y-axis
        ratio = canvas_width / canvas_height
        if method == "from_square":
            ratio = 1.0 / ratio

        for key in inputs:
            if key in ["top", "center_y", "bottom"]:
                inputs[key] = [pos_converter(z, ratio) for z in inputs[key]]
            if key in ["height"]:
                inputs[key] = [scale_converter(z, ratio) for z in inputs[key]]
    else:
        # extend / shrink along x-axis
        ratio = canvas_height / canvas_width
        if method == "from_square":
            ratio = 1.0 / ratio

        for key in inputs:
            if key in ["left", "center_x", "right"]:
                inputs[key] = [pos_converter(z, ratio) for z in inputs[key]]
            if key in ["width"]:
                inputs[key] = [scale_converter(z, ratio) for z in inputs[key]]
    return inputs


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str | None = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self) -> str:
        messages: List[List[str]] = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:  # type: ignore
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if "mmtag" in self.version:
                messages[0] = (init_role, init_msg)  # type: ignore
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))  # type: ignore
                messages.insert(1, (self.roles[1], "Received."))  # type: ignore
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)  # type: ignore

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:  # type: ignore
                        message, _, _ = message  # type: ignore
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]  # type: ignore
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:  # type: ignore
                        message, _, _ = message  # type: ignore
                    ret += role + ": " + message + seps[i % 2]  # type: ignore
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:  # type: ignore
                        message, _, _ = message  # type: ignore
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:

            def wrap_sys(msg: str) -> str:
                return f"<<SYS>>\n{msg}\n<</SYS>>\n\n"

            def wrap_inst(msg: str) -> str:
                return f"[INST] {msg} [/INST]"

            ret = ""
            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:  # type: ignore
                        message, _, _ = message  # type: ignore
                    if i == 0:
                        message = wrap_sys(self.system) + message  # type: ignore
                    if i % 2 == 0:
                        message = wrap_inst(message)  # type: ignore
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2  # type: ignore
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:  # type: ignore
                        message, _, _ = message  # type: ignore
                    ret += message + seps[i % 2]  # type: ignore
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role: str, message: str) -> None:
        self.messages.append([role, message])

    def get_images(self, return_pil: bool = False) -> list[Image.Image | str]:
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:  # type: ignore
                    import base64
                    from io import BytesIO

                    from PIL import Image

                    msg, image, image_process_mode = msg  # type: ignore
                    if image_process_mode == "Pad":  # type: ignore

                        def expand2square(
                            pil_img: Image.Image,
                            background_color: tuple[int, int, int] = (122, 116, 104),
                        ) -> Image:
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(
                                    pil_img.mode, (width, width), background_color
                                )
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(
                                    pil_img.mode, (height, height), background_color
                                )
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image)  # type: ignore
                    elif image_process_mode in ["Default", "Crop"]:  # type: ignore
                        pass
                    elif image_process_mode == "Resize":  # type: ignore
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(
                            f"Invalid image_process_mode: {image_process_mode}"  # type: ignore
                        )
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if longest_edge != max(image.size):
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self) -> list[list[str | None]]:
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:  # type: ignore
                    import base64
                    from io import BytesIO

                    msg, image, image_process_mode = msg  # type: ignore
                    max_hw, min_hw = max(image.size), min(image.size)  # type: ignore
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size  # type: ignore
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))  # type: ignore
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace("<image>", "").strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self) -> "Conversation":
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self) -> dict:
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [
                    [x, y[0] if type(y) is tuple else y]
                    for x, y in self.messages  # type: ignore
                ],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


def validate_text_placement(
    layout: Layout,
    prompt: str,
    trigger: str = "Input:",
    **kwargs: Any,
) -> None:
    # extract user-given keywords
    start = prompt.index(trigger) + len(trigger)
    end = prompt.index(" ASSISTANT")
    user_input = json.loads(prompt[start:end])
    keywords_input = [x for x in user_input]
    keywords_input_set = set(keywords_input)

    keywords_output = [getattr(e, "text") for e in layout.elements]  # type: ignore
    keywords_output_set = set(keywords_output)

    assert len(keywords_output) == len(
        keywords_input
    ), f"Length mismatch, got {len(keywords_output)}, expected {len(keywords_input)}"
    assert (
        keywords_output_set == keywords_input_set
    ), f"Set mismatch, got {keywords_output_set}, expected {keywords_input_set}"


_LLAVA_OUTPUT_VALIDATOR_FACTORY = {
    "text_placement": validate_text_placement,
}


def llava_output_validator_factory(name: str) -> Callable:
    assert name in _LLAVA_OUTPUT_VALIDATOR_FACTORY, f"Unknown parser name: {name}"
    return _LLAVA_OUTPUT_VALIDATOR_FACTORY[name]


def llava_output_validator_factory_names() -> list[str]:
    return list(_LLAVA_OUTPUT_VALIDATOR_FACTORY.keys())


def load_llava(
    pretrained_model_name_or_path: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> tuple[LlavaForConditionalGeneration, LlavaProcessor]:
    assert not (load_in_4bit and load_in_8bit), "Cannot load both 4bit and 8bit models"
    kwargs: dict[str, Any] = {"device_map": "auto"}
    kwargs["torch_dtype"] = torch.float16
    kwargs["low_cpu_mem_usage"] = True
    kwargs["use_safetensors"] = True
    kwargs["offload_state_dict"] = False
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    kwargs["quantization_config"] = bnb_config

    model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path, **kwargs
    )
    processor = LlavaProcessor.from_pretrained(pretrained_model_name_or_path)
    return model, processor
