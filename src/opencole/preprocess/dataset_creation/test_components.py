from pinjected import Injected, instances, providers

from opencole.preprocess.dataset_creation.crello_instances import (
    crello_v4_0_train_sample,
    crello_v4_0_train_sample_preview,
)
from opencole.preprocess.dataset_creation.image_to_detail import a_image_to_detail
from opencole.preprocess.dataset_creation.image_to_intention import test_ig_cxt
from opencole.preprocess.dataset_creation.vision_llm_llava import a_vision_llm__llava

test_llava: Injected = a_vision_llm__llava(
    text="<image> what do you see in this image? \n ASSISTANT:",
    images=Injected.list(crello_v4_0_train_sample_preview),
)

test_image_to_detail: Injected = a_image_to_detail(crello_v4_0_train_sample)
test_crello_to_intention: Injected = test_ig_cxt

__meta_design__ = instances(
    overrides=providers(
        # a_vision_llm=a_vision_llm__llava,
        # a_vision_llm=a_vision_llm__gpt4
    )
)
