import datasets as ds
import PIL
from matplotlib import pyplot as plt
from pinjected import Injected, injected, instance


@instance
def crello_dataset_v4_0():
    dataset_dict = ds.load_dataset("cyberagent/crello", revision="v4.0.0")
    return dataset_dict


crello_v4_0_train_sample: Injected = crello_dataset_v4_0["train"][0]
crello_v4_0_train_sample_preview = crello_v4_0_train_sample["preview"]


@injected
async def show_image(img: PIL.Image):
    plt.imshow(img)
    plt.show()
    return img


show_train_sample: Injected = show_image(crello_v4_0_train_sample)
