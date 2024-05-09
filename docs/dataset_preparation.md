[TOC]

# Image to Detail Creation
Please use this to create a dataset of details from images.
```bash
python -m pinjected run opencole.preprocess.dataset_creation.a_pinjected_store_all_details_in_dir \
    --images-root="root_dir/to/source/images" \
    --dst-root="path/to/save/details" 
```

# Image to Intention Creation
AsyncIterator[tuple[str,IntentionGenerationContext]] can be used with "store_all_intentions".
Here, we use the "opencole.preprocess.dataset_creation.image_to_intention.crello_v4_0_id_cxt_pairs" for generating intentions.

```bash
python -m pinjected run opencole.preprocess.dataset_creation.store_all_intentions \
  --id-cxt-pairs="{opencole.preprocess.dataset_creation.image_to_intention.crello_v4_0_id_cxt_pairs}" \
  --dst-root="path/to/save/intentions" 
```

For more details about `python -m pinjected`, please refer to [pinjected](https://github.com/proboscis/pinjected) and [CLI support](https://github.com/proboscis/pinjected?tab=readme-ov-file#example-cli-calls) section.