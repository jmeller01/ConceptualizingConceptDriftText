# ConceptualizingConceptDriftText
Sensitivity analysis and text-domain. transfer of the concept-based explanation pipeline.

Code and results for the bachelor's thesis *Conceptualizing Concept Drift:
Transferring Concept-Based Drift Explanations to Text Domain* (Bielefeld
University, 2026). The thesis takes the Concept² Drift Distribution pipeline
of Roberts et al. (2025) and applies it to text: a fine-tuned BERT model with
a ReLU pooler as embedding, sliding-window patches instead of image crops, and
occlusion keywords to describe the concepts.

## Layout

```
concept_helpers/      Roberts et al., unchanged
experiment_helpers/   Roberts et al., one line changed in helper_function.py
text_helpers/         own code: CustomBertModel.py, CraftText.py
notebooks/images/     reproduction and ablations on D1, D2 and Fish-Head
notebooks/text/       main experiment, ablations, case studies, gradual drift
```

The CSVs behind the tables of the thesis lie in a `results/` folder next to
each notebook; the gradual drift CSV lies next to its notebook and
`relu_substitution_accuracy.csv` in `notebooks/text/`.


## Running

The notebooks were run in Google Colab (GPU runtime) and expect this folder
under `MyDrive/ConceptualizingConceptDrift/` plus an empty folder
`MyDrive/results/` for the CSVs. The first cell of each notebook installs
`xplique`, `timm`, `opencv-python` and `wordcloud` the rest comes with Colab.

Data:

- Text datasets and the fine-tuned BERT checkpoints are loaded from the
  Hugging Face Hub inside the notebooks.
- NINCO (`MyDrive/data/ninco/NINCO/NINCO_OOD_classes`, zenodo record 8013288)
  and the ImageNet subset (`MyDrive/data/imageNet/imagenet_images` class list
  in the README of Roberts et al.) and `fish_head_embedding.npz` (the D3 notebook expects it under  `MyDrive/data/fishHead/`) have to be downloaded separately.
 

## License

MIT for the code in this repository; see `LICENSE` for the parts that come
from Roberts et al., COCKATIEL and Hinder et al.
