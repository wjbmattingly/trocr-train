# CATMuS Medieval Script Trainer

This project provides a training pipeline for models on the [CATMuS Medieval dataset](https://huggingface.co/datasets/CATMuS/medieval), which is a multilingual, multiscript medieval HTR dataset. The training portion comes [largely from this notebook.](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb) It has been modified to align with CATMuS.

## Installation

```bash
git clone https://github.com/wjbmattingly/catmus-train
cd catmus-train
```

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py --shuffle_seed 42 --select_range 100000 --batch_size 18 --epochs 10 --logging_steps 1000 --save_steps 1000 --save_limit 2 --device mps:0 --output_dir yiddish
```



### List of Pre-trained Models

1. [medieval-data/trocr-medieval-base](https://huggingface.co/medieval-data/trocr-medieval-base)
2. [medieval-data/trocr-medieval-latin-caroline](https://huggingface.co/medieval-data/trocr-medieval-latin-caroline)
3. [medieval-data/trocr-medieval-castilian-hybrida](https://huggingface.co/medieval-data/trocr-medieval-castilian-hybrida)
4. [medieval-data/trocr-medieval-humanistica](https://huggingface.co/medieval-data/trocr-medieval-humanistica)
5. [medieval-data/trocr-medieval-textualis](https://huggingface.co/medieval-data/trocr-medieval-textualis)
6. [medieval-data/trocr-medieval-cursiva](https://huggingface.co/medieval-data/trocr-medieval-cursiva)
7. [medieval-data/trocr-medieval-semitextualis](https://huggingface.co/medieval-data/trocr-medieval-semitextualis)
8. [medieval-data/trocr-medieval-praegothica](https://huggingface.co/medieval-data/trocr-medieval-praegothica)
9. [medieval-data/trocr-medieval-semihybrida](https://huggingface.co/medieval-data/trocr-medieval-semihybrida)
10. [medieval-data/trocr-medieval-print](https://huggingface.co/medieval-data/trocr-medieval-print)

## Citation

If you use the CATMuS Medieval dataset, please cite the following paper:

### BibTeX

```bibtex
@unpublished{clerice:hal-04453952,
  TITLE = {{CATMuS Medieval: A multilingual large-scale cross-century dataset in Latin script for handwritten text recognition and beyond}},
  AUTHOR = {Cl{\'e}rice, Thibault and Pinche, Ariane and Vlachou-Efstathiou, Malamatenia and Chagu{\'e}, Alix and Camps, Jean-Baptiste and Gille-Levenson, Matthias and Brisville-Fertin, Olivier and Fischer, Franz and Gervers, Michaels and Boutreux, Agn{\`e}s and Manton, Avery and Gabay, Simon and O'Connor, Patricia and Haverals, Wouter and Kestemont, Mike and Vandyck, Caroline and Kiessling, Benjamin},
  URL = {https://inria.hal.science/hal-04453952},
  NOTE = {working paper or preprint},
  YEAR = {2024},
  MONTH = Feb,
  KEYWORDS = {Historical sources ; medieval manuscripts ; Latin scripts ; benchmarking dataset ; multilingual ; handwritten text recognition},
  PDF = {https://inria.hal.science/hal-04453952/file/ICDAR24___CATMUS_Medieval-1.pdf},
  HAL_ID = {hal-04453952},
  HAL_VERSION = {v1},
}
```

### APA

Thibault Clérice, Ariane Pinche, Malamatenia Vlachou-Efstathiou, Alix Chagué, Jean-Baptiste Camps, et al.. CATMuS Medieval: A multilingual large-scale cross-century dataset in Latin script for handwritten text recognition and beyond. 2024. ⟨hal-04453952⟩

## License

This project is licensed under the MIT License.