# Data Augmentation for Improving the Prediction of Validity and Novelty of Argumentative Conclusions

This is the code for the paper which you can cite as follows:

````txt
@inproceedings{heinisch-etal-2022-data,
    title = "Data Augmentation for Improving the Prediction of Validity and Novelty of Argumentative Conclusions",
    author = "Heinisch, Philipp  and
      Plenz, Moritz and 
      Opitz, Juri and 
      Frank, Anette  and
      Cimiano, Philipp",
    booktitle = "Proceedings of the 9th Workshop on Argument Mining",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "Association for Computational Linguistics"
}
````

## How to use

First, you have to install the required libraries (we recommend Python V3.9):

``
python -m pip install -r requirements.txt
``

Then, have a look into the subfolders of the folder _ArgumentData_. You have to download and move the data files into the right places. For this, download the datasets as given in the _ref.txt_-files in the subfolders.

---

After doing this, have a look into the ``Main.py``, the main file for executing the code.

This code is argument-based. With ``python Main.py --help``, you see your options:

> You want to know whether your conclusion candidate given a premise is valid and/or novel? Use this tool! (NLP->Argument Mining)

With ``Main.py``, you train and evaluate a model predicting validity and novelty in a multi-task-manner.

### Other useful scripts

- **FindBestValidatedModels.py**: If you repeat a configuration with ``Main.py --repetitions >= 2`` multiple times, this script finds the best repetition (model) based on the best ValNov-performance on the development split
- **Inference.py**: generates for all test samples predictions based on a given [fine-tuned] model (probably in the _.out_-folder)

## Demo

You can play around [here](https://huggingface.co/spaces/pheinisch/ConclusionValidityNoveltyClassifier-Augmentation).