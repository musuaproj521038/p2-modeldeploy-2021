# p2-datascience-2021 -> p2-modeldeploy-2021

## Data Stream
### Requirements:
* [Python v3.8+](https://www.python.org/downloads/)
* [PIP](https://docs.python.org/3/installing/index.html)
* Azure Account with Azure ML Studio up.
* Config file from Azure ML instance overview. To be placed in project directory.

Optional
* [Docker](https://docs.docker.com/get-docker/)

### How to use (if running locally):
1.) Run the following command:
```bash
pip install -r requirements.txt
```

2.) Load jupyter notebooks (should be installed in step 1) and then open `upload-to-azure-to-use.ipynb`.

3.) Run the first four code blocks.
On the fourth block uncomment the `2nd - 3rd` line and comment out the `5th - 6th` line to host your model online.
Leave the fourth block as is if you want to host locally. Note that you need Docker installed to run the local version.

### `commonreadaility-generate-model.ipynb`
Expanded to get model files and process the Guthenberg files.

The Kaggle for this file can be opened by clicking [here](https://www.kaggle.com/kaisen420/commonreadaility) and also includes the csv files of the processed Guthenberg data for the scraped top 100 ebooks, results for the guthenberg data and includes the required generated models.

### `upload-to-azure-to-use.ipynb`
Upload model generated by `commonreadaility-generate-model.ipynb` to Azure with the option to host locally or in Azure.

You need the following files before running this notebook:
* `config.json` - obtained from Azure Mchine Learning portal.
* `model_0.joblib` - generated from `commonreadaility-generate-model.ipynb`. The file is available [here](https://www.kaggle.com/kaisen420/commonreadaility?scriptVersionId=75405639) in the output section.
* `model_1.joblib` - same as `model_0.joblib`.