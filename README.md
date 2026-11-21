# COMP 472 Project, Fall 2026: Team 10

Team members: Omar, Roham, Sean

We compare four machine learning algorithms on two classification tasks from the course's Kaggle competition:

- **Image:** 500 features per image, 50 object classes
- **Sentiment:** 768 BERT features per review, 3 classes (Negative, Neutral, Positive)

The algorithms are k-nearest neighbours (k-NN), a decision tree, a random forest and a feed-forward neural network. We also ran clustering experiments: a Gaussian mixture model (GMM) and k-means.

The project report will be added to this repository when it is final.

## Results

Validation accuracy (%) on one shared stratified 80/20 split (seed 42):

| Model | Image | Sentiment | Overall |
|---|---|---|---|
| k-NN | 56.3 | 64.7 | 60.5 |
| Neural network | 51.6 | 67.1 | 59.3 |
| Random forest | 26.2 | 58.4 | 42.3 |
| Decision tree | 12.7 | 48.5 | 30.6 |

**Final Kaggle submission:** k-NN for image and the neural network for sentiment. It scored 0.617 on validation and **0.628 on the public leaderboard**.

## What is in this repository

| File | What it does |
|---|---|
| `comparison.ipynb` | **Main notebook.** Runs all four models on the same validation split, compares them (accuracy, timing, per-class results, shared mistakes, plots), retrains the best model for each task on all the training data and writes the Kaggle submission (Sections 13 and 14). |
| `clustering.ipynb` | Clustering: GMM on both tasks and k-means on image, with purity, adjusted Rand index, BIC and silhouette scores. |
| `knn-kmean-Roham/knn_complete.ipynb` | k-NN tuning: distances, feature scaling, voting and k, with cross-validation of the best settings. |
| `knn-kmean-Roham/kmeans.ipynb` | k-means used as a classifier (each cluster labelled with its most common class): feature scaling, number of clusters and several random starts. |
| `decision_tree_random_forest_sean/decision_tree_random_forest_gmm.ipynb` | Decision tree and random forest tuning. Its GMM cells were later moved to `clustering.ipynb`. |
| `nn_omar/nn_experiments.ipynb` | Neural network experiments: baseline, dropout, learning rate, network size, averaging several networks. |
| `requirements.txt` | Python packages needed to run the notebooks. |

## Coding rules we followed

- k-NN, the decision tree, the random forest, the GMM and k-means are written from scratch with NumPy.
- PyTorch is used only for the neural network.
- scikit-learn is used only for splitting the data, PCA preprocessing and evaluation metrics.

## Reproducibility

**Random seed: 42.** It fixes the validation split, the k-means and GMM starting points, and the random forest's samples. Each neural network has its own seed (0 to 9 for image, 0 to 4 for sentiment), set for Python, NumPy and PyTorch by `set_seed`.

**Validation:** a stratified 80/20 split of each training file (image: 16,000 train / 4,000 validation; sentiment: 9,600 / 2,400). Feature scaling is learned on the training part only. Every model in `comparison.ipynb` uses this same split, and the final model for each task is the one with the highest validation accuracy (Section 13). The Kaggle leaderboard was not used to choose settings.

**Environment:** `comparison.ipynb`, `clustering.ipynb` and `nn_omar/nn_experiments.ipynb` ran on Google Colab with Python 3.13.16, NumPy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, Matplotlib 3.10.0 and PyTorch 2.14.1 (CUDA 13.0, T4 GPU). The other notebooks ran locally (`knn-kmean-Roham`: Python 3.14.2; `decision_tree_random_forest_sean`: Python 3.12.0). `requirements.txt` pins the Colab versions.

**Final settings** (from `comparison.ipynb`, Section 9):

| Model | Image | Sentiment |
|---|---|---|
| k-NN | square-root scaling (rows rescaled to length 1), Euclidean distance, k = 1 | PCA whitening to 64 dimensions, cosine distance, weighted vote, k = 3,000 |
| Decision tree (Gini) | max depth 60, min 50 rows to split | max depth 8, min 2 rows to split |
| Random forest | 300 trees, max depth 60, 11 features per split, min 50 rows to split | 50 trees, max depth 10, 28 features per split, min 200 rows to split |
| Neural network | [512, 256] hidden units, L2 scaling, AdamW (learning rate 0.001, weight decay 0.01), cosine schedule over 60 epochs, label smoothing 0.1, input dropout 0.5, batch size 128; average of 10 networks | [256, 64] hidden units, standardized features, Adam (learning rate 0.001), batch size 128, early stopping (patience 10, at most 100 epochs); average of 5 networks |

The submitted model is k-NN for image and the neural network for sentiment, retrained on all the training rows (Section 14).

**Running without a GPU or outside Colab:** the networks run on the CPU too, just more slowly. Outside Colab, delete the two `drive` lines at the top of the first code cell and set `ZIP_PATH` (and `OUT_DIR` in `comparison.ipynb`) to folders on your computer.

## How to run

### 1. Get the data

Download `comp-472-project-fall-2026-concordia.zip` from the Kaggle competition. The data is not included in this repository. The notebooks expect it in different places:

| Notebooks | Where to put the data |
|---|---|
| `comparison.ipynb`, `clustering.ipynb`, `nn_omar/nn_experiments.ipynb` | Google Drive, at `MyDrive/comp472/comp-472-project-fall-2026-concordia.zip`. The first cells mount Drive and unzip it. To use another location, change `ZIP_PATH` in the first code cell. |
| `knn-kmean-Roham/*.ipynb` | Unzip it next to the notebook, so the CSVs are in `comp-472-project-fall-2026-concordia/`. |
| `decision_tree_random_forest_sean/*.ipynb` | Put the four CSVs in a `datasets/` folder next to the notebook. |

### 2. Run the notebooks

**On Google Colab (recommended for the first three notebooks):** open the notebook in Colab and choose Runtime > Run all. The neural network sections need a GPU runtime (Runtime > Change runtime type > T4 GPU).

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook
```

Run each notebook's cells from top to bottom: later cells use variables from earlier ones.

### Notes on `comparison.ipynb`

- Each model section saves its results to the folder in `OUT_DIR` (set in Section 1) and skips itself on a rerun if its results are already there. To rerun a model from scratch, delete its files from that folder.
- A full run from scratch takes over an hour. Most of that time is the random forest on image (about 40 minutes); the neural networks take about 5 minutes on a T4 GPU.
- **To reproduce the Kaggle submission:** run the whole notebook. Section 14 writes `submission_final.csv` to `OUT_DIR`.
- **Known issue:** in `train_model` (Section 8.1), the two `keep_last_epoch` lines are indented inside the training loop, so in Section 14 each final sentiment network trained for 1 epoch instead of the planned 1 or 2. We found this after submitting and left the code as submitted. The fix is to move those two lines out of the loop, just above `model.load_state_dict(best['state'])`. It does not affect any validation result.

### Notes on `clustering.ipynb`

The GMM sections are slow: about 10 minutes for the sentiment BIC loop, 16 minutes for the image BIC loop and 10 minutes for the 50-cluster image GMM. The k-means sections (Section 6) take a few minutes in total.
