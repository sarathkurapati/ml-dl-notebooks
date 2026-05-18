# Machine Learning & Deep Learning Portfolio

Twenty Jupyter notebooks built while learning applied machine learning —
regression, classification, deep learning, and image classification with
transfer learning. Each notebook works through one problem end-to-end on a
real dataset.

**Author:** K Sarath &nbsp;·&nbsp; **Reg No:** 20BCI0171

Every notebook ships with a working "Open in Colab" badge — click it on
GitHub and the notebook opens directly in Colab where you can re-run it.

---

## What's inside

| Category | Count | Problems covered |
|---|---|---|
| Regression | 8 | Prices, sales, charges, air quality index, revenue forecasting |
| Classification | 8 | Loan approval, churn, diagnoses, machine failure, water potability |
| Image classification (transfer learning) | 3 | Weather, fabric defects, chest X-ray |
| Exploratory data analysis | 1 | Shark Tank India investor behaviour |

The two diabetes notebooks were submitted as **Lab Assignment I — Advanced
Predictive Analytics (MDI3003)** and follow a fixed 8-step structure
(column listing, null checks, descriptive stats, histograms, pairplots,
correlation heatmap, class-balance check, decision tree fit + evaluation).

---

## Notebook index

### Regression

| Notebook | Dataset | Models |
|---|---|---|
| `ADVANCED_REG_ON_HOUSE_PRICE.ipynb` | `train.csv` (Kaggle House Prices) | Linear Regression, Decision Tree, Random Forest, XGBoost, ANN |
| `ADVERTISING.ipynb` | `Advertising.csv` (TV/Radio/Newspaper → Sales) | Linear Regression (with MinMaxScaler) |
| `Air_quality.ipynb` | `AQI Data Set.csv` | Linear Regression, Decision Tree, Random Forest, XGBoost, ANN |
| `Building_ANN.ipynb` | `startups.csv` (50_Startups → profit) | ANN |
| `car_price.ipynb` | `ds.csv` (used-car prices) | Linear Regression, ANN |
| `GOLD_PRICE_PREDICTION.ipynb` | `gold_data.csv` + live `yfinance` feed | Facebook Prophet (time series) |
| `Medical_Insurance.ipynb` | `Medical_insurance.csv` | Linear Regression, Decision Tree, Random Forest, XGBoost, ANN |
| `Spotify_Revenue__Expenses_and_Its_Premium_Users.ipynb` | `Spotify Quarterly.csv` | ANN regressor |

### Classification

| Notebook | Dataset | Models |
|---|---|---|
| `BCWdata.ipynb` | `BCWdata.csv` (Breast Cancer Wisconsin) | ANN |
| `Bank_marketing.ipynb` | `Bank Marketing.csv` | Decision Tree, Random Forest, SVM, KNN, Naive Bayes, ANN |
| `Binary_Classification_of_Machine_Failures.ipynb` | `train.csv` (Kaggle Playground S3E17) | ANN |
| `DIABETES_DECISION_TREE.ipynb` | `diabetes.csv` (Pima Indians) | Decision Tree — lab assignment |
| `DIABETES_DATA_DECISION_TREE.ipynb` | `diabetes.csv` | Decision Tree — extended lab analysis |
| `loan_prediction.ipynb` | `loan_data.csv` | Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes |
| `SUNBASEDATA_customer_churn_dataset.ipynb` | Sunbase customer churn | Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, ANN |
| `water_quality_ann.ipynb` | `water_potability.csv` | ANN |

### Image classification (transfer learning)

| Notebook | Dataset | Models |
|---|---|---|
| `predictions.ipynb` | Multi-class Weather Dataset (cloudy / rain / shine / sunrise) | VGG16, ResNet50, InceptionV3, Xception |
| `AITEX.ipynb` | AITEX fabric-defect images | VGG16, ResNet50, InceptionV3, Xception |
| `chest_xray_cnn.ipynb` | Chest X-Ray Pneumonia dataset | Custom CNN (Conv2D + MaxPooling + Dense) |

### Exploratory data analysis

| Notebook | Dataset | What it covers |
|---|---|---|
| `SHARK_TANK_INDIAN_COMPANIES.ipynb` | `ShartankIndiaAllPitches.csv` | Deal/no-deal rates and total amount invested per shark — Anupam, Ashneer, Namita, Aman, Peyush, Vineeta, Ghazal |

---

## Actual results

These are best validation/test scores I observed in the notebook outputs —
not benchmarks, just what each training run produced.

| Notebook | Best result observed |
|---|---|
| `predictions.ipynb` (weather) | **VGG16 ≈ 0.87 val accuracy** — outperformed ResNet50, InceptionV3, Xception |
| `Binary_Classification_of_Machine_Failures.ipynb` | **~98% val accuracy** |
| `chest_xray_cnn.ipynb` | ~96% train / ~75–80% val (visible overfitting) |
| `Bank_marketing.ipynb` | **~88% accuracy** (multiple models converge here) |
| `BCWdata.ipynb` (breast cancer ANN) | **~86% val accuracy** |
| `loan_prediction.ipynb` | Random Forest **~83%**, Logistic Regression ~75% |
| `DIABETES_*.ipynb` | Decision Tree **~77% accuracy** |
| `water_quality_ann.ipynb` | ~63% val (dataset is genuinely hard) |
| `Medical_Insurance.ipynb` | R² ≈ 0.43 on the linear baseline; tree models improve on this |
| `SUNBASEDATA_customer_churn_dataset.ipynb` | ~50% — barely above chance, very weak signal |

The churn and water-potability notebooks were a useful reminder that not
every problem is solvable by throwing a default ANN at it — feature
engineering, class balancing, and tuning are sometimes worth more than
model choice.

---

## How each notebook is structured

Most notebooks follow the same flow with small variations:

1. **Load** — `pd.read_csv(...)`, or drive-mount + `!unzip` for image notebooks
2. **Inspect** — `.head()`, `.shape`, `.info()`, `.isnull().sum()`
3. **Clean** — drop irrelevant columns, impute nulls (mean/median), fix dtypes
4. **EDA** — histograms, boxplots (outliers), correlation heatmaps, pairplots
5. **Encode + scale** — `LabelEncoder` or one-hot for categoricals; `MinMaxScaler` / `StandardScaler` for numeric features
6. **Split** — `train_test_split` (typically 80/20)
7. **Fit** — one model per cell; for image notebooks the pre-trained CNN base is frozen and a small dense head is trained
8. **Evaluate** —
   - Regression → R², MAE, MSE, RMSE
   - Classification → accuracy, `classification_report`, `confusion_matrix`
   - Image → training/validation accuracy and loss across epochs
9. **Compare** — for notebooks with multiple models, a summary picks the best

---

## Tech stack

```text
Python 3
pandas, numpy
matplotlib, seaborn
scikit-learn       # Linear/Logistic Regression, Decision Tree,
                   # Random Forest, SVM, KNN, Naive Bayes,
                   # preprocessing, metrics
xgboost            # gradient boosting
tensorflow / keras # ANNs, CNNs, transfer learning
                   # (VGG16, ResNet50, InceptionV3, Xception)
prophet            # time-series forecasting (gold price)
yfinance           # live financial data
```

Notebooks were authored in Google Colab. Drive-mount cells appear in some
files — replace them with local paths when running outside Colab.

---

## Run a notebook

### Option 1 — Open in Colab (zero setup)

Open any notebook on GitHub and click the **Open in Colab** badge at the
top. Colab provides the runtime; you only need to upload the dataset file
(or mount your own Drive).

### Option 2 — Local

```bash
git clone https://github.com/sarathkurapati/ml-dl-notebooks.git
cd ml-dl-notebooks
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Suggested `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
tensorflow
prophet
yfinance
pillow
jupyter
```

---

## Datasets

Datasets are **not bundled** in this repo (most are Kaggle / UCI and have
redistribution restrictions). Each notebook expects its CSV in the same
folder as the notebook. Sources:

| File expected | Where to get it |
|---|---|
| `train.csv` (house prices) | [Kaggle: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| `Advertising.csv` | [Advertising — ISLR dataset](https://www.kaggle.com/datasets/ashydv/advertising-dataset) |
| `AQI Data Set.csv` | [Kaggle: Air Quality Index](https://www.kaggle.com/datasets) |
| `startups.csv` | 50_Startups (widely available on Kaggle) |
| `ds.csv` (car price) | [Kaggle: Used Cars](https://www.kaggle.com/datasets) |
| `gold_data.csv` | Pulled live via `yfinance` |
| `Medical_insurance.csv` | [Kaggle: Medical Cost Personal](https://www.kaggle.com/datasets/mirichoi0218/insurance) |
| `Spotify Quarterly.csv` | [Kaggle: Spotify Quarterly Data](https://www.kaggle.com/datasets) |
| `BCWdata.csv` | UCI Breast Cancer Wisconsin |
| `Bank Marketing.csv` | UCI Bank Marketing |
| `train.csv` (machine failures) | [Kaggle Playground Series S3E17](https://www.kaggle.com/competitions/playground-series-s3e17) |
| `diabetes.csv` | Pima Indians Diabetes (Kaggle / UCI) |
| `loan_data.csv` | [Kaggle: Loan Prediction](https://www.kaggle.com/datasets) |
| Sunbase churn data | Sunbase customer churn dataset |
| `water_potability.csv` | [Kaggle: Water Quality](https://www.kaggle.com/datasets/adityakadiwal/water-potability) |
| Multi-class Weather Dataset | [Kaggle: Multi-class Weather](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset) |
| AITEX fabric images | AITEX fabric image database |
| Chest X-Ray dataset | [Kaggle: Chest X-Ray (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| `ShartankIndiaAllPitches.csv` | [Kaggle: Shark Tank India](https://www.kaggle.com/datasets) |

For image notebooks, download the dataset, extract it locally, and update
`base_dir` at the top of the notebook to point at your extracted folder.

---

## Honest notes

- These notebooks were written iteratively while learning, so coding style,
  feature engineering, and tuning depth vary across files.
- Hyperparameter tuning is light in most notebooks — there's clear scope to
  add `GridSearchCV` / `RandomizedSearchCV` and proper cross-validation.
- Image notebooks freeze the pre-trained CNN base entirely. Unfreezing the
  upper convolutional blocks (and using a smaller LR) would likely improve
  validation accuracy further.
- `chest_xray_cnn.ipynb` shows clear overfitting (96% train vs ~78% val) —
  data augmentation and dropout could close this gap.
- Several notebooks were authored on Colab with `/content/...` paths.
  These need updating for local execution.

---

## License

Released for educational and portfolio purposes. Datasets retain their
original licenses from their respective sources.
