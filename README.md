# Astronomy Data Analysis — ML/AI Learning Project

A hands-on journey through machine learning and AI using a real star dataset (`star_dataset.csv`). Each phase introduces new libraries and concepts, building from raw data exploration all the way to convolutional neural networks.

---

## Setup

Install all libraries needed for the full project in one command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras-tuner polars jupyter
```

Launch Jupyter to open any notebook:

```bash
jupyter notebook
```

---

## Dataset

`star_dataset.csv` contains observations of named stars with six columns:

| Column | Description |
| --- | --- |
| `Name` | Star name (e.g., Sirius, Betelgeuse) |
| `Distance (ly)` | Distance from Earth in light years |
| `Luminosity (L/Lo)` | Luminosity relative to the Sun |
| `Radius (R/Ro)` | Radius relative to the Sun |
| `Temperature (K)` | Surface temperature in Kelvin |
| `Spectral Class` | Stellar classification (e.g., A1V, M2Iab, B7V) |

Some rows contain negative luminosity values (data artifacts) and some stars appear multiple times as repeated observations — both issues are addressed in Phase 1.

---

## Phase 1 — Data Exploration & Cleaning

**Notebook:** `phase1_exploration.ipynb`
**Libraries:** pandas, numpy

**What you'll learn:**

- Loading CSV data with `pd.read_csv()` and inspecting it with `.info()`, `.describe()`, `.value_counts()`, `.isnull()`
- Identifying and removing negative luminosity rows (data artifacts)
- Handling duplicate star observations by grouping and averaging
- Core numpy array operations and broadcasting
- Exporting a cleaned dataset to a new CSV for use in later phases

**Why this matters:** Every ML project starts here. No amount of model sophistication fixes bad input data, and building the habit of thoroughly understanding your data before modeling is the single most important skill in the field.

---

## Phase 2 — Data Visualization

**Notebook:** `phase2_visualization.ipynb`
**Libraries:** matplotlib, seaborn

**What you'll learn:**

- Histograms, scatter plots, and box plots with matplotlib
- The **Hertzsprung-Russell (HR) diagram**: Temperature on the x-axis (reversed, hotter left) vs. Luminosity on a log-scale y-axis — the signature plot of stellar astronomy. Stars naturally cluster into the Main Sequence, Giants, and Supergiants
- Correlation heatmap and pairplot colored by spectral class using seaborn
- Plot styling: figure size, axis labels, colormaps, log scales, legends

**Why this matters:** Visual intuition is essential before modeling. The HR diagram alone reveals the natural structure that your classifiers and clustering algorithms will later learn from data — seeing it first makes those results meaningful.

---

## Phase 3 — Classical Classification

**Notebook:** `phase3_classification.ipynb`
**Libraries:** scikit-learn, pandas

**What you'll learn:**

- Feature engineering: extract the broad spectral class letter (O, B, A, F, G, K, M) from the full `Spectral Class` string to use as the target label
- Feature normalization with `StandardScaler`
- Train/test split with `train_test_split`
- Three classifiers side-by-side: K-Nearest Neighbors, Logistic Regression, Decision Tree
- Evaluating results: confusion matrix, classification report (precision, recall, F1-score)
- Cross-validation with `cross_val_score` to get reliable performance estimates

**Why this matters:** Introduces the core scikit-learn API (`fit` / `predict` / `score`) and the model evaluation mindset. Comparing three classifiers on the same data teaches you how to make principled model selection decisions.

---

## Phase 4 — Ensemble Methods & Unsupervised Learning

**Notebook:** `phase4_ensemble_clustering.ipynb`
**Libraries:** scikit-learn

**What you'll learn:**

- **Random Forest:** how bagging and feature randomness reduce overfitting; reading `feature_importances_` to understand which star properties matter most
- **K-Means Clustering:** the elbow method for choosing K, inspecting cluster centroids, scoring with the silhouette score
- **PCA (Principal Component Analysis):** reducing 4 features to 2 components for 2D visualization, reading the explained variance plot
- Overlaying discovered clusters against true spectral classes to evaluate unsupervised structure

**Why this matters:** Random Forest is one of the most reliable and widely-used classifiers in practice. K-Means + PCA is the standard unsupervised workflow. Together they expose both sides of ML: learning from labels and discovering structure without them.

---

## Phase 5 — Modern Data Processing with Polars

**Notebook:** `phase5_polars.ipynb`
**Libraries:** polars

**What you'll learn:**

- Polars vs. pandas: columnar storage, zero-copy operations, and the difference between eager and lazy evaluation
- Core Polars API: `pl.read_csv()`, `filter()`, `select()`, `group_by().agg()`
- The lazy API: `.lazy()` → chain transformations → `.collect()` — letting Polars optimize the entire query plan before running it
- Expressive method chaining (a natural Polars pattern)
- Side-by-side timing comparison of the same operations in pandas and polars

**Why this matters:** Polars is the modern, high-performance replacement for pandas in production data pipelines. Knowing both makes you versatile. This phase also reinforces all the data manipulation concepts from Phase 1 through a different lens.

---

## Phase 6 — Neural Networks

**Notebook:** `phase6_neural_networks.ipynb`
**Libraries:** tensorflow (keras)

**What you'll learn:**

- Building a `Sequential` model with `Dense`, `Dropout`, and `BatchNormalization` layers
- Activation functions: ReLU (hidden layers), Softmax (output layer for multi-class)
- Compiling with the Adam optimizer and categorical crossentropy loss
- Training with `EarlyStopping` and `ModelCheckpoint` callbacks
- Plotting training vs. validation loss and accuracy curves to diagnose overfitting
- Benchmarking the neural network against the Phase 3 classifiers

**Why this matters:** This is your first deep learning model. The star classification task is small enough to train in seconds on a CPU, making it the perfect environment to learn the Keras API, understand the training loop, and develop intuition for what the loss curves are telling you.

---

## Phase 7 — Hyperparameter Tuning

**Notebook:** `phase7_keras_tuner.ipynb`
**Libraries:** keras-tuner, tensorflow

**What you'll learn:**

- Why hyperparameters (number of layers, units per layer, dropout rate, learning rate) often matter as much as architecture
- Defining a `build_model(hp)` function that keras-tuner can call with different hyperparameter combinations
- Running `kt.RandomSearch` and `kt.Hyperband` tuning strategies
- Inspecting results with `tuner.results_summary()` and retrieving the best model
- Retraining the best-found architecture and comparing it to the Phase 6 baseline

**Why this matters:** Building a model is step one. Building a *well-tuned* model is step two. Keras Tuner is the standard tool for this in the TensorFlow ecosystem, and systematic tuning is what separates experimental scripts from production-grade models.

---

## Phase 8 — Convolutional Neural Networks (CNNs)

**Notebook:** `phase8_cnns.ipynb`
**Libraries:** tensorflow, matplotlib, numpy

**What you'll learn:**

- The core idea behind CNNs: locally-connected filters learn spatial patterns, and stacking them builds a spatial hierarchy of features
- Applying CNNs to tabular data via the **stellar fingerprint** technique: each star's 4 normalized features are reshaped into a 2×2 pixel image, creating a small image dataset the CNN can train on. This is a real technique used when practitioners want to apply CNN architectures to non-image data
- Building a CNN with `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers
- Data augmentation concepts (`RandomFlip`, `RandomRotation`) and when they apply
- Comparing CNN accuracy to the dense network from Phase 6
- Bonus: how production CNN architectures (ResNet, VGG, EfficientNet) are used for real astronomy image tasks such as galaxy morphology classification and exoplanet transit detection

**Why this matters:** CNNs are the foundation of modern computer vision and power applications far beyond images — including signal processing and 1D sequence data. Learning the architecture in a familiar context prepares you to apply it to any spatially-structured data.

---

## Learning Path Summary

| Phase | Topic | Libraries | Notebook |
| --- | --- | --- | --- |
| 1 | Data Exploration & Cleaning | pandas, numpy | `phase1_exploration.ipynb` |
| 2 | Data Visualization | matplotlib, seaborn | `phase2_visualization.ipynb` |
| 3 | Classical Classification | scikit-learn | `phase3_classification.ipynb` |
| 4 | Ensemble Methods & Clustering | scikit-learn | `phase4_ensemble_clustering.ipynb` |
| 5 | Modern Data Processing | polars | `phase5_polars.ipynb` |
| 6 | Neural Networks | tensorflow | `phase6_neural_networks.ipynb` |
| 7 | Hyperparameter Tuning | keras-tuner | `phase7_keras_tuner.ipynb` |
| 8 | Convolutional Neural Networks | tensorflow | `phase8_cnns.ipynb` |

---

## Superphase 2 — Advanced ML/AI with Gen1 Pokémon Data

A second series of phases using `Gen1_Pokemon.csv` (151 Pokémon, columns: `Name, Type 1, Type 2, Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed`). Built on **PyTorch** throughout, covering four advanced topics: transformers, generative models, reinforcement learning, and MLOps.

**Setup:**

```bash
pip install torch torchvision mlflow fastapi uvicorn requests umap-learn
```

---

## Superphase SP1 — PyTorch Foundations & Multi-Label Classification

**Notebook:** `sp1_pytorch_foundations.ipynb`
**Libraries:** torch, pandas, numpy, matplotlib, scikit-learn

**What you'll learn:**

- PyTorch tensors vs. numpy arrays; autograd and the `.backward()` computation graph
- `nn.Module`, `nn.Linear`, `nn.ReLU`; the PyTorch training loop (`zero_grad → forward → loss → backward → step`)
- Custom `Dataset` and `DataLoader` with batching and shuffling
- **Multi-label classification:** encode all 18 Pokémon types as a binary vector per Pokémon, use `BCEWithLogitsLoss`, evaluate with F1 (not plain accuracy)
- Compare to the equivalent Keras model from Phase 6 to make the framework transition concrete

**Why this matters:** The training loop you write by hand here is exactly what Keras hides. Understanding it makes every other Superphase 2 phase possible, and PyTorch is the dominant framework in both research and production.

---

## Superphase SP2 — Transformers & Self-Attention

**Notebook:** `sp2_transformers.ipynb`
**Libraries:** torch, math, matplotlib

**What you'll learn:**

- Scaled dot-product attention from scratch: `Attention(Q,K,V) = softmax(QKᵀ / √d_k) V`
- Multi-head attention: why multiple heads capture different relationships simultaneously
- Positional encoding applied to tabular data (each stat as a "token")
- Build a small Transformer encoder: `MultiHeadAttention → LayerNorm → FeedForward → LayerNorm`
- Apply to Pokémon type prediction; visualize attention weights to see which stats the model focuses on per type
- Brief tour of HuggingFace Transformers to see how production transformers are packaged

**Why this matters:** Transformers are the architecture behind GPT, BERT, and modern AI systems. Building one from scratch on a small dataset demystifies the mechanism before scaling up.

---

## Superphase SP3 — Generative Models (VAE + GAN)

**Notebook:** `sp3_generative_models.ipynb`
**Libraries:** torch, matplotlib, numpy, umap-learn

**What you'll learn:**

- **VAE (Variational Autoencoder):** encode 6 stats → mean/log-variance → sample latent vector → decode; the ELBO loss (reconstruction + KL divergence); the reparameterization trick
- Visualize the latent space with t-SNE/UMAP colored by type — see if types cluster without supervision
- Sample from the latent space to generate new Pokémon stat distributions
- **GAN (Generative Adversarial Network):** generator creates fake stat vectors, discriminator distinguishes real vs. fake; alternating training updates; detecting mode collapse
- Compare VAE vs. GAN: VAE gives a structured latent space, GAN gives sharper samples

**Why this matters:** The VAE latent space visualization is one of the most satisfying results in ML — types separating without any labels is a concrete demonstration of representation learning. Generative models power image synthesis, drug discovery, and data augmentation.

---

## Superphase SP4 — Reinforcement Learning (DQN)

**Notebook:** `sp4_reinforcement_learning.ipynb`
**Libraries:** torch, numpy, collections

**What you'll learn:**

- RL fundamentals: agent, environment, state, action, reward, episode; the Bellman equation
- Build a simplified Pokémon battle environment as a Python class: state = [own HP, opponent HP, stats], actions = [attack, special attack, switch], reward = +1 win / −1 lose / small negative per turn
- **Q-Learning** with a table first to build intuition for the update rule
- **DQN (Deep Q-Network):** replace the Q-table with a neural net; experience replay buffer; target network (frozen copy updated every N steps to stabilize training)
- Plot episode rewards over training to watch the agent improve
- Stretch: incorporate type effectiveness so the agent learns to exploit matchups

**Why this matters:** RL is the framework behind AlphaGo, robotics, and recommendation systems. The Pokémon battle context makes the reward signal intuitive, and building DQN from scratch gives you the foundation to read any modern RL paper.

---

## Superphase SP5 — MLOps & Model Deployment

**Notebook:** `sp5_mlops.ipynb` + supporting scripts
**Libraries:** mlflow, fastapi, uvicorn, torch, requests

**What you'll learn:**

- **MLflow:** log experiments (hyperparameters, metrics, artifacts) from SP1 and SP2; compare runs in the MLflow UI; register a model version
- Save/load PyTorch models correctly (`torch.save` / `torch.load` with state dicts)
- **FastAPI:** wrap the best classifier in a REST API; POST endpoint accepts 6 stats as JSON and returns predicted type probabilities
- Test the API with `requests` from inside the notebook
- **Model monitoring concept:** log prediction distributions and flag inputs that fall outside the training range (simple drift detection)
- Conceptual overview of scaling: Docker containers, cloud deployment

**Why this matters:** A model that can't be served is a research artifact. FastAPI + MLflow is the standard lightweight stack for going from notebook to production, and this phase closes the full ML lifecycle started in SP1.

---

## Superphase 2 Summary

| Phase | Topic | Libraries | Notebook |
| --- | --- | --- | --- |
| SP1 | PyTorch Foundations & Multi-Label Classification | torch, pandas, sklearn | `sp1_pytorch_foundations.ipynb` |
| SP2 | Transformers & Self-Attention | torch | `sp2_transformers.ipynb` |
| SP3 | Generative Models (VAE + GAN) | torch, matplotlib, umap-learn | `sp3_generative_models.ipynb` |
| SP4 | Reinforcement Learning (DQN) | torch, numpy | `sp4_reinforcement_learning.ipynb` |
| SP5 | MLOps & Model Deployment | mlflow, fastapi, torch | `sp5_mlops.ipynb` |
