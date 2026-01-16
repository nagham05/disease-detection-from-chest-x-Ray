#  Disease Detection from Chest X-Ray Images using Deep Learning

##  Project Overview

This project focuses on **automatic disease detection from chest X-ray images**, specifically **binary classification of Pneumonia vs. Normal cases**, using **deep learning and transfer learning techniques**.

The work is implemented entirely in a **Jupyter Notebook (`main.ipynb`)** and demonstrates the **end-to-end machine learning pipeline**, including:

* Data loading and preprocessing
* Baseline CNN experimentation
* Transfer learning using a pretrained CNN
* Model training, validation, and evaluation
* Performance analysis using accuracy and AUC

The project highlights how **AI can assist medical imaging analysis** while also addressing common deep learning challenges such as **overfitting** and **limited dataset size**.

---

##  Project Goals

* Build a **binary image classification model** for chest X-ray images
* Detect **Pneumonia** from radiology images
* Explore why a **basic CNN overfits** on this dataset
* Improve performance using **pretrained CNNs (transfer learning)**
* Evaluate model performance using **Accuracy and AUC metrics**

---

##  Dataset

**Chest X-Ray Images (Pneumonia)**
Source: Kaggle
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Dataset Details

* **Total images:** 5,863
* **Classes:**

  * `NORMAL`
  * `PNEUMONIA`
* **Directory structure:**

  ```
  chest_xray/
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── val/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  └── test/
      ├── NORMAL/
      └── PNEUMONIA/
  ```

⚠️ **Important Note:**
This project **does not support dataset upload via a web interface**.
All X-ray images **must be downloaded and stored locally** and the dataset path must be updated accordingly in the notebook.

---

##  Problem Understanding

### What problem is being solved?

* **Binary classification task**
* Input: Chest X-ray images
* Output:

  * `0` → Normal
  * `1` → Pneumonia

### Key Challenges

* High image similarity between classes
* Limited validation data
* Overfitting when training CNNs from scratch

---

##  Approach & Methodology

### 1. Initial CNN Experimentation

* A **basic CNN model** was implemented first
* Observed **severe overfitting**:

  * High training accuracy
  * Poor validation generalization

This highlighted the need for **better feature extraction**.

---

### 2. Transfer Learning Strategy

To address overfitting, the project uses:

####  Pretrained Model

* **EfficientNetB0**
* Pretrained on **ImageNet**
* Used as a **fixed feature extractor**

```python
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

base_model.trainable = False
```

#### Why EfficientNet?

* Strong performance on image classification
* Parameter-efficient
* Better generalization on small datasets

---

### 3. Data Preprocessing

* Image resizing to **224 × 224**
* Pixel normalization
* Data generators for:

  * Training
  * Validation
  * Testing

---

### 4. Model Architecture

* Pretrained EfficientNet backbone
* Global Average Pooling
* Fully connected classification head
* Sigmoid activation for binary classification

---

### 5. Training Configuration

* Optimizer: **Adam**
* Loss function: **Binary Crossentropy**
* Metrics:

  * Accuracy
  * AUC
* Learning rate tuning applied to stabilize training

---

##  Results Summary

###  Training & Validation

* Training loss steadily decreased
* Validation accuracy improved after fixing preprocessing and learning rate
* Validation AUC approached **0.99**, though results should be interpreted cautiously due to validation size

---

###  Test Set Evaluation

* **Test Accuracy:** ~88%
* **Test AUC:** ~0.954

 Performance drop from validation to test is expected and reflects real-world generalization.

---

##  Evaluation Metrics

* **Accuracy** – overall classification correctness
* **AUC (Area Under ROC Curve)** – model’s ability to distinguish between classes

AUC is especially important in medical classification problems where class imbalance exists.

---

## ⚠️ Limitations

* Dataset size is relatively small
* Validation set is limited
* Model performance should not be considered clinical-grade
* Further fine-tuning and external validation are required

---

##  How to Run the Notebook

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Download the Dataset

* Download from Kaggle
* Extract locally
* Update dataset paths inside `main.ipynb`

### 3. Install Dependencies

```bash
pip install tensorflow keras numpy matplotlib
```

### 4. Run the Notebook

```bash
jupyter notebook main.ipynb
```

---

## File Structure

```
.
├── main.ipynb        # Complete project notebook
├── README.md         # Project documentation
└── chest_xray/       # Dataset (local only, not included)
```

---

##  Ethical & Educational Disclaimer

This project is **for educational purposes only**.
It **must not** be used for real medical diagnosis or clinical decision-making.

---

##  Key Takeaways

* Transfer learning significantly outperforms basic CNNs on small datasets
* Proper preprocessing and learning rate tuning are critical
* Medical image classification requires careful evaluation and validation


