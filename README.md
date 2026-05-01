# Gradient Guided Adversarial Patch Attack

This project implements a gradient-based adversarial patch attack pipeline for evaluating model robustness.

---

##  Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/RishavKumarIIT/Gradient-Guided-Adversarial-Patch-Attack.git
cd Gradient-Guided-Adversarial-Patch-Attack
```

---

##  Dataset Setup

Create a `data` folder in the root directory:

```bash
mkdir data
```

Download the ImageNet dataset and organize it in the following structure:

```
data/
└── imagenet/
    └── train/
        └── 1/
            └── images/
                ├── img1.jpg
                ├── img2.jpg
                └── ...
```

>  Note: Ensure the dataset follows this directory structure for correct loading.

---

##  Install Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

##  Run the Project

Execute the main evaluation script:

```bash
python src/ResultEval.py
```

---

##  Notes

* Make sure your environment has Python 3.8+ installed.
* GPU is recommended for faster execution.
* Ensure all dependencies are properly installed before running.

---

##  Directory Structure (Expected)

```
Gradient-Guided-Adversarial-Patch-Attack/
│
├── src/
├── data/
├── outputs/
├── saved_images/
├── requirements.txt
└── README.md
```

---

##  Author

**Rishav Kumar, Umesh kashyap, Dr. SK Subid Ali**
