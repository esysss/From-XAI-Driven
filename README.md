# Process Mining of Surrogate Decision Paths for Clinical Explainability

This repository contains the implementation for the paper _"Process Mining of Surrogate Decision Paths for Clinical Explainability"_. The goal of this project is to extract, cluster, and analyze decision paths from surrogate models trained to approximate black-box machine learning models used in kidney transplant outcome prediction.

## 🔍 Overview

The code performs the following key steps:
- Extracts decision paths from surrogate models (CART trees)
- Preprocesses and encodes paths using **TF-IDF vectorization**
- Applies **K-Means** and **Spectral Clustering** to group similar decision traces
- Converts clustered decision paths into **PM4Py-compatible event logs**
- Performs **process discovery** using the **Inductive Miner**
- Evaluates the resulting models using **Precision**, **Recall**, and **F-score**

## 📂 Repository Structure

- `main_notebook.ipynb` — The primary Jupyter notebook that contains the **entire workflow**, including data preprocessing, clustering, process model discovery, and evaluation.
