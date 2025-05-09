# Streamlining Product Categorization for E-Commerce Using LoRA

This project explores the use of **Low-Rank Adaptation (LoRA)** to fine-tune **Large Language Models (LLMs)** such as GPT-2, BERT, and DistilRoBERTa for scalable and efficient product categorization in e-commerce platforms. By reducing the number of trainable parameters, LoRA significantly improves training efficiency without sacrificing accuracy.

![]()
## 🧠 Project Overview

Manual product categorization in e-commerce is labor-intensive and inconsistent. This project automates the process using fine-tuned transformer-based models enhanced by LoRA, enabling high performance with minimal computational cost. We experimented with multiple LLMs and demonstrated substantial gains in AUC and efficiency using LoRA.

## 🧪 Models & Methodology

- **Models Used**: GPT-2, BERT, DistilRoBERTa
- **Technique**: Fine-tuning using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685)
- **Dataset**: [Amazon Reviews Dataset (2023)](https://amazon-reviews-2023.github.io/) with categories such as *Grocery*, *Fashion*, *Handmade*, etc.
- **Preprocessing**:
  - Tokenization via HuggingFace tokenizers
  - Label encoding
  - Stratified train/validation/test split (80/10/10)

## 📊 Results

- **GPT-2 with LoRA** outperformed BERT and DistilRoBERTa across all metrics (Accuracy, Precision, Recall, F1-score).
- Significant **AUC improvement** observed in minority classes (e.g., Handmade: 0.80 → 0.93).
- Achieved **comparable accuracy** to full fine-tuning with **reduced compute and memory** requirements.

| Model         | Best Class AUC | Lowest Class AUC | Overall Performance |
|---------------|----------------|------------------|---------------------|
| **GPT-2 + LoRA** | 0.97 (Grocery) | 0.79 (Amazon Home) | 🏆 Highest |
| BERT + LoRA   | 0.96 (Grocery) | 0.69 (All Beauty) | Moderate |
| DistilRoBERTa + LoRA | 0.93 | 0.71 | Moderate |

## ⚙️ Tech Stack

- Python
- Transformers (Hugging Face)
- PyTorch
- PEFT & LoRA
- scikit-learn
- matplotlib / seaborn

## 🚀 Future Work

- Test on additional LLMs like RoBERTa, T5, and ALBERT
- Handle class imbalance with stratified sampling and advanced augmentation
- Apply cross-validation for robust tuning
- Extend to multilingual categorization

## 📚 References

- [LoRA: Low-Rank Adaptation of LLMs (Hu et al.)](https://arxiv.org/abs/2106.09685)
- [Amazon Reviews Dataset](https://amazon-reviews-2023.github.io/)
