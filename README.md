# LRC-Net: Deep LUT-based Logic Networks with Local Random Connectivity and Identity-Mapping Initialization

<p align="center">
  <img width="500" height="400" alt="LRC-Net Overview" src="https://github.com/user-attachments/assets/49e40f6c-614c-426e-8179-868cb3350474" />
</p>

## 📌 Overview

**LRC-Net** is a deep LUT-based logic network framework designed to improve the scalability and trainability of logic neural networks.  
The proposed method introduces two key techniques:

- **Local Random Connectivity (LRC)**: injects convolution-like spatial inductive bias into LUT-based logic networks.
- **Identity-Mapping Initialization (IMI)**: alleviates gradient vanishing and enables deeper logic-based architectures.

Together, LRC and IMI allow LUT-based networks to achieve better accuracy, faster convergence, and improved depth scalability on image classification tasks.

---

## 🔗 Local Random Connectivity (LRC)

The core idea of **Local Random Connectivity (LRC)** is to introduce a convolutional-like spatial inductive bias into logic networks.

Instead of fully random global connections, LRC restricts the input connections of each LUT neuron to a local spatial region. This design helps the network better capture local patterns in images while maintaining the hardware-friendly characteristics of LUT-based logic computation.

<p align="center">
  <img width="500" height="400" alt="Local Random Connectivity" src="https://github.com/user-attachments/assets/2fbe5ab6-1add-4ab2-8dca-ebb88a1c88ef" />
</p>

---

## 🧩 Identity-Mapping Initialization (IMI)

**Identity-Mapping Initialization (IMI)** is introduced to improve the optimization of deep logic networks.

By preserving information flow across layers, IMI helps prevent vanishing gradients during training. This makes it possible for logic-based models to scale to greater depths and achieve better performance.

<p align="center">
  <img width="500" height="400" alt="Identity-Mapping Initialization" src="https://github.com/user-attachments/assets/29c1d354-b066-42fe-9d0b-ff90789b0f4b" />
</p>

---

## 📈 Test Accuracy Convergence Curves

The following figures show the test accuracy convergence curves of LRC-Net on benchmark datasets.

### CIFAR-10

<p align="center">
  <img width="500" height="400" alt="CIFAR-10 Accuracy Curve" src="https://github.com/user-attachments/assets/689e4cab-ff55-4333-856d-75fa87c7523a" />
</p>

### MNIST

<p align="center">
  <img width="500" height="400" alt="MNIST Accuracy Curve" src="https://github.com/user-attachments/assets/39e9e26b-80ef-473c-865a-5a73a1ea292d" />
</p>

---

## ✨ Key Features

- Deep LUT-based logic network architecture.
- Local random connectivity for image-structured input.
- Identity-mapping initialization for stable deep training.
- Improved convergence behavior on image classification tasks.
- Hardware-friendly design suitable for LUT-based acceleration.

---

## 📚 Citation

If this project is useful for your research, please consider citing or starring this repository.

```bibtex
@article{lrcnet,
  title={LRC-Net: Deep LUT-based Logic Networks with Local Random Connectivity and Identity-Mapping Initialization},
  author={Your Name},
  journal={},
  year={}
}
