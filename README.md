# LRC-Net: Deep LUT-based Logic Networks with Local Random Connectivity and Identity-Mapping Initialization

<p align="center">
  <img width="688" height="304" alt="image" src="https://github.com/user-attachments/assets/05247f11-ea0f-4ebf-b5a9-2aff8fac170f" />
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
  <img width="687" height="298" alt="image" src="https://github.com/user-attachments/assets/ef13ffb3-0755-4c84-b794-527bb5f2b728" />
</p>

---

## 📈 Test Accuracy Convergence Curves

The following figures show the test accuracy convergence curves of LRC-Net on benchmark datasets.

### CIFAR-10 and MNIST

<p align="center">
  <img width="400" height="302" alt="image" src="https://github.com/user-attachments/assets/62e9d7ae-b1d2-49ad-bfa0-e569fa779d54" /> <img width="399" height="303" alt="image" src="https://github.com/user-attachments/assets/88ad6cd1-9aa5-447b-aed8-684a1a297cd8" />

</p>
---

## ✨ Key Features

- Deep LUT-based logic network architecture.
- Local random connectivity for image-structured input.
- Identity-mapping initialization for stable deep training.
- Improved convergence behavior on image classification tasks.
- Hardware-friendly design suitable for LUT-based acceleration.

---

## 📚 REFERENCES
[1]	Ramírez I, Garcia-Espinosa F J, Concha D, et al. LLNN: A Scalable LUT-Based Logic Neural Network Architecture for FPGAs[J]. IEEE Transactions on Circuits and Systems I: Regular Papers, 2025.
[2] Ramírez I, Garcia-Espinosa F J, Concha D, et al. Logic neural networks for efficient FPGA implementation[J]. IEEE Transactions on Circuits and Systems I: Regular Papers, 2024, 72(7): 3390-3398.
[3] Petersen F, Borgelt C, Kuehne H, et al. Deep differentiable logic gate networks[J]. Advances in Neural Information Processing Systems, 2022, 35: 2006-2018.

