# LRC-Net: Local Random Connectivity and Logic-Buffer Initialization for Deep LUT-Based Logic Neural Networks

<p align="center">
  <img width="1128" height="831" alt="image" src="https://github.com/user-attachments/assets/9e3c2649-2bb6-4d7d-86d4-b7dc2414fed2" />
</p>

## 📌 Overview

**LRC-Net** is a deep LUT-based logic network framework designed to improve the scalability and trainability of logic neural networks.  
The proposed method introduces two key techniques:

- **Local Random Connectivity (LRC)**: injects convolution-like spatial inductive bias into LUT-based logic networks.
- **Logic-buffer initialization (LBI)**: alleviates gradient vanishing and enables deeper logic-based architectures.

Together, LRC and LBI allow LUT-based networks to achieve better accuracy, faster convergence, and improved depth scalability on image classification tasks.

---

## 🔗 Local Random Connectivity (LRC)

The core idea of **Local Random Connectivity (LRC)** is to introduce a convolutional-like spatial inductive bias into logic networks.

Instead of fully random global connections, LRC restricts the input connections of each LUT neuron to a local spatial region. This design helps the network better capture local patterns in images while maintaining the hardware-friendly characteristics of LUT-based logic computation.

<p align="center">
  <img width="1167" height="537" alt="image" src="https://github.com/user-attachments/assets/f0e131c1-bc59-4f2f-af0b-81be43f60ea8" />
</p>

---

## 📈 Test Accuracy Convergence Curves

The following figures show the test accuracy convergence curves of LRC-Net on benchmark datasets.

### CIFAR-10 and MNIST

<p align="center">
  <img width="600" height="450" alt="image" src="https://github.com/user-attachments/assets/7091c96e-6ab0-40cb-8b14-825034816f2e" /> 
  <img width="600" height="450" alt="image" src="https://github.com/user-attachments/assets/e7654e01-8611-438d-845a-bb0da718ec87" />
</p>
---

## ✨ Key Features

- Deep LUT-based logic network architecture.
- Local random connectivity for image-structured input.
- Logic buffer initialization for stable deep training.
- Improved convergence behavior on image classification tasks.
- Hardware-friendly design suitable for LUT-based acceleration.

---

## 📚 REFERENCES
- [1]	Ramírez I, Garcia-Espinosa F J, Concha D, et al. LLNN: A Scalable LUT-Based Logic Neural Network Architecture for FPGAs[J]. IEEE Transactions on Circuits and Systems I: Regular Papers, 2025.
- [2] Ramírez I, Garcia-Espinosa F J, Concha D, et al. Logic neural networks for efficient FPGA implementation[J]. IEEE Transactions on Circuits and Systems I: Regular Papers, 2024, 72(7): 3390-3398.
- [3] Petersen F, Borgelt C, Kuehne H, et al. Deep differentiable logic gate networks[J]. Advances in Neural Information Processing Systems, 2022, 35: 2006-2018.

