# 🧠 MiniGPT - Train a Tiny Language Model from Scratch

This project implements a GPT-style language model trained from scratch using PyTorch, on a subset of the daily_dialog dataset. The model was built and trained on a single consumer GPU (NVIDIA GTX 3050 Ti, 4GB VRAM).

##  Highlights

- Fully custom GPT implementation (no HuggingFace, no pre-trained weights).
- Trained on real-world English daily_dialog.
- Achieved **val loss ~3.7**, trained in under 4 hours on limited GPU.
- Inspired by the [Create a Large Language Model from Scratch with Python](https://www.youtube.com/watch?v=UU1WVnMk4E8&t=19003s).

---

## 🔧 Model Specs

| Parameter       | Value         |
|----------------|---------------|
| Architecture   | GPT (Transformer Decoder) |
| n_layer        | 8             |
| n_head         | 8             |
| n_embd         | 384           |
| block_size     | 128           |
| batch_size     | 32            |
| dropout        | 0.2           |
| learning_rate  | 1e-4          |
| max_iters      | 10000         |

---

## 📉 Training Logs (Sample)

step: 0, train loss: 4.294, val loss: 4.325
step: 200, train loss: 4.212, val loss: 4.257
step: 400, train loss: 4.174, val loss: 4.232
step: 600, train loss: 4.111, val loss: 4.167
step: 800, train loss: 4.075, val loss: 4.169
step: 1000, train loss: 4.000, val loss: 4.134
step: 1200, train loss: 3.996, val loss: 4.073
step: 1400, train loss: 3.942, val loss: 4.051
step: 1600, train loss: 3.904, val loss: 4.071
step: 1800, train loss: 3.858, val loss: 4.061
step: 2000, train loss: 3.841, val loss: 4.015
step: 2200, train loss: 3.781, val loss: 3.937
step: 2400, train loss: 3.759, val loss: 3.983
step: 2600, train loss: 3.703, val loss: 3.943
step: 2800, train loss: 3.697, val loss: 3.935
step: 3000, train loss: 3.662, val loss: 3.963
step: 3200, train loss: 3.580, val loss: 3.892
step: 3400, train loss: 3.595, val loss: 3.903
step: 3600, train loss: 3.570, val loss: 3.867
step: 3800, train loss: 3.552, val loss: 3.861
step: 4000, train loss: 3.513, val loss: 3.869
step: 4200, train loss: 3.470, val loss: 3.823
step: 4400, train loss: 3.420, val loss: 3.831
step: 4600, train loss: 3.380, val loss: 3.857
step: 4800, train loss: 3.325, val loss: 3.832
step: 5000, train loss: 3.322, val loss: 3.782
step: 5200, train loss: 3.337, val loss: 3.794
step: 5400, train loss: 3.275, val loss: 3.788
step: 5600, train loss: 3.194, val loss: 3.765
step: 5800, train loss: 3.217, val loss: 3.790
step: 6000, train loss: 3.262, val loss: 3.739
step: 6200, train loss: 3.206, val loss: 3.749
step: 6400, train loss: 3.169, val loss: 3.717
step: 6600, train loss: 3.098, val loss: 3.731
step: 6800, train loss: 3.055, val loss: 3.763
step: 7000, train loss: 3.069, val loss: 3.712
step: 7200, train loss: 3.041, val loss: 3.702
step: 7400, train loss: 3.026, val loss: 3.692
step: 7600, train loss: 2.952, val loss: 3.713
step: 7800, train loss: 2.950, val loss: 3.679
step: 8000, train loss: 2.899, val loss: 3.715
step: 8200, train loss: 2.937, val loss: 3.747
step: 8400, train loss: 2.887, val loss: 3.719
step: 8600, train loss: 2.840, val loss: 3.704
step: 8800, train loss: 2.818, val loss: 3.764
step: 9000, train loss: 2.780, val loss: 3.722
step: 9200, train loss: 2.733, val loss: 3.716
step: 9400, train loss: 2.704, val loss: 3.709
step: 9600, train loss: 2.730, val loss: 3.637
step: 9800, train loss: 2.697, val loss: 3.674
2.935636520385742
model saved


## 📷 Test Logs (Sample)


(cuda) PS C:\LLM> python chatbot.py -batch_size 32
batch size: 32
cuda
loading model parameters...
loaded successfully!
Prompt:
You are stupid
cuda
loading model parameters...
loaded successfully!
Prompt:
You are stupid
Completion:
you are stupid at ? I'm sweating up a bit too slow down . I'm not sure . What were you glad you



Prompt:
what is your name?
Completion:
what is your name ? Tim , your mother looks wonderful in China . Oh wow , that sounds fantastic ! What a beautiful

Prompt:
can you give some cakes?
Completion:
can you give some cakes ? We have reduced emission of plus a project . I've got many items too many things to do .

Prompt:
