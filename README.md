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

step: 9000, train loss: 2.780, val loss: 3.722
step: 9200, train loss: 2.733, val loss: 3.716
step: 9400, train loss: 2.704, val loss: 3.709
step: 9600, train loss: 2.730, val loss: 3.637
step: 9800, train loss: 2.697, val loss: 3.674
2.935636520385742


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
