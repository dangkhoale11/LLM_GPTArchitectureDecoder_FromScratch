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

## 📈 Training Progress

| Steps  | Train Loss | Val Loss | 
|--------|------------|----------|
| 0      | 4.294      | 4.325    |
| 1,000  | 4.000      | 4.134    | 
| 2,000  | 3.841      | 4.015    | 
| 3,000  | 3.662      | 3.963    |
| 4,000  | 3.513      | 3.869    |
| 5,000  | 3.322      | 3.782    |
| 6,000  | 3.262      | 3.739    |
| 7,000  | 3.069      | 3.712    |
| 8,000  | 2.899      | 3.715    |
| 9,000  | 2.780      | 3.722    |
| 9,800  | 2.697      | 3.674    | 


## 💬 Example Outputs

**Prompt:** You are stupid  
**Completion:** you are stupid at ? I'm sweating up a bit too slow down . I'm not sure . What were you glad you

**Prompt:** what is your name?  
**Completion:** what is your name ? Tim , your mother looks wonderful in China . Oh wow , that sounds fantastic ! What a beautiful

**Prompt:** can you give some cakes?  
**Completion:** can you give some cakes ? We have reduced emission of plus a project . I've got many items too many things to do .
