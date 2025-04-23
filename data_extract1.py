from datasets import load_dataset

# Load dataset
dataset = load_dataset('./daily_dialog.py', trust_remote_code=True)

# Output files
output_file_train = 'output_train.txt'
output_file_val = 'output_val.txt'
vocab_file = 'vocab.txt'

data_train = dataset['train']
data_val = dataset['validation']

# Hàm xử lý và lưu các câu trong dialog vào file
def save_dialogs(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in data:
            dialog = sample['dialog']
            for sentence in dialog:
                f.write(sentence.strip() + '\n')
            f.write('\n')  # ngăn cách giữa các đoạn hội thoại

# Lưu dialog của tập train và val
save_dialogs(data_train, output_file_train)
save_dialogs(data_val, output_file_val)

# Tạo vocab từ tập train
vocab_set = set()
for sample in data_train:
    for sentence in sample['dialog']:
        words = sentence.strip().split()
        vocab_set.update(words)

# Sắp xếp và lưu vocab
with open(vocab_file, 'w', encoding='utf-8') as f:
    for word in sorted(vocab_set):
        f.write(word + '\n')

print(f"Train dialogues saved to {output_file_train}")
print(f"Validation dialogues saved to {output_file_val}")
print(f"Vocabulary saved to {vocab_file}")
