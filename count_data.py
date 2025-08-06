# 📁 Image Split Summary:

# 🟢 Images to train for 'cat': 8749
# 🟢 Images to train for 'dog': 8749
# 🟢 Images to val for 'cat': 1874
# 🟢 Images to val for 'dog': 1874
# 🟢 Images to test for 'cat': 1876
# 🟢 Images to test for 'dog': 1876


import os

base_path = 'data'
splits = ['train', 'val', 'test']
classes = ['cat', 'dog']

print("📁 Image Split Summary:\n")

for split in splits:
    for cls in classes:
        folder = os.path.join(base_path, split, cls)
        count = len(os.listdir(folder))
        print(f"🟢 Images to {split} for '{cls}': {count}")
