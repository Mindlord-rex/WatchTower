# WatchTower
WatchTower is a text-based person search system for surveillance footage. It combines object detection (YOLOv8) with vision-language models (CLIP) to let you find frames using natural language queries like “person in red jacket” or “someone carrying a backpack”.

# Features
- Processes video files and extracts people using YOLOv8

- Generates CLIP embeddings for each detected person

- Stores embeddings in Chroma for fast similarity search

- Query by text – get the most relevant frames with bounding boxes
