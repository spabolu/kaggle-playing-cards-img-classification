# Playing Card Image Classification

A machine learning project that classifies images of playing cards into their exact identity (e.g., "Ace of Hearts", "King of Spades") using convolutional neural networks (CNNs) and transfer learning.  

This project was developed as part of the **DAT 402: ML for Data Science** course at Arizona State University.

---

## Dataset

- **Source**: [Cards Image Dataset - Classification (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)  
- **Size**:  
  - **Train**: 7,624 images  
  - **Validation**: 265 images  
  - **Test**: 265 images  
- **Image Shape**: 224×224×3 (RGB)  
- **Classes**: Standard 52-card deck + Joker (Joker class dropped for final modeling)

**Goal**: Build an accurate model to recognize the card’s suit and rank from a photo, even under variations in orientation, lighting, and background.

---

## Approach

### **Preprocessing**

- Loaded images using `Path` and `ImageDataGenerator` for efficient training and augmentation.
- Renamed class folders to replace spaces with underscores for easier file handling.
- Dropped **Joker** class to match a standard 52-card deck.
- Applied **data augmentation**:
  - Rotation
  - Zoom
  - Horizontal/vertical flips
- Normalized pixel values to `[0,1]`.

### **Model A: CNN from Scratch**

- Built a Sequential CNN with multiple Conv2D + MaxPooling layers.
- Activation: ReLU for hidden layers, Softmax for output.
- Optimizer: Adam  
- Loss: Categorical Crossentropy
- Early stopping based on validation loss.
- **Test Accuracy**: ~94%

### **Model B: Transfer Learning with MobileNetV2**

- Used pretrained MobileNetV2 (ImageNet weights) as feature extractor.
- Fine-tuned last few layers for card-specific classification.
- Added GlobalAveragePooling + Dense output layer (Softmax).
- **Test Accuracy**: ~98%
- Outperformed CNN-from-scratch in both accuracy and convergence speed.

---

## 📈 Results

| Model                      | Metric      | Score   |
|----------------------------|-------------|---------|
| **Random Forest**          | Accuracy    | ~47%    |
| **ResNet50 Transfer Learning**   | Accuracy    | ~56%    |
| **Custom CNN**             | Accuracy    | ~80%    |

---

## Tools & Libraries

- Python  
- TensorFlow / Keras  
- pandas  
- numpy  
- matplotlib  
- pathlib

---

## How to Run

```bash
git clone https://github.com/spabolu/kaggle-playing-cards-img-classification.git
cd kaggle-playing-cards-img-classification
pip install -r requirements.txt # need to be pushed
jupyter notebook project2.ipynb
```

---

## Future Improvements

- Add multi-output modeling (predict rank & suit separately).
