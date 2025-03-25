![zddfdxzdf.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/MoMzO3b8sULhwKf45yfrj.png)

# **Multisource-121-DomainNet**

> **Multisource-121-DomainNet** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify images into 121 domain categories using the **SiglipForImageClassification** architecture.

![- visual selection(2).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/yfp_IYXqDyZfgZsJQ-7Bo.png)

*Moment Matching for Multi-Source Domain Adaptation* : https://arxiv.org/pdf/1812.01754

*SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features* https://arxiv.org/pdf/2502.14786

```py
Classification Report:
                         precision    recall  f1-score   support

                   barn     0.7483    0.8370    0.7902       270
           baseball_bat     0.9197    0.9333    0.9265       270
                 basket     0.8302    0.8148    0.8224       270
                  beach     0.7059    0.7556    0.7299       270
                   bear     0.7500    0.7444    0.7472       270
                  beard     0.5496    0.5741    0.5616       270
                    bee     0.9004    0.9037    0.9020       270
                   bird     0.7352    0.7815    0.7576       270
              blueberry     0.7230    0.7926    0.7562       270
                 bowtie     0.8726    0.8370    0.8544       270
               bracelet     0.7328    0.7111    0.7218       270
                  brain     0.8925    0.9222    0.9071       270
                  bread     0.5573    0.6667    0.6071       270
               broccoli     0.9200    0.7667    0.8364       270
                    bus     0.8442    0.8630    0.8535       270
              butterfly     0.9321    0.9148    0.9234       270
                 circle     0.6038    0.8185    0.6950       270
                  cloud     0.8201    0.8444    0.8321       270
            cruise_ship     0.8545    0.8481    0.8513       270
                dolphin     0.8286    0.8593    0.8436       270
               dumbbell     0.8705    0.8963    0.8832       270
               elephant     0.8598    0.8630    0.8614       270
                    eye     0.8603    0.8667    0.8635       270
             eyeglasses     0.8425    0.7926    0.8168       270
                feather     0.8413    0.7852    0.8123       270
                   fish     0.8169    0.8593    0.8375       270
                 flower     0.7973    0.8741    0.8339       270
                   foot     0.8152    0.8333    0.8242       270
                   frog     0.9270    0.8000    0.8588       270
                giraffe     0.9026    0.8926    0.8976       270
                 goatee     0.5171    0.5037    0.5103       270
              golf_club     0.6466    0.6778    0.6618       270
                 grapes     0.8731    0.8407    0.8566       270
                  grass     0.7359    0.6296    0.6786       270
                 guitar     0.8386    0.8852    0.8613       270
              hamburger     0.8535    0.8630    0.8582       270
                   hand     0.7824    0.6926    0.7348       270
                    hat     0.7333    0.7741    0.7532       270
             headphones     0.8971    0.9037    0.9004       270
             helicopter     0.8992    0.8259    0.8610       270
                hexagon     0.9113    0.8370    0.8726       270
           hockey_stick     0.8419    0.8481    0.8450       270
                  horse     0.8081    0.8889    0.8466       270
              hourglass     0.9161    0.9296    0.9228       270
                  house     0.7524    0.8778    0.8103       270
              ice_cream     0.8821    0.8593    0.8705       270
                 jacket     0.8621    0.7407    0.7968       270
                 ladder     0.7051    0.8148    0.7560       270
                    leg     0.5916    0.5741    0.5827       270
               lipstick     0.8889    0.8000    0.8421       270
              megaphone     0.8710    0.9000    0.8852       270
                 monkey     0.8370    0.8556    0.8462       270
                   moon     0.8527    0.8148    0.8333       270
               mushroom     0.8774    0.8481    0.8625       270
               necklace     0.8670    0.7481    0.8032       270
                    owl     0.9179    0.9111    0.9145       270
                  panda     0.9490    0.8963    0.9219       270
                   pear     0.8832    0.8963    0.8897       270
                   peas     0.7743    0.8259    0.7993       270
                penguin     0.8618    0.8778    0.8697       270
                    pig     0.6767    0.8296    0.7454       270
                 pillow     0.7359    0.6296    0.6786       270
              pineapple     0.9213    0.9111    0.9162       270
                  pizza     0.9173    0.9444    0.9307       270
                   pool     0.6717    0.6593    0.6654       270
               popsicle     0.7390    0.8074    0.7717       270
                 rabbit     0.8345    0.8778    0.8556       270
             rhinoceros     0.9219    0.9185    0.9202       270
                  rifle     0.9256    0.8296    0.8750       270
                  river     0.6067    0.7370    0.6656       270
               sailboat     0.8606    0.9148    0.8869       270
               sandwich     0.7638    0.7667    0.7652       270
             sea_turtle     0.8794    0.9185    0.8986       270
                  shark     0.8114    0.8444    0.8276       270
                   shoe     0.8097    0.8667    0.8372       270
             skyscraper     0.7727    0.8185    0.7950       270
                snorkel     0.8238    0.6926    0.7525       270
                snowman     0.8736    0.8444    0.8588       270
            soccer_ball     0.9395    0.8630    0.8996       270
              speedboat     0.7649    0.7593    0.7621       270
                 spider     0.9212    0.8222    0.8689       270
                  spoon     0.8165    0.8074    0.8119       270
                 square     0.4669    0.6259    0.5348       270
               squirrel     0.8394    0.7741    0.8054       270
            stethoscope     0.8566    0.8630    0.8598       270
             strawberry     0.8629    0.7926    0.8263       270
            streetlight     0.5000    0.6852    0.5781       270
              submarine     0.6850    0.6926    0.6888       270
               suitcase     0.8259    0.7556    0.7892       270
                    sun     0.8082    0.6556    0.7239       270
                sweater     0.5912    0.6963    0.6395       270
                  sword     0.8258    0.8074    0.8165       270
                  table     0.5502    0.5481    0.5492       270
                 teapot     0.9019    0.8852    0.8935       270
             teddy-bear     0.7906    0.8111    0.8007       270
              telephone     0.7836    0.7778    0.7807       270
                   tent     0.7579    0.7074    0.7318       270
       The_Eiffel_Tower     0.8633    0.8889    0.8759       270
The_Great_Wall_of_China     0.8893    0.8333    0.8604       270
          The_Mona_Lisa     0.8152    0.9148    0.8621       270
                  tiger     0.8577    0.8259    0.8415       270
                toaster     0.6788    0.6889    0.6838       270
                  tooth     0.8807    0.7926    0.8343       270
                tornado     0.7530    0.7000    0.7255       270
                tractor     0.9372    0.8296    0.8802       270
                  train     0.7692    0.7407    0.7547       270
                   tree     0.7639    0.8148    0.7885       270
               triangle     0.8852    0.8000    0.8405       270
               trombone     0.6653    0.5963    0.6289       270
                  truck     0.7049    0.7963    0.7478       270
                trumpet     0.7463    0.5667    0.6442       270
               umbrella     0.9144    0.8704    0.8918       270
                   vase     0.8148    0.7333    0.7719       270
                 violin     0.8966    0.7704    0.8287       270
             watermelon     0.7970    0.8000    0.7985       270
                  whale     0.7769    0.6963    0.7344       270
               windmill     0.8963    0.8963    0.8963       270
             wine_glass     0.8996    0.8630    0.8809       270
                   yoga     0.7406    0.8037    0.7709       270
                  zebra     0.9144    0.7519    0.8252       270
                 zigzag     0.6502    0.6333    0.6417       270

               accuracy                         0.7995     32670
              macro avg     0.8052    0.7995    0.8006     32670
           weighted avg     0.8052    0.7995    0.8006     32670
```




The model categorizes images into the following 121 classes:
- **Class 0:** "barn"  
- **Class 1:** "baseball_bat"  
- **Class 2:** "basket"  
- **Class 3:** "beach"  
- **Class 4:** "bear"  
- **Class 5:** "beard"  
- **Class 6:** "bee"  
- **Class 7:** "bird"  
- **Class 8:** "blueberry"  
- **Class 9:** "bowtie"  
- **Class 10:** "bracelet"  
- **Class 11:** "brain"  
- **Class 12:** "bread"  
- **Class 13:** "broccoli"  
- **Class 14:** "bus"  
- **Class 15:** "butterfly"  
- **Class 16:** "circle"  
- **Class 17:** "cloud"  
- **Class 18:** "cruise_ship"  
- **Class 19:** "dolphin"  
- **Class 20:** "dumbbell"  
- **Class 21:** "elephant"  
- **Class 22:** "eye"  
- **Class 23:** "eyeglasses"  
- **Class 24:** "feather"  
- **Class 25:** "fish"  
- **Class 26:** "flower"  
- **Class 27:** "foot"  
- **Class 28:** "frog"  
- **Class 29:** "giraffe"  
- **Class 30:** "goatee"  
- **Class 31:** "golf_club"  
- **Class 32:** "grapes"  
- **Class 33:** "grass"  
- **Class 34:** "guitar"  
- **Class 35:** "hamburger"  
- **Class 36:** "hand"  
- **Class 37:** "hat"  
- **Class 38:** "headphones"  
- **Class 39:** "helicopter"  
- **Class 40:** "hexagon"  
- **Class 41:** "hockey_stick"  
- **Class 42:** "horse"  
- **Class 43:** "hourglass"  
- **Class 44:** "house"  
- **Class 45:** "ice_cream"  
- **Class 46:** "jacket"  
- **Class 47:** "ladder"  
- **Class 48:** "leg"  
- **Class 49:** "lipstick"  
- **Class 50:** "megaphone"  
- **Class 51:** "monkey"  
- **Class 52:** "moon"  
- **Class 53:** "mushroom"  
- **Class 54:** "necklace"  
- **Class 55:** "owl"  
- **Class 56:** "panda"  
- **Class 57:** "pear"  
- **Class 58:** "peas"  
- **Class 59:** "penguin"  
- **Class 60:** "pig"  
- **Class 61:** "pillow"  
- **Class 62:** "pineapple"  
- **Class 63:** "pizza"  
- **Class 64:** "pool"  
- **Class 65:** "popsicle"  
- **Class 66:** "rabbit"  
- **Class 67:** "rhinoceros"  
- **Class 68:** "rifle"  
- **Class 69:** "river"  
- **Class 70:** "sailboat"  
- **Class 71:** "sandwich"  
- **Class 72:** "sea_turtle"  
- **Class 73:** "shark"  
- **Class 74:** "shoe"  
- **Class 75:** "skyscraper"  
- **Class 76:** "snorkel"  
- **Class 77:** "snowman"  
- **Class 78:** "soccer_ball"  
- **Class 79:** "speedboat"  
- **Class 80:** "spider"  
- **Class 81:** "spoon"  
- **Class 82:** "square"  
- **Class 83:** "squirrel"  
- **Class 84:** "stethoscope"  
- **Class 85:** "strawberry"  
- **Class 86:** "streetlight"  
- **Class 87:** "submarine"  
- **Class 88:** "suitcase"  
- **Class 89:** "sun"  
- **Class 90:** "sweater"  
- **Class 91:** "sword"  
- **Class 92:** "table"  
- **Class 93:** "teapot"  
- **Class 94:** "teddy-bear"  
- **Class 95:** "telephone"  
- **Class 96:** "tent"  
- **Class 97:** "The_Eiffel_Tower"  
- **Class 98:** "The_Great_Wall_of_China"  
- **Class 99:** "The_Mona_Lisa"  
- **Class 100:** "tiger"  
- **Class 101:** "toaster"  
- **Class 102:** "tooth"  
- **Class 103:** "tornado"  
- **Class 104:** "tractor"  
- **Class 105:** "train"  
- **Class 106:** "tree"  
- **Class 107:** "triangle"  
- **Class 108:** "trombone"  
- **Class 109:** "truck"  
- **Class 110:** "trumpet"  
- **Class 111:** "umbrella"  
- **Class 112:** "vase"  
- **Class 113:** "violin"  
- **Class 114:** "watermelon"  
- **Class 115:** "whale"  
- **Class 116:** "windmill"  
- **Class 117:** "wine_glass"  
- **Class 118:** "yoga"  
- **Class 119:** "zebra"  
- **Class 120:** "zigzag"

# **Run with Transformers🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Multisource-121-DomainNet"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def multisource_classification(image):
    """Predicts the domain category for an input image."""
    # Convert the input numpy array to a PIL Image and ensure it is in RGB format
    image = Image.fromarray(image).convert("RGB")
    
    # Process the image and convert it to model inputs
    inputs = processor(images=image, return_tensors="pt")
    
    # Get model predictions without gradient calculations
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    # Mapping from class indices to domain labels
    labels = {
        "0": "barn", "1": "baseball_bat", "2": "basket", "3": "beach", "4": "bear",
        "5": "beard", "6": "bee", "7": "bird", "8": "blueberry", "9": "bowtie",
        "10": "bracelet", "11": "brain", "12": "bread", "13": "broccoli", "14": "bus",
        "15": "butterfly", "16": "circle", "17": "cloud", "18": "cruise_ship", "19": "dolphin",
        "20": "dumbbell", "21": "elephant", "22": "eye", "23": "eyeglasses", "24": "feather",
        "25": "fish", "26": "flower", "27": "foot", "28": "frog", "29": "giraffe",
        "30": "goatee", "31": "golf_club", "32": "grapes", "33": "grass", "34": "guitar",
        "35": "hamburger", "36": "hand", "37": "hat", "38": "headphones", "39": "helicopter",
        "40": "hexagon", "41": "hockey_stick", "42": "horse", "43": "hourglass", "44": "house",
        "45": "ice_cream", "46": "jacket", "47": "ladder", "48": "leg", "49": "lipstick",
        "50": "megaphone", "51": "monkey", "52": "moon", "53": "mushroom", "54": "necklace",
        "55": "owl", "56": "panda", "57": "pear", "58": "peas", "59": "penguin",
        "60": "pig", "61": "pillow", "62": "pineapple", "63": "pizza", "64": "pool",
        "65": "popsicle", "66": "rabbit", "67": "rhinoceros", "68": "rifle", "69": "river",
        "70": "sailboat", "71": "sandwich", "72": "sea_turtle", "73": "shark", "74": "shoe",
        "75": "skyscraper", "76": "snorkel", "77": "snowman", "78": "soccer_ball", "79": "speedboat",
        "80": "spider", "81": "spoon", "82": "square", "83": "squirrel", "84": "stethoscope",
        "85": "strawberry", "86": "streetlight", "87": "submarine", "88": "suitcase", "89": "sun",
        "90": "sweater", "91": "sword", "92": "table", "93": "teapot", "94": "teddy-bear",
        "95": "telephone", "96": "tent", "97": "The_Eiffel_Tower", "98": "The_Great_Wall_of_China",
        "99": "The_Mona_Lisa", "100": "tiger", "101": "toaster", "102": "tooth", "103": "tornado",
        "104": "tractor", "105": "train", "106": "tree", "107": "triangle", "108": "trombone",
        "109": "truck", "110": "trumpet", "111": "umbrella", "112": "vase", "113": "violin",
        "114": "watermelon", "115": "whale", "116": "windmill", "117": "wine_glass", "118": "yoga",
        "119": "zebra", "120": "zigzag"
    }
    
    # Create a dictionary mapping each label to its corresponding probability (rounded)
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=multisource_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Multisource-121-DomainNet Classification",
    description="Upload an image to classify it into one of 121 domain categories."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

---

# **Intended Use:**

The **Multisource-121-DomainNet** model is designed for multi-source image classification. It can categorize images into a diverse set of 121 domains, covering various objects, scenes, and landmarks. Potential use cases include:

- **Cross-Domain Image Analysis:** Enabling robust classification across a wide range of visual domains.
- **Multimedia Retrieval:** Assisting in content organization and retrieval in multimedia databases.
- **Computer Vision Research:** Serving as a benchmark for evaluating domain adaptation and transfer learning techniques.
- **Interactive Applications:** Enhancing user interfaces with diverse, real-time image recognition capabilities.
