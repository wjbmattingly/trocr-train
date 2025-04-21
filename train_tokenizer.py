from transformers import BertTokenizerFast
import datasets
from transformers import TrOCRProcessor, ViTImageProcessor

# Load your Yiddish dataset
dataset = datasets.load_dataset("johnlockejrr/yiddish_synth", split="train")

# Print dataset structure to understand format
# print("Dataset features:", dataset.features)
# print("Sample item:", dataset[0])

# Function to get all texts from your dataset
def get_all_texts():
    for item in dataset:
        # Check the actual structure of your dataset
        if isinstance(item, dict) and "text" in item:
            yield item["text"]
        elif isinstance(item, str):
            yield item
        else:
            # Print the problematic item to diagnose
            print(f"Unexpected item format: {type(item)}, content: {item}")
            continue

# Create and train a new tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=False)
tokenizer = tokenizer.train_new_from_iterator(get_all_texts(), vocab_size=30000)

# Save the tokenizer
tokenizer.save_pretrained("yiddish_tokenizer")

# Get the image processor from the original model
image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Create a new processor with your tokenizer
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
processor.save_pretrained("yiddish_processor")