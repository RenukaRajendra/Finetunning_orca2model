import logging
import pandas as pd
import yaml
from ludwig.api import LudwigModel
import matplotlib.pyplot as plt

qa_pairs = [
    {"Question": "Who is the founder of Microsoft?", "Answer": "Bill Gates"},
    {"Question": "What is the tallest building in the world?", "Answer": "Burj Khalifa"},
    {"Question": "What is the currency of Brazil?", "Answer": "Real"},
    {"Question": "What is the boiling point of mercury in Celsius?", "Answer": "-38.83"},
    {"Question": "What is the most commonly spoken language in the world?", "Answer": "Mandarin"},
    {"Question": "What is the diameter of the Earth?", "Answer": "12,742 km"},
    {"Question": 'Who wrote the novel "1984"?', "Answer": "George Orwell"},
    {"Question": "What is the name of the largest moon of Neptune?", "Answer": "Triton"},
    {"Question": "What is the speed of light in meters per second?", "Answer": "299,792,458 m/s"},
    {"Question": "What is the smallest country in Africa by land area?", "Answer": "Seychelles"},
    {"Question": "What is the largest organ in the human body?", "Answer": "Skin"},
    {"Question": 'Who directed the film "The Godfather"?', "Answer": "Francis Ford Coppola"},
    {"Question": "What is the name of the smallest planet in our solar system?", "Answer": "Mercury"},
    {"Question": "What is the largest lake in Africa?", "Answer": "Lake Victoria"},
    {"Question": "What is the smallest country in Asia by land area?", "Answer": "Maldives"},
    {"Question": "What is the chemical symbol for gold?", "Answer": "Au"},
    {"Question": "What is the name of the famous Swiss mountain known for skiing?", "Answer": "The Matterhorn"},
    {"Question": "What is the largest flower in the world?", "Answer": "Rafflesia arnoldii"},
    {"Question": "What is the capital of Japan?", "Answer": "Tokyo"},
    {"Question": "Who painted the Mona Lisa?", "Answer": "Leonardo da Vinci"},
    {"Question": "What is the chemical formula for water?", "Answer": "H2O"},
    {"Question": "What is the distance from the Earth to the Moon?", "Answer": "384,400 km"},
    {"Question": "Who is the author of 'Pride and Prejudice'?", "Answer": "Jane Austen"},
    {"Question": "What is the longest river in the world?", "Answer": "Nile River"},
    {"Question": "Who discovered penicillin?", "Answer": "Alexander Fleming"},
    {"Question": "What is the currency of Japan?", "Answer": "Yen"},
    {"Question": "What is the largest desert in the world?", "Answer": "Sahara Desert"},
    {"Question": "What is the speed of sound in air?", "Answer": "343 m/s"},
    {"Question": "What is the name of the largest ocean?", "Answer": "Pacific Ocean"},
    {"Question": "Who wrote 'Romeo and Juliet'?", "Answer": "William Shakespeare"},
    {"Question": "What is the capital of France?", "Answer": "Paris"},
    {"Question": "What is the chemical symbol for iron?", "Answer": "Fe"},
    {"Question": "What is the largest mammal in the world?", "Answer": "Blue Whale"},
    {"Question": "What is the hardest natural substance on Earth?", "Answer": "Diamond"},
    {"Question": "What is the highest mountain in the world?", "Answer": "Mount Everest"},
    {"Question": "What is the main ingredient in glass?", "Answer": "Silica (sand)"},
    {"Question": "What is the currency of the United Kingdom?", "Answer": "Pound Sterling"},
    {"Question": "Who invented the telephone?", "Answer": "Alexander Graham Bell"},
    {"Question": "What is the capital of Canada?", "Answer": "Ottawa"},
    {"Question": "What is the largest planet in our solar system?", "Answer": "Jupiter"},
    {"Question": "What is the national animal of Australia?", "Answer": "Kangaroo"},
    {"Question": "What is the pH of pure water?", "Answer": "7"},
    {"Question": "Who is known as the father of modern physics?", "Answer": "Albert Einstein"},
    {"Question": "What is the deepest part of the world's oceans?", "Answer": "Mariana Trench"},
    {"Question": "What is the tallest mountain in Africa?", "Answer": "Mount Kilimanjaro"},
    {"Question": "What is the most abundant gas in the Earth's atmosphere?", "Answer": "Nitrogen"},
    {"Question": "What is the capital of Italy?", "Answer": "Rome"},
    {"Question": "What is the formula for calculating the area of a circle?", "Answer": "πr²"},
    {"Question": "Who was the first woman to fly solo across the Atlantic?", "Answer": "Amelia Earhart"},
    {"Question": "What is the main language spoken in Brazil?", "Answer": "Portuguese"},
    {"Question": "What is the capital of Russia?", "Answer": "Moscow"},
    {"Question": "What is the largest reptile in the world?", "Answer": "Saltwater Crocodile"},
    {"Question": "What is the smallest bone in the human body?", "Answer": "Stapes (ear bone)"},
    {"Question": "What is the capital of Egypt?", "Answer": "Cairo"},
    {"Question": "Who was the first person to step on the Moon?", "Answer": "Neil Armstrong"},
    {"Question": "What is the main ingredient in chocolate?", "Answer": "Cocoa Beans"},
    {"Question": "What is the currency of China?", "Answer": "Yuan"},
    {"Question": "What is the name of the longest river in Asia?", "Answer": "Yangtze River"}
]

df = pd.DataFrame(qa_pairs)

def plot_sequence_lengths(df):
    sequence_lengths = []
    too_long = []
    for idx, row in df.iterrows():
        seq_length = len(row['Question']) + len(row['Answer'])
        sequence_lengths.append(seq_length)
        if seq_length > 512:  # adjust threshold as needed
            too_long.append(idx)

    plt.hist(sequence_lengths, bins=30)
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title('Distribution of Text Sequence Lengths')
    plt.savefig('plot_sequence_lengths.png')
    return too_long

plot_sequence_lengths(df)

config = yaml.safe_load(
    """
        input_features:
            - name: Question
              type: text
        output_features:
            - name: Answer
              type: text
        model_type: llm
        preprocessing:
            max_sequence_length: 18
        generation:
            temperature: 0.1
            top_p: 0.75
            top_k: 40
            num_beams: 4
            max_new_tokens: 5
        base_model: microsoft/Orca-2-7b
        quantization:
            bits: 4
        adapter:
            type: lora
        trainer:
            type: finetune
            learning_rate: 0.0001
            batch_size: 1
            gradient_accumulation_steps: 16
            epochs: 3
            learning_rate_scheduler:
                warmup_fraction: 0.01
    """
)

model = LudwigModel(config, logging_level=logging.INFO)

# Train the model
(
    train_stats,  
    preprocessed_data,  
    output_directory,  
) = model.train(
    dataset=df, experiment_name="simple_experiment", model_name="simple_model", skip_save_processed_input=True
)

model.save("results")

# batch prediction
training_set, val_set, test_set, _ = preprocessed_data
preds, _ = model.predict(test_set, skip_save_predictions=False)
# print(preds.iloc[0].to_string())
# print(preds.iloc[1].to_string())