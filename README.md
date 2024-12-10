# Chess Transformer Project

Custom transformer written in Jax to predict the next chess move in a given sequence :)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/stackviolator/todo-lol
   cd todo-lol
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Training the Model

The `train.py` script is used to train the Transformer model. Below is an example usage:

   ```bash
   python train.py -c configs/transformer_dev.cfg -a configs/training_args.cfg -t src/tokenizer/vocab.json -d
   ```

---

## Generating Moves

The `generate.py` script generates chess moves using the trained Transformer model.

### Example Usage

1. **Prepare the Model:**
   Ensure the trained model is saved in `trained_models/` and matches the configuration in `configs/transformer_inference.cfg`.

2. **Run the Script:**
   ```bash
   python generate.py -m trained_models/dev -o output/temp.txt -d -k 5
   ```

3. **View Results:**
   The generated move(s) will be printed to the console or saved to a specified output file.

---

## Key Components

### Tokenizer
- Found in `src/tokenizer/`.
- Custom tokenizer, maps a move in SAN notation to an integer. Trained with `train_tokenizer.py`

### Dataset
- Found in `src/dataset/`.
- `GamesDataset.py` Custom dataset for handling chess moves -- loading and batching for training.

### Model
- Found in `src/model/`.
- `Transformer.py` defines the Transformer architecture.

### Sampler
- Found in `src/sampler/`.
- `Sampler.py` implements logic for sampling moves during inference.

---

## Testing

Unit tests are provided in the `tests/` directory. To run the tests:
```bash
python -m unittest tests/{file}
```
