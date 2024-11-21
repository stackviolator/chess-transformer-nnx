from src.dataset.GamesDataset import GamesDataset
from src.model.Trainer import TransformerTrainingArgs, TransformerTrainer
from src.model.Transformer import Transformer, TransformerConfig
from src.tokenizer.tokenizer import ChessTokenizer
from torch.utils.data import DataLoader
import warnings
import sys

train_file = 'data/clean/games_data.csv'

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Tokenizer
    tokenizer = ChessTokenizer()
    tokenizer.load_tokenizer("src/tokenizer/vocab.json")

    # The model config and model itself
    cfg = TransformerConfig(
        d_model=768,
        d_vocab=len(tokenizer.tokens.values()),
        d_head=64,
        d_mlp=3072,
        n_heads=12,
        n_layers=12,
        ctx_len=100
    )

    # Traning args
    args = TransformerTrainingArgs(
        epochs=15,
        max_steps_per_epoch=500,
        debug=False,
    )

    transformer = Transformer(cfg)

    # Dataset and loaders
    dataset = GamesDataset(train_file, tokenizer, context_length=cfg.ctx_len)
    dataset_dict = dataset.train_test_split(test_size=1000)

    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=GamesDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=GamesDataset.collate_fn)


    # Train the model
    trainer = TransformerTrainer(args, transformer, train_loader=train_loader, test_loader=test_loader)
    trainer.train()
