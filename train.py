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


    pad_token_id = int(tokenizer.encode(["[PAD]"])[0])

    # The model config and model itself
    cfg = TransformerConfig(
        d_model=768,
        d_vocab=len(tokenizer.tokens.values()),
        d_head=64,
        d_mlp=3072,
        n_heads=12,
        n_layers=12,
        ctx_len=128,
        pad_token_id=pad_token_id,
        ckpt_dir="trained_models/dev"
    )

    '''
    Note on ctx_len:
    this will prob need to be played with. since there is a bunch of padding.. however this might not be a big deal if i implement masked loss
    '''

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
    try:
        trainer.train()
    except:
        print(f"Exception occured")

    # Save the model
    transformer.save()

    # Test loading the model
    print("testing load :)...")
    test_model = transformer.load(cfg.ckpt_dir)
