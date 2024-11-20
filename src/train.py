args = TransformerTrainingArgs()

dataset = GamesDataset("chess_games.csv", tokenizer, context_length=128)
dataset_dict = dataset.train_test_split(test_size=1000)

train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=GamesDataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=GamesDataset.collate_fn)

args = TransformerTrainingArgs(epochs=15, max_steps_per_epoch=500)
trainer = TransformerTrainer(args, transformer, train_loader=train_loader, test_loader=test_loader)
trainer.train()