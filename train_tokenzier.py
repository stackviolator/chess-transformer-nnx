from src.tokenizer.tokenizer import ChessTokenizer

if __name__ == "__main__":
    tokenizer = ChessTokenizer()
    tokenizer.train()
    tokenizer.save_tokenizer('src/tokenizer/vocab_new.json')
