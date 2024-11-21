from src.tokenizer.tokenizer import ChessTokenizer
import unittest

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ChessTokenizer()
        self.tokenizer.load_tokenizer('src/tokenizer/vocab.json')
        self.test_game = ["<|startofgame|>", "e2e4", "c7c5", "g1f3", "d7d6", "f1b5", "c8d7", "d1e2", "g8f6", "b2b3", "e7e6", "c1b2", "f8e7", "e4e5", "d6e5", "f3e5", "e8g8", "e1g1", "a7a6", "e5d7", "b8d7", "b5d3", "b7b5", "a2a4", "c5c4", "b3c4", "b5b4", "d3e4", "a8b8", "d2d3", "a6a5", "b1d2", "d7c5", "b2e5", "b8b6", "e4f3", "d8d7", "d2b3", "b6a6", "b3c5", "e7c5", "d3d4", "c5e7", "f1d1", "f8c8", "c4c5", "a6a7", "e2b5", "f6d5", "b5d7", "a7d7", "d1d3", "f7f6", "f3g4", "g8f7", "e5g3", "f6f5", "g4h5", "g7g6", "h5f3", "e7f6", "g3d6", "d7d6", "c5d6", "c8c2", "f3d5", "e6d5", "a1e1", "c2c6", "d6d7", "c6d6", "e1e8", "d6d7", "e8a8", "f6d8", "g2g3", "f7e6", "a8a6", "e6f7", "g1g2", "g6g5", "g2f3", "d8c7", "a6a7", "g5g4", "f3g2", "f7e6", "a7b7", "e6d6", "b7b5", "d7e7", "g2f1", "e7e4", "b5b7", "h7h5", "b7b5", "f5f4", "f2f3", "g4f3", "g3f4", "e4f4", "f1f2", "c7d8", "b5b7", "d8h4", "f2f1", "f3f2", "b7b6", "1-0", "<|endofgame|>"]

    def test_encode(self):
        # I trust normal encoding works, so im just going to test edge cases
        unk_id = self.tokenizer.encode(['[UNK]'])[0]

        # Unknown tokens get encoded
        self.assertEqual(unk_id, self.tokenizer.encode(['does not exist'])[0])
        # Real tokens get encoded
        self.assertNotEqual(unk_id, self.tokenizer.encode(['e2e4'])[0])

    def test_encode_and_pad(self):
        pad_id = self.tokenizer.encode(['[PAD]'])[0]
        self.assertEqual(self.tokenizer.encode_and_pad([],1)[0], pad_id)

        # Empty pad
        self.assertEqual(self.tokenizer.encode_and_pad([], 3).all(), self.tokenizer.encode_and_pad(["[PAD]"]*3, 3).all())

        # Context too long
        self.assertEqual(self.tokenizer.encode_and_pad(["e2e4"]*10, 5).all(), self.tokenizer.encode_and_pad(["e2e4"]*5, 5).all())

    def test_decode(self):
        pass