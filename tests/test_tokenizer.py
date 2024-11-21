from src.tokenizer.tokenizer import ChessTokenizer
import unittest

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ChessTokenizer()
        self.tokenizer.load_tokenizer('src/tokenizer/vocab.json')

    def test_encode(self):
        # I trust normal encoding works, so im just going to test edge cases
        unk_id = self.tokenizer.encode(['[UNK]'])[0]

        # Unknown tokens get encoded
        self.assertEqual(unk_id, self.tokenizer.encode(['does not exist'])[0])
        # Real tokens get encoded
        self.assertNotEqual(unk_id, self.tokenizer.encode(['e4'])[0])

    def test_encode_and_pad(self):
        pad_id = self.tokenizer.encode(['[PAD]'])[0]
        self.assertEqual(self.tokenizer.encode_and_pad([],1)[0], pad_id)

        # Empty pad
        self.assertEqual(self.tokenizer.encode_and_pad([], 3).all(), self.tokenizer.encode_and_pad(["[PAD]"]*3, 3).all())

        # Context too long
        self.assertEqual(self.tokenizer.encode_and_pad(["e4"]*10, 5).all(), self.tokenizer.encode_and_pad(["e4"]*5, 5).all())

    def test_decode(self):
        pass