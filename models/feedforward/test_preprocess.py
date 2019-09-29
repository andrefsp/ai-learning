import unittest
from .preprocess import preprocess


class PreprocessingTestCase(unittest.TestCase):

    def test_preprocess_string_and_array(self):
        self.assertEqual(
            preprocess('andre')[0].tolist(),
            preprocess(['andre'])[0].tolist()
        )

    def test_preprocess_many(self):
        arrays = preprocess(['andre', 'filipe'])
        self.assertEqual(len(arrays), 2)
