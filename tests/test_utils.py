import unittest
from joint_nlu import utils

class TestUtils(unittest.TestCase):

    def test_convert_to_bio(self):
        sentence = "Book a [flight] from [city : New York] to [city : Los Angeles]"
        expected_output = 'O O B-flight O B-city I-city O O B-city I-city'
        self.assertEqual(utils.convert_to_bio(sentence), expected_output)

    def test_convert_to_flattag(self):
        sentence = "Book a [flight] from [city : New York] to [city : Los Angeles]"
        expected_output = 'O O flight O city city O O city city'
        self.assertEqual(utils.convert_to_flattag(sentence), expected_output)

    def test_get_all_iob_tokens(self):
        dataset = ["B-person O B-location I-location", "O B-person O O"]
        expected_output = ['B-person', 'O', 'B-location', 'I-location']
        self.assertEqual(utils.get_all_iob_tokens(dataset), expected_output)

    def test_pad_or_truncate_labels(self):
        label_ids = [1, 2, 3]
        max_length = 5
        expected_output = [1, 2, 3, -100, -100]
        self.assertEqual(utils.pad_or_truncate_labels(label_ids, max_length), expected_output)

if __name__ == '__main__':
    unittest.main()
