import unittest
from joint_nlu import data_preprocessing

class TestUtils(unittest.TestCase):

    def test_convert_to_bio(self):
        test_cases = [
            ("Book a [flight] from [city : New York] to [city : Los Angeles]",
             "o o o o b-city i-city o b-city i-city"),
            ("[greeting] everyone at [location : the office]", "o o o b-location i-location"),
            ("Just a regular sentence without slots", "o o o o o o"),
            ("[business_type : wine shop]", "b-business_type i-business_type"),
            ("[currency : u. s. d.] position now", "b-currency i-currency i-currency o o"),
            ("any new emails after [time : five p.m.]", "o o o o b-time i-time"),
            ("[movie-title: Star Wars: A New Hope] is great",
             "b-movie-title i-movie-title i-movie-title i-movie-title i-movie-title o o"),
            ("Set an alarm for [time: 9:00am], please.", "o o o o b-time o"),
            ("number in [slot : 123]", "o o b-slot"),
        ]

        for sentence, expected_output in test_cases:
            self.assertEqual(data_preprocessing.convert_to_bio(sentence), expected_output)

    def test_convert_to_flattag(self):
        sentence = "Book a [flight] from [city : New York] to [city : Los Angeles]"
        expected_output = 'o o o o city city o city city'
        self.assertEqual(data_preprocessing.convert_to_flattag(sentence), expected_output)

    def test_get_all_iob_tokens(self):
        dataset = ["B-person O B-location I-location", "O B-person O O"]
        expected_output = ['B-person', 'O', 'B-location', 'I-location']
        self.assertEqual(data_preprocessing.get_all_iob_tokens(dataset), expected_output)

    def test_pad_or_truncate_labels(self):
        label_ids = [1, 2, 3]
        max_length = 5
        expected_output = [1, 2, 3, -100, -100]
        self.assertEqual(data_preprocessing.pad_or_truncate_labels(label_ids, max_length), expected_output)

if __name__ == '__main__':
    unittest.main()
