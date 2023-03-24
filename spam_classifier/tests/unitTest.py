import unittest
from spam_classifier.Models.NaiveBayes import NaiveBayes


class TestSpamOrHamModel(unittest.TestCase):

    def setUp(self):
        self.nb = NaiveBayes()
        self.nb.run_model()

    def test_spam(self):
        test_string = "Congratulations! You've won a free trip to Hawaii. Click here to claim your prize."
        message = f"Expected {self.nb.predict_text(test_string)} for string: {test_string}"
        self.assertEqual(self.nb.predict_text(test_string), 'spam', message)

    def test_ham(self):
        test_string = "Hi John, can you please send me the report by end of day? Thanks!"
        message = f"Expected {self.nb.predict_text(test_string)} for string: {test_string}"
        self.assertEqual(self.nb.predict_text(test_string), 'ham', message)

    def test_spam2(self):
        test_string = "Get rich quick! Make money fast with this amazing opportunity."
        message = f"Expected {self.nb.predict_text(test_string)} for string: {test_string}"
        self.assertEqual(self.nb.predict_text(test_string), 'spam', message)

    def test_ham2(self):
        test_string = "Click here to check all sexy girls around you."
        message = f"Expected {self.nb.predict_text(test_string)} for string: {test_string}"
        self.assertEqual(self.nb.predict_text(test_string), 'spam', message)

    def test_spam3(self):
        test_string = "Click this link to get your free gift. https://www.google.com"
        message = f"Expected {self.nb.predict_text(test_string)} for string: {test_string}"
        self.assertEqual(self.nb.predict_text(test_string), 'spam', message)

    def test_ham3(self):
        test_string = "My name is Aneruth and I am a Data Scientist."
        message = f"Expected {self.nb.predict_text(test_string)} for string: {test_string}"
        self.assertEqual(self.nb.predict_text(test_string), 'ham', message)


if __name__ == '__main__':
    unittest.main(verbosity=2)
