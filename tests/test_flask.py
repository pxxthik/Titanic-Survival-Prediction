import unittest
from flask.app import app


class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Titanic Survival Prediction</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data=dict(
            pclass=1,
            sex='male',
            age=30,
            fare=140,
            embarked="Q",
            familysize=3,
            title="Master"
        ))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'SURVIVED!' in response.data or b'DID NOT SURVIVE' in response.data,
            "Response should contain either 'SURVIVED!' or 'DID NOT SURVIVE'"
        )


if __name__ == '__main__':
    unittest.main()
