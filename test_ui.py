"""Exercise Streamlit reruns and the main interaction branches without network."""
import unittest
from streamlit.testing.v1 import AppTest

class UserFlowTests(unittest.TestCase):
    def test_views_profiles_and_pairs(self):
        a=AppTest.from_file('app.py',default_timeout=30).run()
        self.assertFalse(a.exception)
        self.assertEqual(len(a.get('plotly_chart')),2)
        for freq in ['Annual','Daily','Monthly']:
            a.selectbox(key='frequency').set_value(freq).run()
            self.assertFalse(a.exception)
            self.assertEqual(len(a.get('plotly_chart')),2)
        for profile in ['Trader / investor','Company · payments / debt in C1','Company · receipts / assets in C1']:
            a.selectbox(key='profile').set_value(profile).run()
            self.assertFalse(a.exception)
        a.selectbox(key='c1').set_value('G163').run()
        self.assertFalse(a.exception)
        self.assertEqual(len(a.get('plotly_chart')),2)
        a.selectbox(key='reference_G163USA').set_value('Course assumption · January 1999').run()
        self.assertFalse(a.exception)
        a.selectbox(key='c1').set_value('FRA').run()
        a.selectbox(key='c2').set_value('DEU').run()
        self.assertFalse(a.exception)
        a.selectbox(key='c1').set_value('ARG').run()
        a.selectbox(key='frequency').set_value('Daily').run()
        self.assertFalse(a.exception)
        self.assertTrue(a.error)

if __name__=='__main__': unittest.main()
