"""Exercise Streamlit reruns and the main interaction branches without network."""
import unittest
from streamlit.testing.v1 import AppTest

class UserFlowTests(unittest.TestCase):
    def test_views_profiles_and_pairs(self):
        a=AppTest.from_file('app.py',default_timeout=30).run()
        self.assertFalse(a.exception)
        self.assertEqual(len(a.get('plotly_chart')),2)
        self.assertEqual(a.selectbox(key='frequency').value,'Monthly')
        self.assertEqual(a.selectbox(key='frequency').options,['Daily','Monthly','Annual'])
        self.assertEqual(len(a.date_input),0)
        next(x for x in a.selectbox if x.label=='From year').set_value(1980).run()
        self.assertFalse(a.exception)
        self.assertEqual(next(x for x in a.selectbox if x.label=='From year').value,1980)
        next(x for x in a.button if x.label=='All').click().run()
        self.assertFalse(a.exception)
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
        a.selectbox(key='basis').set_value('FRED · previous homework').run()
        self.assertFalse(a.exception)
        self.assertEqual(a.selectbox(key='reference_G163USA_True').value,'Course assumption · January 1999')
        self.assertEqual(next(x for x in a.selectbox if x.label=='From year').value,1999)
        self.assertTrue(any('Possibly undervalued' in h.value for h in a.markdown))
        for c in ['CHL','PER']:
            a.selectbox(key='c1').set_value(c).run()
            self.assertFalse(a.exception)
            self.assertEqual(len(a.get('plotly_chart')),2)
            a.selectbox(key='frequency').set_value('Daily').run()
            self.assertFalse(a.exception)
            self.assertEqual(len(a.get('plotly_chart')),2)
            self.assertEqual(len(a.date_input),0)
            self.assertIn('January',next(x for x in a.selectbox if x.label=='From month').options)
        a.selectbox(key='frequency').set_value('Monthly').run()
        a.selectbox(key='c1').set_value('FRA').run()
        a.selectbox(key='c2').set_value('DEU').run()
        self.assertFalse(a.exception)
        a.selectbox(key='c1').set_value('NPL').run()
        a.selectbox(key='frequency').set_value('Daily').run()
        self.assertFalse(a.exception)
        self.assertTrue(a.error)

if __name__=='__main__': unittest.main()
