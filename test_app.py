"""Economic invariants and failure paths. Run: python -m unittest -v."""
import unittest
import numpy as np
import pandas as pd
import app

class EconomicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.m,_=app.monthly_pair('GBR','USA')

    def test_quote_and_rer_formula(self):
        f=self.m.dropna(subset=['rer'])
        np.testing.assert_allclose(f.nominal,1/f.fx1_per_usd)
        np.testing.assert_allclose(f.rer,f.nominal*f.cpi1/f.cpi2)

    def test_inverse_pair(self):
        r,_=app.monthly_pair('USA','GBR')
        f=self.m[['nominal','rer']].join(r[['nominal','rer']],rsuffix='_inverse').dropna()
        np.testing.assert_allclose(f.nominal*f.nominal_inverse,1)
        np.testing.assert_allclose(f.rer*f.rer_inverse,1)

    def test_mean_100_after_date_changes(self):
        means=[]
        for start,end in [('1995-01','2005-12'),('2015-01','2020-12'),('2020-01','2023-12')]:
            f,mean=app.normalize(self.m.loc[start:end]); means.append(mean)
            self.assertAlmostEqual(f.real_index.mean(),100,places=10)
        self.assertGreater(len(set(means)),1)

    def test_missing_month_and_annual_average(self):
        f=self.m.loc['2020-01':'2020-12'].copy()
        f.loc['2020-06-01',['cpi1','rer']]=np.nan
        n,_=app.normalize(f)
        self.assertTrue(pd.isna(n.loc['2020-06-01','real_index']))
        self.assertAlmostEqual(n.real_index.mean(),100)
        a=app.annual_view(n)
        self.assertEqual(a.valid_months.iloc[0],11)
        self.assertAlmostEqual(a.rer.iloc[0],f.rer.mean())

    def test_daily_lag(self):
        fx=pd.DataFrame({'nominal':[1.2,1.3,1.25]},index=pd.to_datetime(['2020-03-02','2020-03-03','2020-04-01']))
        d=app.make_daily(self.m,fx)
        self.assertEqual(d.cpi_month.tolist(),['2020-01','2020-01','2020-02'])
        self.assertAlmostEqual(d.cpi1.iloc[0],self.m.loc['2020-01-01','cpi1'])

    def test_benchmark_conversion_and_gap_denominators(self):
        f=pd.DataFrame({'cpi1':[110.],'cpi2':[100.],'nominal':[1.],'rer':[1.1]},index=pd.to_datetime(['2020-01-01']))
        f,mean=app.normalize(f)
        b=dict(available=True,value=1.25,low=1.25,high=1.25,end='2010-01-01')
        out=app.add_benchmark(f,b,mean)
        self.assertAlmostEqual(out.implied_nominal.iloc[0],1.25*100/110)
        self.assertAlmostEqual(out.gap_pct.iloc[0],-12)
        self.assertAlmostEqual(out.return_to_reference_pct.iloc[0],100*(1.25/1.1-1))
        self.assertEqual(app.valuation(out,b,5)['status'],'Possibly undervalued')

    def test_unsupported_and_future_benchmarks(self):
        b=app.benchmark(self.m,'ARG','USA',{})
        self.assertFalse(b['available'])
        self.assertFalse(app.valuation(self.m,b,5)['actionable'])
        future=dict(available=True,value=1,low=1,high=1,end='2099-12-01')
        self.assertFalse(app.valuation(self.m,future,5)['actionable'])

    def test_course_reference(self):
        self.assertFalse(app.benchmark(self.m,'GBR','USA',{},course=True)['available'])
        e,_=app.monthly_pair('G163','USA')
        b=app.benchmark(e,'G163','USA',{},course=True)
        if pd.isna(e.loc['1999-01-01','rer']):
            self.assertFalse(b['available'])
            self.assertIn('unavailable',b['reason'])
        else:
            self.assertTrue(b['available'])
            self.assertAlmostEqual(b['value'],e.loc['1999-01-01','rer'])

    def test_same_currency_and_short_horizon(self):
        self.assertIn('no bilateral currency trade',app.recommendation({'actionable':False},'Trader / investor','EUR','EUR','2+ years',True))
        self.assertIn('No short-horizon trade signal',app.recommendation({'actionable':True},'Trader / investor','GBP','USD','Under 2 years'))

    def test_chart_modes_and_axes(self):
        f,mean=app.normalize(self.m.loc['2020-01':'2020-12'])
        for mode in ['Real','Nominal','Both']:
            fig=app.chart(f,mode,'USD per GBP',mean)
            self.assertEqual(len(fig.data),2 if mode=='Both' else 1)
            self.assertEqual(fig.layout.yaxis2.visible,mode=='Both')
        self.assertEqual(app.chart(f,'Both','USD per GBP',mean).data[1].yaxis,'y2')

    def test_duplicates_rejected_and_scale_respected(self):
        raw=pd.DataFrame({'COUNTRY':['USA','USA'],'TIME_PERIOD':['2020-M01']*2,'OBS_VALUE':[10,10],'SCALE':[1,1]})
        with self.assertRaises(ValueError): app.parse_imf(raw,('USA',))
        self.assertEqual(app.parse_imf(raw.iloc[:1],('USA',)).iloc[0,0],100)

if __name__=='__main__': unittest.main()
