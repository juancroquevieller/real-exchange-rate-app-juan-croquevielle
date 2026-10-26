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

    def test_fred_homework_reproduction_and_inverse(self):
        m,d,_=app.fred_pair('G163','USA')
        b=app.benchmark(m,'G163','USA',{},course=True)
        self.assertAlmostEqual(b['value'],0.4026143434667008,places=8)
        self.assertAlmostEqual(m.loc['2026-09-01','rer'],0.3560979952626863,places=8)
        self.assertAlmostEqual(b['value']*m.loc['2026-09-01','cpi2']/m.loc['2026-09-01','cpi1'],1.3007659112826748,places=8)
        inv,_,_=app.fred_pair('USA','G163')
        joined=m[['nominal','rer']].join(inv[['nominal','rer']],rsuffix='_inv').dropna()
        np.testing.assert_allclose(joined.rer*joined.rer_inv,1)
        np.testing.assert_allclose(joined.nominal*joined.nominal_inv,1)
        self.assertLessEqual(app.make_daily(m,d).dropna(subset=['rer']).index.min(),pd.Timestamp('1999-01-04'))

    def test_extended_daily_history_and_quote(self):
        f,_=app.daily_fx('GBP','USD')
        self.assertLessEqual(f.index.min(),pd.Timestamp('1980-01-01'))
        np.testing.assert_allclose(f.nominal,1/f.fx1_per_usd)
        for cur in ['CLP','PEN']:
            x,_=app.daily_fx(cur,'USD')
            self.assertGreater(len(x),1000)
        euro,_=app.daily_fx('EUR','USD')
        self.assertGreaterEqual(euro.index.min(),pd.Timestamp('1999-01-01'))

    def test_broader_screen_and_no_future_reference(self):
        growth,_=app.growth_data()
        for code in ['CHL','PER','G163']:
            m,_=app.monthly_pair(code,'USA')
            b=app.benchmark(m,code,'USA',growth,as_of='2026-09-30')
            self.assertTrue(b['available'],b.get('reason'))
            self.assertLessEqual(b['end'],'2024-12-01')
            self.assertEqual(b['diagnostics']['months'],60)
            early=app.benchmark(m,code,'USA',growth,as_of='2010-06-01')
            if early['available']: self.assertLessEqual(early['end'],'2009-12-01')

    def test_unstable_data_does_not_gain_benchmark(self):
        idx=pd.date_range('2000-01-01','2019-12-01',freq='MS')
        f=pd.DataFrame({'cpi1':100*1.03**np.arange(len(idx)),'cpi2':100*1.001**np.arange(len(idx)),
                        'rer':np.ones(len(idx))},index=idx)
        g={c:{str(y):2 for y in range(2000,2020)} for c in ['CHL','USA']}
        b=app.benchmark(f,'CHL','USA',g)
        self.assertFalse(b['available'])
        self.assertIn('Inflation',b['reason'])

    def test_raw_rer_chart_preserves_homework_scale(self):
        f,_,_=app.fred_pair('G163','USA')
        b=app.benchmark(f,'G163','USA',{},course=True)
        n,mean=app.normalize(f.loc['1999-01':'2026-09'])
        enriched=app.add_benchmark(n,b,mean)
        fig=app.chart(enriched,'Both','USD per EUR',mean,b,raw=True)
        np.testing.assert_allclose(fig.data[0].y,enriched.rer,equal_nan=True)
        self.assertAlmostEqual(fig.data[2].y[0],b['value'])
        self.assertEqual(fig.layout.yaxis.title.text,'Raw RER')

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

    def test_readable_precision_preserves_small_quotes(self):
        for value,expected in [(1234.56,'1,235'),(950.21,'950'),(1.34,'1.34'),(.75,'0.75'),(.0151,'0.015'),(.00105495,'0.0011'),(.000021,'0.000021')]:
            self.assertEqual(app.number(value,app.display_decimals([value],'nominal')),expected)
        self.assertEqual(app.display_decimals([95,100,105]),1)
        self.assertEqual(app.display_decimals([100.01,100.03,100.09]),2)
        self.assertEqual(app.display_decimals([.35,.4,.45]),2)
        self.assertNotEqual(app.number(.00001,app.display_decimals([.00001])), '0.00')
        f,mean=app.normalize(self.m)
        before=f.copy(deep=True)
        fig=app.chart(f,'Both','USD per GBP',mean)
        self.assertTrue(fig.layout.yaxis.tickformat.startswith(','))
        self.assertTrue(fig.layout.yaxis2.tickformat.startswith(','))
        self.assertIn(',.',fig.data[1].hovertemplate)
        pd.testing.assert_frame_equal(f,before)

    def test_custom_period_anchor_and_missing_inputs(self):
        f,_,_=app.fred_pair('G163','USA')
        one=app.custom_benchmark(f,'1999-01','1999-01',as_of='2026-09-01')
        self.assertTrue(one['available'])
        self.assertEqual(one['months'],1)
        self.assertAlmostEqual(one['value'],.4026143434667008)
        self.assertEqual(one['low'],one['high'])
        for method in ['Median','Mean']:
            b=app.custom_benchmark(f,'2015-01','2019-12',method,as_of='2026-09-01')
            self.assertEqual(b['months'],60)
            expected=getattr(f.loc['2015':'2019','rer'],method.lower())()
            self.assertAlmostEqual(b['value'],expected)
        for start,end,asof,daily in [('2025-09','2025-11','2026-09',False),('2000-02','2000-01','2026-09',False),('2026-08','2026-09','2026-08',False),('1999-01','1999-01','1999-01-15',True)]:
            self.assertFalse(app.custom_benchmark(f,start,end,as_of=asof,daily=daily)['available'])
        self.assertTrue(app.custom_benchmark(f,'1999-01','1999-01',as_of='1999-01-31',daily=True)['available'])

if __name__=='__main__': unittest.main()
