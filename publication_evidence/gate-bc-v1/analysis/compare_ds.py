"""Compare calculation outputs to the six-decimal published DS review values."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import calculate

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
 'Delta_P':(58.235451,3.838773,53.468986,63.001917),
 'Delta_F':(15.521411,2.826719,12.011576,19.031246),
 'I':(-19.961680,5.882122,-27.265299,-12.658061),
 'Delta_002':(2.722600,1.740615,0.561341,4.883859),
 'P_at_small_F':(68.216291,6.373470,60.302583,76.129999),
 'P_at_large_F':(48.254612,2.480104,45.175156,51.334067),
 'F_at_small_P':(25.502251,1.708900,23.380371,27.624130),
 'F_at_large_P':(5.540571,5.509993,-1.300989,12.382131)}
EXPECTED_CONDITIONS={'B00':(239.634456,1.234881),'B01':(265.136707,0.711087),
 'B10':(307.850747,6.006105),'B11':(313.391318,2.633425),'C_002':(242.357056,2.046916)}


def main():
 result=calculate.calculate(ROOT);comparisons=[]
 for label,expected in EXPECTED.items():
  for field,value in zip(('mean','sample_sd','ci_lower','ci_upper'),expected):
   actual=result['contrasts'][label][field];error=abs(actual-value)
   assert error<=0.000000500001,(label,field,actual,value)
   comparisons.append(dict(label=label,field=field,actual_full_precision=actual,ds_display=value,absolute_error=error))
 for label,expected in EXPECTED_CONDITIONS.items():
  for field,value in zip(('mean','sample_sd'),expected):
   actual=result['conditions'][label][field];assert abs(actual-value)<=0.000000500001
   comparisons.append(dict(label=label,field=field,actual_full_precision=actual,ds_display=value,absolute_error=abs(actual-value)))
 receipt={'status':'PASS','compared_values':len(comparisons),'absolute_tolerance':0.000000500001,
          'reference':'DS_SCIENTIFIC_DECISION.md, six-decimal published values',
          'precision_limit':'Independent unrounded DS machine output not supplied; no claim of bitwise equality to it.',
          'computation':'full-precision accepted CSV floats, frozen paired t formula','comparisons':comparisons}
 (ROOT/'tables/DS_COMPARISON.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('DS_COMPARISON_PASS',len(comparisons),'published values')


if __name__=='__main__':main()
