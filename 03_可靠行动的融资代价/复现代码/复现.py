"""Run the delivered programs in a temporary copy, preserve published evidence."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,tempfile
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--check-only',action='store_true',help='Check local bounds and message law without full path simulation')
args=parser.parse_args()
out=root/'复现输出';out.mkdir(exist_ok=True)
jobs=['verify_local_and_logs.py'] if args.check_only else ['continuous_closed_loop_20261013.py','verify_local_and_logs.py']
rows=[]
with tempfile.TemporaryDirectory(prefix='temporary_',dir=out) as temp:
    local=Path(temp)/'material'
    shutil.copytree(root,local,ignore=shutil.ignore_patterns('复现输出','__pycache__','*.pyc'))
    for job in jobs:
        proc=subprocess.run([sys.executable,str(local/'复核程序'/job)],cwd=local,capture_output=True,text=True,encoding='utf-8',env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'))
        (out/(job+'.log')).write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
        if proc.returncode:raise RuntimeError(job+' failed; see 复现输出')
        rows.append({'program':job,'exit_code':proc.returncode})
    compared=[]
    files=['局部行动界核验.json','独立消息生成器核验.json','估计器归一化核验.json']
    if not args.check_only:files.append('连续闭环案例.json')
    for name in files:
        same=json.loads((root/name).read_text('utf-8'))==json.loads((local/name).read_text('utf-8'))
        if not same:raise RuntimeError(name+' differs from archive; inspect numerical environment')
        compared.append(name)
result={'passed':True,'check_only':args.check_only,'programs':rows,'archive_comparisons':compared,'published_results_overwritten':False}
(out/'复现核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True))
