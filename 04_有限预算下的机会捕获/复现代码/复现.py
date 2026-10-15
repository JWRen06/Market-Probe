from pathlib import Path
import subprocess, sys, json
root=Path(__file__).resolve().parent
files=['second_paper_bounded_revision.py', 'second_paper_bounded_independent.py', 'second_paper_bounded_deployment.py', 'second_paper_general_task_separation.py', 'second_paper_geometric_precision.py', 'second_paper_pilot_sphere.py', 'application_contract_bridge.py']
results=[]
for name in files:
    process=subprocess.run([sys.executable,str(root/'experiments'/name)],cwd=root,capture_output=True,text=True,encoding='utf-8')
    (root/(name+'.log')).write_text(process.stdout+'\n'+process.stderr,encoding='utf-8')
    results.append({'script':name,'exit_code':process.returncode})
    if process.returncode: raise RuntimeError(name+' failed; see log')
(root/'复现核验.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=True))
