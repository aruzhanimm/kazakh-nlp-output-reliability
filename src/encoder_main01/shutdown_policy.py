"""Fail-closed completion/shutdown decisions, separately testable without OS actions."""
def terminal_kind(budget,outcome,completed_present):
    status=budget.get('status')
    if status=='running':return None
    if status not in ['complete','incomplete_budget']:raise ValueError('Main run did not finish safely: '+str(status))
    attempts=budget.get('attempts',[])
    if not attempts or attempts[-1].get('returncode')!=0:raise ValueError('No successful terminal process outcome')
    if attempts[-1].get('status')!=status or outcome.get('status')!=status:raise ValueError('Terminal records disagree')
    if status=='complete' and (not completed_present or outcome['progress']['phase']!='complete'):raise ValueError('Missing explicit completed training state')
    if status=='incomplete_budget' and (completed_present or outcome['progress']['phase'] not in ['train','evaluate']):raise ValueError('Inconsistent graceful incomplete state')
    return status

def grace_ready(elapsed,cancelled,verified):
    if cancelled:return False
    if not verified:raise ValueError('Finalization not verified')
    return elapsed>=60.
